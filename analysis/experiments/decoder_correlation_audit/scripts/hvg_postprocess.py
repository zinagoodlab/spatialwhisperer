"""
Post-hoc HVG-restricted per-gene Pearson on the existing decoder checkpoint
output (no retraining, no GPU). Uses the per_gene.parquet from the main audit
and the val target matrix from per_gene (val_mean, val_var) plus the per-spot
predictions and targets reconstructed from per_spot.parquet → no, the per-spot
parquet doesn't keep raw matrices. So for HVG selection we use scanpy's "seurat"
flavor on a reconstructed AnnData of the val target counts (recovered from the
log1p values stored in per_gene + per_spot? — not stored either).

A pragmatic alternative: HVG ranking by dispersion (val_var / val_mean) on the
gene-level summary statistics already in per_gene.parquet. We additionally
contrast the rank with the MT-gene set to make the cellularity confound
explicit. We then bootstrap the median per-gene r in each HVG bucket.

Outputs land alongside the main audit:
  hvg_per_gene_summary.json  - bootstrap medians (±95% CI) for each HVG cut
  hvg_per_gene_table.csv     - tabular form for the manuscript
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-gene", required=True, help="per_gene.parquet from main audit")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--min-mean",
        type=float,
        default=0.05,
        help="drop genes with val_mean < this (near-zero in val)",
    )
    p.add_argument(
        "--top-ns",
        type=int,
        nargs="+",
        default=[50, 500, 2000],
        help="HVG cuts to report",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="bootstrap iterations for median+CI (DeepSpot-style)",
    )
    return p.parse_args()


def bootstrap_median(values: np.ndarray, n_iter: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    medians = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)
    return {
        "median": float(np.median(values)),
        "bootstrap_median_mean": float(medians.mean()),
        "ci_low": float(np.quantile(medians, 0.025)),
        "ci_high": float(np.quantile(medians, 0.975)),
        "n": int(n),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gd = pd.read_parquet(args.per_gene)

    # Filter to genes with non-trivial val expression
    gd = gd.copy()
    gd["is_mt"] = gd["gene_name"].str.startswith("MT-")
    expressed = gd[gd["val_mean"] >= args.min_mean].copy()
    # Dispersion = val_var / (val_mean+eps); a Seurat-style HVG proxy on log1p data
    expressed["dispersion"] = expressed["val_var"] / (expressed["val_mean"] + 1e-9)
    expressed_sorted = expressed.sort_values("dispersion", ascending=False)

    rows = []
    summary = {
        "n_genes_total": int(len(gd)),
        "n_genes_expressed_min_mean_%.3f" % args.min_mean: int(len(expressed)),
        "n_mt_genes_expressed": int(expressed["is_mt"].sum()),
        "cuts": {},
    }

    # Reference: all expressed genes
    all_stats = bootstrap_median(
        expressed["per_gene_r_model"].dropna().values, args.n_bootstrap
    )
    rows.append({"hvg_cut": "all_expressed", "filter": "min_mean", **all_stats})

    for n in args.top_ns:
        top = expressed_sorted.head(n)
        top_no_mt = expressed_sorted[~expressed_sorted["is_mt"]].head(n)

        for label, df in [("top%d_HVG" % n, top), ("top%d_HVG_noMT" % n, top_no_mt)]:
            vals = df["per_gene_r_model"].dropna().values
            stats = bootstrap_median(vals, args.n_bootstrap)
            mt_count = int(df["is_mt"].sum())
            rows.append(
                {
                    "hvg_cut": label,
                    "filter": "ranked by val_var/val_mean dispersion",
                    "n_in_set_mt": mt_count,
                    **stats,
                }
            )

        summary["cuts"]["top%d_HVG" % n] = {
            "with_mt": {
                "n": int(len(top)),
                "n_mt": int(top["is_mt"].sum()),
                "median_r": float(top["per_gene_r_model"].median()),
            },
            "without_mt": {
                "n": int(len(top_no_mt)),
                "median_r": float(top_no_mt["per_gene_r_model"].median()),
            },
        }

    # MT-only: how concentrated is the inflation?
    mt = expressed[expressed["is_mt"]]
    mt_stats = bootstrap_median(mt["per_gene_r_model"].dropna().values, args.n_bootstrap)
    rows.append({"hvg_cut": "MT_only", "filter": "MT- prefix", **mt_stats})
    summary["mt_block"] = {
        "n_mt_genes": int(len(mt)),
        "median_r": float(mt["per_gene_r_model"].median()),
        "mean_r": float(mt["per_gene_r_model"].mean()),
    }

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "hvg_per_gene_table.csv", index=False)
    with open(out_dir / "hvg_per_gene_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== HVG post-hoc per-gene Pearson (median, bootstrap 95% CI) ===")
    print(df_out.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
