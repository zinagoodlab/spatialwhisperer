#!/usr/bin/env python3
"""
Diff the recheck (`<model>_recheck/...`) score CSVs against the canonical
ones. Prints a summary table of (model, benchmark) → max abs diff per
score column, mean abs diff, and the macro-AUROC of the recheck minus the
canonical macro-AUROC.

Run on Sherlock after the recheck controller (sbatch 24043918) completes.

Usage:
    cd /home/groups/zinaida/moritzs/cellwhisperer_private
    conda run -n cellwhisperer python \
        analysis/experiments/two_stage_baseline/compare_recheck.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PD = Path("/home/groups/zinaida/moritzs/cellwhisperer_private")
RESULTS = PD / "results" / "pathocell_evaluation"

# Map: (model_canonical_dir, benchmark, score_subdir-relative-to-model)
PAIRS = [
    # PathoCell CRC for all 6 models
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m", "PathoCell_CRC", ""),
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k", "PathoCell_CRC", ""),
    ("spotwhisperer_quilt1m", "PathoCell_CRC", ""),
    ("two_stage_baseline", "PathoCell_CRC", ""),
    ("omiclip", "PathoCell_CRC", ""),
    ("omiclip_pseudobulk", "PathoCell_CRC", ""),
    # Bimodal bridge on Lizard / PanNuke
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k", "Lizard", "lizard"),
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k", "PanNuke", "pannuke"),
]


def load(d: Path):
    out = {}
    for f in sorted(d.glob("*_patch_scores_seed0.csv")):
        out[f.name] = pd.read_csv(f)
    return out


def diff_one(canonical: Path, recheck: Path):
    a = load(canonical)
    b = load(recheck)
    common = sorted(set(a) & set(b))
    only_a = set(a) - set(b)
    only_b = set(b) - set(a)
    if not common:
        return {
            "n_common": 0,
            "only_canonical": len(only_a),
            "only_recheck": len(only_b),
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
        }
    max_abs = 0.0
    mean_abs = []
    for k in common:
        df_a, df_b = a[k], b[k]
        cols = [c for c in df_a.columns if c in df_b.columns]
        if not cols:
            continue
        diff = (df_a[cols].values - df_b[cols].values).astype(float)
        max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
        mean_abs.append(float(np.nanmean(np.abs(diff))))
    return {
        "n_common": len(common),
        "only_canonical": len(only_a),
        "only_recheck": len(only_b),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean(mean_abs)) if mean_abs else float("nan"),
    }


if __name__ == "__main__":
    rows = []
    for model, bench, sub in PAIRS:
        canonical = RESULTS / model / sub
        recheck = RESULTS / f"{model}_recheck" / sub
        if not recheck.exists():
            print(f"  {model:60s} {bench:13s}: recheck dir missing — skipping")
            continue
        r = diff_one(canonical, recheck)
        rows.append({"model": model, "benchmark": bench, **r})
        print(
            f"  {model[:60]:60s} {bench:13s}: "
            f"common={r['n_common']:3d}  max|Δ|={r['max_abs_diff']:.6g}  "
            f"mean|Δ|={r['mean_abs_diff']:.6g}"
        )

    df = pd.DataFrame(rows)
    out_path = (
        RESULTS / "comparison" / "patch" / "tables" / "recheck_diff_summary.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    n_perfect = (df["max_abs_diff"] < 1e-6).sum()
    print(f"\nPerfect matches (max|Δ| < 1e-6): {n_perfect}/{len(df)}")
    n_close = (df["max_abs_diff"] < 1e-3).sum()
    print(f"Close matches (max|Δ| < 1e-3): {n_close}/{len(df)}")
