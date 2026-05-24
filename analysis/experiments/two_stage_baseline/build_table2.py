#!/usr/bin/env python3
"""
Build Table 2 (and per-class appendix table) for the rebuttal "Two-stage
pipeline baselines" analysis. Six methods × three benchmarks at seed=0:

  Methods:  Trimodal (ours), Bimodal bridge T↔G+G↔I, Bimodal I↔T (Quilt-1M),
            Two-stage UNI2→GF (ours), OmiCLIP short markers,
            OmiCLIP extended (pseudobulk) markers
  Benchmarks: PathoCell CRC (13-class), Lizard (3-class reduced),
              PanNuke (4-class reduced)

Mirrors the presence-based AUROC aggregation in
`compute_reduced_class_table2_style.py` (CLASSES → DATASETS → MEAN), which is
what the existing manuscript Table 1 uses.

Outputs:
  results/pathocell_evaluation/comparison/patch/tables/table2_two_stage_baselines.csv
  results/pathocell_evaluation/comparison/patch/tables/table2_two_stage_baselines_per_class.csv

Usage (on Sherlock, after the controller job completes):
  conda run -n cellwhisperer python build_table2.py
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PD = Path("/home/groups/zinaida/moritzs/cellwhisperer_private")
RESULTS = PD / "results" / "pathocell_evaluation"
H5AD_BASE = PD / "resources" / "pathocell" / "processed"
OUT_DIR = RESULTS / "comparison" / "patch" / "tables"

# Six methods, ordered for the manuscript table.
METHODS = [
    ("Trimodal (ours)", "spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"),
    ("Bimodal bridge T↔G+G↔I (ours)", "spotwhisperer_cellxgene_census__archs4_geo__hest1k"),
    ("Bimodal I↔T (Quilt-1M only)", "spotwhisperer_quilt1m"),
    ("Two-stage UNI2→GF (ours)", "two_stage_baseline"),
    ("OmiCLIP, short marker list", "omiclip"),
    ("OmiCLIP, extended marker list", "omiclip_pseudobulk"),
]

# ── Reduced-class config (matches compute_reduced_class_table2_style.py) ──────
LIZARD_LEUKOCYTE_ORIG = {"Neutrophil", "Lymphocyte", "Eosinophil"}
LIZARD_DROP = {"Plasma"}
LIZARD_REDUCED_CLASSES = ["Epithelial", "Leukocyte", "Fibroblast"]

PANNUKE_DROP = {"Dead Cells"}
PANNUKE_REDUCED_CLASSES = [
    "Epithelial",
    "Connective/Soft tissue cells",
    "Inflammatory",
    "Neoplastic cells",
]

CRC_EXCLUDE = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}


def load_gt_for_dir(h5ad_dir: Path, pattern: str = "*_patch.h5ad"):
    gt = {}
    for f in sorted(h5ad_dir.glob(pattern)):
        sample = f.stem.replace("_patch", "")
        adata = anndata.read_h5ad(str(f))
        counts_df = adata.obsm["cell_type_counts_coarse"]
        non_bg = [c for c in counts_df.columns if c.lower() != "background"]
        counts_nobg = counts_df[non_bg]
        counts_arr = counts_nobg.values.astype(float)
        true_probs = counts_arr / (counts_arr.sum(axis=1, keepdims=True) + 1e-12)
        gt[sample] = {
            "classes": list(counts_nobg.columns),
            "counts": counts_arr,
            "true_probs": true_probs,
        }
    return gt


def load_scores(score_dir: Path, seed: str = "seed0"):
    out = {}
    for f in sorted(score_dir.glob(f"*_scores_{seed}.csv")):
        sample = f.stem.replace(f"_patch_scores_{seed}", "")
        out[sample] = pd.read_csv(f)
    return out


def presence_auroc(scores_col, true_probs_col):
    y_bin = (true_probs_col > 0).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    return roc_auc_score(y_bin, scores_col)


def crc_col_to_class(col: str) -> str:
    # CRC baseline CSVs sometimes use "A sample of X cells" naming; the score
    # CSVs from our model rules use the bare class name. Strip the wrapper.
    import re

    m = re.match(r"A sample of (.+?)( cells)?$", col)
    return m.group(1) if m else col


def auroc_crc(scores_by_sample, gt_by_sample):
    first_info = next(iter(gt_by_sample.values()))
    keep = [
        c
        for c in first_info["classes"]
        if c not in CRC_EXCLUDE and c.lower() != "background"
    ]
    keep_idx = {c: first_info["classes"].index(c) for c in keep}
    per_ds_class = []
    for sample in sorted(gt_by_sample):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        sc.columns = [crc_col_to_class(c) for c in sc.columns]
        true_probs = info["true_probs"]
        aucs = {}
        for cls in keep:
            j = keep_idx[cls]
            if cls not in sc.columns:
                continue
            aucs[cls] = presence_auroc(sc[cls].values, true_probs[:, j])
        per_ds_class.append(aucs)
    return _aggregate(per_ds_class, keep)


def auroc_lizard(scores_by_sample, gt_by_sample):
    per_ds_class = []
    for sample in sorted(gt_by_sample):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig = info["classes"]
        counts = info["counts"]

        epi_idx = orig.index("Epithelial")
        fibro_idx = orig.index("Connective tissue")
        leuk_indices = [orig.index(c) for c in LIZARD_LEUKOCYTE_ORIG if c in orig]
        merged = np.column_stack(
            [
                counts[:, epi_idx],
                counts[:, leuk_indices].sum(axis=1),
                counts[:, fibro_idx],
            ]
        )
        merged_probs = merged / (merged.sum(axis=1, keepdims=True) + 1e-12)

        epi_score = sc.get("Epithelial", pd.Series(np.zeros(len(sc)))).values
        leuk_score = sum(
            sc[c].values for c in LIZARD_LEUKOCYTE_ORIG if c in sc.columns
        )
        if isinstance(leuk_score, int):
            leuk_score = np.zeros(len(sc))
        fibro_score = sc.get("Connective tissue", pd.Series(np.zeros(len(sc)))).values

        dominant = np.array(orig)[counts.argmax(axis=1)]
        keep_mask = np.array([d not in LIZARD_DROP for d in dominant])

        aucs = {
            "Epithelial": presence_auroc(epi_score[keep_mask], merged_probs[keep_mask, 0]),
            "Leukocyte": presence_auroc(leuk_score[keep_mask], merged_probs[keep_mask, 1]),
            "Fibroblast": presence_auroc(fibro_score[keep_mask], merged_probs[keep_mask, 2]),
        }
        per_ds_class.append(aucs)
    return _aggregate(per_ds_class, LIZARD_REDUCED_CLASSES)


def auroc_pannuke(scores_by_sample, gt_by_sample):
    per_ds_class = []
    for sample in sorted(gt_by_sample):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig = info["classes"]
        counts = info["counts"]

        keep_classes = [c for c in orig if c not in PANNUKE_DROP]
        keep_idx = [orig.index(c) for c in keep_classes]
        kept_counts = counts[:, keep_idx]
        kept_probs = kept_counts / (kept_counts.sum(axis=1, keepdims=True) + 1e-12)

        dominant = np.array(orig)[counts.argmax(axis=1)]
        keep_mask = np.array([d not in PANNUKE_DROP for d in dominant])

        aucs = {}
        for j, cls in enumerate(keep_classes):
            if cls not in sc.columns:
                continue
            aucs[cls] = presence_auroc(sc[cls].values[keep_mask], kept_probs[keep_mask, j])
        per_ds_class.append(aucs)
    return _aggregate(per_ds_class, PANNUKE_REDUCED_CLASSES)


def _aggregate(per_ds_class, class_names):
    if not per_ds_class:
        return np.nan, {}
    pc = {}
    for cls in class_names:
        vals = [
            d[cls] for d in per_ds_class
            if cls in d and not np.isnan(d.get(cls, np.nan))
        ]
        pc[cls] = float(np.mean(vals)) if vals else np.nan
    valid = [v for v in pc.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else np.nan
    return macro, pc


if __name__ == "__main__":
    print("Loading ground truth …")
    crc_gt = load_gt_for_dir(H5AD_BASE)
    lizard_gt = load_gt_for_dir(H5AD_BASE / "lizard")
    pannuke_gt = load_gt_for_dir(H5AD_BASE / "pannuke")

    rows = []
    pc_rows = []

    for label, model_dir_name in METHODS:
        model_dir = RESULTS / model_dir_name
        print(f"\n=== {label}  ({model_dir_name}) ===")

        # CRC scores live at <model_dir>/<dataset>_patch_scores_seed0.csv
        crc_scores = load_scores(model_dir)
        if crc_scores:
            macro, pc = auroc_crc(crc_scores, crc_gt)
            print(f"  PathoCell CRC: AUROC = {macro:.4f}")
            rows.append({"method": label, "benchmark": "PathoCell_CRC", "auroc": macro})
            for c, v in pc.items():
                pc_rows.append({"method": label, "benchmark": "PathoCell_CRC", "class": c, "auroc": v})
        else:
            print("  PathoCell CRC: no scores")

        for benchmark, scorer, gt in (
            ("Lizard", auroc_lizard, lizard_gt),
            ("PanNuke", auroc_pannuke, pannuke_gt),
        ):
            sub = model_dir / benchmark.lower()
            if not sub.exists():
                print(f"  {benchmark}: no scores")
                continue
            scores = load_scores(sub)
            if not scores:
                print(f"  {benchmark}: no scores")
                continue
            macro, pc = scorer(scores, gt)
            print(f"  {benchmark}: AUROC = {macro:.4f}")
            rows.append({"method": label, "benchmark": benchmark, "auroc": macro})
            for c, v in pc.items():
                pc_rows.append({"method": label, "benchmark": benchmark, "class": c, "auroc": v})

    df = pd.DataFrame(rows)
    pc_df = pd.DataFrame(pc_rows)

    print("\n=== Final Table 2 (presence-based AUROC, classes → datasets → mean) ===")
    pivot = df.pivot(index="method", columns="benchmark", values="auroc").reindex(
        [m[0] for m in METHODS]
    )
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}" if not np.isnan(v) else "—"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "table2_two_stage_baselines.csv"
    out_pc_csv = OUT_DIR / "table2_two_stage_baselines_per_class.csv"
    pivot.to_csv(out_csv)
    pc_df.to_csv(out_pc_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(f"Saved {out_pc_csv}")
