#!/usr/bin/env python3
"""
Compute PLIP/CONCH/MUSK AUROC with Table 2 methodology:
  - Presence-based binary labels (true_probs > 0)
  - Classes → datasets → mean aggregation
  - Reduced class schemes for Lizard (3-class) and PanNuke (4-class)
  - CRC: 13-class excl Background + Other cells

Usage (on Sherlock):
    python compute_baselines_table2_style.py
"""

import os
import re
import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Paths — relative to the project root (analysis/scripts/<file> → parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent.parent / "static" / "baselines_animesh"
H5AD_BASE = PROJECT_ROOT / "resources/pathocell/processed"
OUT_DIR = PROJECT_ROOT / "results/pathocell_evaluation/seed_variance"

# Column name mappings from baseline CSVs to h5ad class names
LIZARD_COL_TO_CLASS = {
    "neutrophil": "Neutrophil",
    "epithelial": "Epithelial",
    "lymphocyte": "Lymphocyte",
    "plasma": "Plasma",
    "eosinophil": "Eosinophil",
    "fibroblast": "Connective tissue",
}
PANNUKE_COL_TO_CLASS = {
    "epithelial": "Epithelial",
    "dead cells": "Dead Cells",
    "fibroblasts": "Connective/Soft tissue cells",
    "leukocytes": "Inflammatory",
    "cancer cells": "Neoplastic cells",
}


# CRC (pathocellbench) columns use "A sample of X cells" format
# Map to h5ad class names by stripping "A sample of " and " cells"
def crc_col_to_class(col):
    m = re.match(r"A sample of (.+?)( cells)?$", col)
    return m.group(1) if m else col


# Reduced class configs
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

BASELINES = ["conch", "plip"]  # MUSK dropped (not part of the published paper)
TERMS = os.environ.get("BASELINE_TERMS", "terms2")  # override via env: BASELINE_TERMS=terms1


def load_gt(h5ad_dir, pattern="*_patch.h5ad"):
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


def load_baseline_csv(csv_path):
    """Load baseline CSV and split into per-sample dict of DataFrames."""
    df = pd.read_csv(csv_path)
    if "source_image" in df.columns:
        df = df[df["source_image"].str.contains("_patch.tiff")].copy()
    scores_by_sample = {}
    for src_img, group in df.groupby("source_image"):
        sample = src_img.replace("_patch.tiff", "")
        # Extract sample ID (e.g., reg001_A from reg001_A_patch.tiff)
        m = re.search(r"(reg\d+_[A-Z])", sample)
        if m:
            sample = m.group(1)
        else:
            sample = sample.replace("_patch", "")
        id_cols = {"source_image", "spot_id", "dataset_id"}
        class_cols = [c for c in group.columns if c not in id_cols]
        scores_by_sample[sample] = group[class_cols].reset_index(drop=True)
    return scores_by_sample


def presence_auroc(scores_col, true_probs_col):
    y_bin = (true_probs_col > 0).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    return roc_auc_score(y_bin, scores_col)


def eval_crc(scores_by_sample, gt_by_sample, col_to_class_fn):
    first_sc = next(iter(scores_by_sample.values()))
    sc_cols = list(first_sc.columns)
    # Map score columns to h5ad class names
    col_map = {c: col_to_class_fn(c) for c in sc_cols}
    h5ad_classes_needed = set(col_map.values()) - CRC_EXCLUDE - {"Background"}

    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        true_probs = info["true_probs"]
        gt_classes = info["classes"]

        aucs = {}
        for sc_col in sc_cols:
            cls = col_map[sc_col]
            if cls in CRC_EXCLUDE or cls.lower() == "background":
                continue
            if cls not in gt_classes:
                continue
            j = gt_classes.index(cls)
            aucs[cls] = presence_auroc(sc[sc_col].values, true_probs[:, j])
        per_ds_class.append(aucs)

    return _aggregate(
        per_ds_class,
        sorted(
            h5ad_classes_needed
            & set(gt_by_sample[list(gt_by_sample.keys())[0]]["classes"])
        ),
    )


def eval_lizard(scores_by_sample, gt_by_sample, col_to_class):
    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig_classes = info["classes"]
        counts = info["counts"]

        # Rename columns to h5ad class names
        sc_renamed = sc.rename(columns=col_to_class)

        # Build merged counts
        epi_idx = orig_classes.index("Epithelial")
        fibro_idx = orig_classes.index("Connective tissue")
        leuk_indices = [
            orig_classes.index(c) for c in LIZARD_LEUKOCYTE_ORIG if c in orig_classes
        ]
        merged_counts = np.column_stack(
            [
                counts[:, epi_idx],
                counts[:, leuk_indices].sum(axis=1),
                counts[:, fibro_idx],
            ]
        )
        merged_probs = merged_counts / (
            merged_counts.sum(axis=1, keepdims=True) + 1e-12
        )

        # Build merged scores
        epi_score = sc_renamed["Epithelial"].values
        leuk_score = sum(
            sc_renamed[c].values
            for c in LIZARD_LEUKOCYTE_ORIG
            if c in sc_renamed.columns
        )
        fibro_score = sc_renamed["Connective tissue"].values

        # Filter Plasma-dominant patches
        dominant = np.array(orig_classes)[counts.argmax(axis=1)]
        keep = np.array([d not in LIZARD_DROP for d in dominant])

        aucs = {
            "Epithelial": presence_auroc(epi_score[keep], merged_probs[keep, 0]),
            "Leukocyte": presence_auroc(leuk_score[keep], merged_probs[keep, 1]),
            "Fibroblast": presence_auroc(fibro_score[keep], merged_probs[keep, 2]),
        }
        per_ds_class.append(aucs)

    return _aggregate(per_ds_class, LIZARD_REDUCED_CLASSES)


def eval_pannuke(scores_by_sample, gt_by_sample, col_to_class):
    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig_classes = info["classes"]
        counts = info["counts"]

        sc_renamed = sc.rename(columns=col_to_class)

        keep_classes = [c for c in orig_classes if c not in PANNUKE_DROP]
        keep_idx = [orig_classes.index(c) for c in keep_classes]
        kept_counts = counts[:, keep_idx]
        kept_probs = kept_counts / (kept_counts.sum(axis=1, keepdims=True) + 1e-12)

        dominant = np.array(orig_classes)[counts.argmax(axis=1)]
        keep = np.array([d not in PANNUKE_DROP for d in dominant])

        aucs = {}
        for j, cls in enumerate(keep_classes):
            if cls in sc_renamed.columns:
                aucs[cls] = presence_auroc(
                    sc_renamed[cls].values[keep], kept_probs[keep, j]
                )
        per_ds_class.append(aucs)

    return _aggregate(per_ds_class, PANNUKE_REDUCED_CLASSES)


def _aggregate(per_ds_class, class_names):
    if not per_ds_class:
        return np.nan, {}
    per_class_mean = {}
    for cls in class_names:
        vals = [
            d[cls]
            for d in per_ds_class
            if cls in d and not np.isnan(d.get(cls, np.nan))
        ]
        per_class_mean[cls] = np.mean(vals) if vals else np.nan
    valid = [v for v in per_class_mean.values() if not np.isnan(v)]
    return (np.mean(valid) if valid else np.nan), per_class_mean


if __name__ == "__main__":
    print("Loading ground truth...")
    lizard_gt = load_gt(H5AD_BASE / "lizard")
    pannuke_gt = load_gt(H5AD_BASE / "pannuke")
    crc_gt = load_gt(H5AD_BASE)

    rows = []
    per_class_rows = []

    for baseline in BASELINES:
        print(f"\n{'=' * 60}")
        print(f"  {baseline.upper()}")
        print(f"{'=' * 60}")

        # CRC
        crc_csv = STATIC_DIR / f"{baseline}_logits_{TERMS}.csv"
        if crc_csv.exists():
            scores = load_baseline_csv(crc_csv)
            if scores:
                macro, pc = eval_crc(scores, crc_gt, crc_col_to_class)
                print(f"  CRC 13-class: AUROC = {macro:.4f}")
                rows.append(
                    {
                        "model": baseline.upper(),
                        "benchmark": "CRC_13class",
                        "auroc": macro,
                    }
                )
                for cls, auc in pc.items():
                    per_class_rows.append({"model": baseline.upper(), "benchmark": "CRC", "class": cls, "auroc": auc})

        # Lizard
        liz_csv = STATIC_DIR / "lizard" / f"{baseline}_logits_lizard_{TERMS}.csv"
        if liz_csv.exists():
            scores = load_baseline_csv(liz_csv)
            if scores:
                macro, pc = eval_lizard(scores, lizard_gt, LIZARD_COL_TO_CLASS)
                print(f"  Lizard 3-class: AUROC = {macro:.4f}")
                for cls, auc in pc.items():
                    print(f"    {cls:20s}: {auc:.4f}")
                rows.append(
                    {
                        "model": baseline.upper(),
                        "benchmark": "Lizard_3class",
                        "auroc": macro,
                    }
                )
                for cls, auc in pc.items():
                    per_class_rows.append({"model": baseline.upper(), "benchmark": "Lizard", "class": cls, "auroc": auc})

        # PanNuke
        pan_csv = STATIC_DIR / "pannuke" / f"{baseline}_logits_pannuke_{TERMS}.csv"
        if pan_csv.exists():
            scores = load_baseline_csv(pan_csv)
            if scores:
                macro, pc = eval_pannuke(scores, pannuke_gt, PANNUKE_COL_TO_CLASS)
                print(f"  PanNuke 4-class: AUROC = {macro:.4f}")
                for cls, auc in pc.items():
                    print(f"    {cls:35s}: {auc:.4f}")
                rows.append(
                    {
                        "model": baseline.upper(),
                        "benchmark": "PanNuke_4class",
                        "auroc": macro,
                    }
                )
                for cls, auc in pc.items():
                    per_class_rows.append({"model": baseline.upper(), "benchmark": "PanNuke", "class": cls, "auroc": auc})

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print("SUMMARY (Table 2 style: presence-based AUROC, classes → datasets → mean)")
    print(f"{'=' * 60}")
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        print(f"\n{bench}:")
        for _, row in sub.iterrows():
            print(f"  {row['model']:10s}: AUROC = {row['auroc']:.4f}")

    out_path = OUT_DIR / f"baselines_table2_style_{TERMS}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    pc_df = pd.DataFrame(per_class_rows)
    pc_path = OUT_DIR / f"baselines_table2_style_per_class_{TERMS}.csv"
    pc_df.to_csv(pc_path, index=False)
    print(f"Saved per-class to {pc_path}")
