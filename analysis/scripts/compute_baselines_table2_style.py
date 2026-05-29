"""
Compute PLIP/CONCH AUROC with Table 2 methodology:
  - Presence-based binary labels (true_probs > 0)
  - Classes → datasets → mean aggregation
  - Reduced class schemes for Lizard (3-class) and PanNuke (4-class)
  - CRC: 13-class excl Background + Other cells

Inputs (Snakemake):
- snakemake.input.crc_csvs: list of {baseline}_logits_{terms_id}.csv (one per baseline)
- snakemake.input.lizard_csvs: list of lizard/{baseline}_logits_lizard_{terms_id}.csv
- snakemake.input.pannuke_csvs: list of pannuke/{baseline}_logits_pannuke_{terms_id}.csv
- snakemake.input.crc_gt: list of CRC patch h5ads
- snakemake.input.lizard_gt: list of Lizard patch h5ads
- snakemake.input.pannuke_gt: list of PanNuke patch h5ads
- snakemake.params.baselines: list of baseline names matching the *_csvs lists order

Outputs:
- snakemake.output.macro: summary CSV (baseline x benchmark x auroc)
- snakemake.output.per_class: per-class breakdown CSV
"""

import re
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def crc_col_to_class(col):
    m = re.match(r"A sample of (.+?)( cells)?$", col)
    return m.group(1) if m else col


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


def load_gt(h5ad_files):
    gt = {}
    for f in sorted(map(str, h5ad_files)):
        sample = Path(f).stem.replace("_patch", "")
        adata = anndata.read_h5ad(f)
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
    df = pd.read_csv(csv_path)
    if "source_image" in df.columns:
        df = df[df["source_image"].str.contains("_patch.tiff")].copy()
    scores_by_sample = {}
    for src_img, group in df.groupby("source_image"):
        sample = src_img.replace("_patch.tiff", "")
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

        sc_renamed = sc.rename(columns=col_to_class)

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

        epi_score = sc_renamed["Epithelial"].values
        leuk_score = sum(
            sc_renamed[c].values
            for c in LIZARD_LEUKOCYTE_ORIG
            if c in sc_renamed.columns
        )
        fibro_score = sc_renamed["Connective tissue"].values

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


baselines = list(snakemake.params.baselines)
crc_csvs = [Path(p) for p in snakemake.input.crc_csvs]
lizard_csvs = [Path(p) for p in snakemake.input.lizard_csvs]
pannuke_csvs = [Path(p) for p in snakemake.input.pannuke_csvs]
assert len(crc_csvs) == len(lizard_csvs) == len(pannuke_csvs) == len(baselines)

lizard_gt = load_gt(snakemake.input.lizard_gt)
pannuke_gt = load_gt(snakemake.input.pannuke_gt)
crc_gt = load_gt(snakemake.input.crc_gt)

rows = []
per_class_rows = []

for baseline, crc_csv, liz_csv, pan_csv in zip(
    baselines, crc_csvs, lizard_csvs, pannuke_csvs
):
    crc_scores = load_baseline_csv(crc_csv)
    macro, pc = eval_crc(crc_scores, crc_gt, crc_col_to_class)
    rows.append(
        {"model": baseline.upper(), "benchmark": "CRC_13class", "auroc": macro}
    )
    for cls, auc in pc.items():
        per_class_rows.append(
            {"model": baseline.upper(), "benchmark": "CRC", "class": cls, "auroc": auc}
        )

    liz_scores = load_baseline_csv(liz_csv)
    macro, pc = eval_lizard(liz_scores, lizard_gt, LIZARD_COL_TO_CLASS)
    rows.append(
        {"model": baseline.upper(), "benchmark": "Lizard_3class", "auroc": macro}
    )
    for cls, auc in pc.items():
        per_class_rows.append(
            {
                "model": baseline.upper(),
                "benchmark": "Lizard",
                "class": cls,
                "auroc": auc,
            }
        )

    pan_scores = load_baseline_csv(pan_csv)
    macro, pc = eval_pannuke(pan_scores, pannuke_gt, PANNUKE_COL_TO_CLASS)
    rows.append(
        {"model": baseline.upper(), "benchmark": "PanNuke_4class", "auroc": macro}
    )
    for cls, auc in pc.items():
        per_class_rows.append(
            {
                "model": baseline.upper(),
                "benchmark": "PanNuke",
                "class": cls,
                "auroc": auc,
            }
        )

pd.DataFrame(rows).to_csv(snakemake.output.macro, index=False)
pd.DataFrame(per_class_rows).to_csv(snakemake.output.per_class, index=False)
