"""
Compute reduced-class AUROC with Table 2 aggregation strategy:
  1. For each (dataset, class): compute one-vs-rest AUROC using presence-based binary labels
  2. Average across datasets within each class → per-class AUROC
  3. Average across classes → final macro AUROC

Presence-based: a patch is "positive" for class j if it contains ANY cells of class j
(true_probs[:, j] > 0).

Inputs (Snakemake):
- snakemake.input.crc_scores, lizard_scores, pannuke_scores: per-dataset score CSVs
  under PATHOCELL_RESULTS/{model}/{[lizard|pannuke]/}{ds}_patch_scores_seed0.csv
- snakemake.input.crc_gt, lizard_gt, pannuke_gt: corresponding patch h5ads
- snakemake.params.models: list of model dirnames (one per training seed)
- snakemake.params.seed_labels: human-readable label per model (same length/order)

Outputs:
- snakemake.output.macro: seed x benchmark x auroc summary CSV
- snakemake.output.per_class: per-class breakdown CSV
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def load_gt_files(h5ad_files):
    gt = {}
    for f in sorted(map(str, h5ad_files)):
        sample = Path(f).stem.replace("_patch", "")
        adata = anndata.read_h5ad(f)
        counts_df = adata.obsm["cell_type_counts_coarse"]
        non_bg = [c for c in counts_df.columns if c.lower() != "background"]
        counts_nobg = counts_df[non_bg]
        counts_arr = counts_nobg.values.astype(float)
        true_probs = counts_arr / (counts_arr.sum(axis=1, keepdims=True) + 1e-12)
        gt_labels = list(counts_nobg.columns[counts_arr.argmax(axis=1)])
        gt[sample] = {
            "classes": list(counts_nobg.columns),
            "counts": counts_arr,
            "true_probs": true_probs,
            "gt_labels": gt_labels,
        }
    return gt


def scores_by_model(score_files, models):
    """Split a flat list of score CSVs into a dict keyed by model dirname."""
    groups = {m: {} for m in models}
    for f in map(Path, score_files):
        sample = f.stem.replace("_patch_scores_seed0", "")
        for m in models:
            if f"/{m}/" in str(f):
                groups[m][sample] = pd.read_csv(f)
                break
    return groups


def presence_auroc(scores_col, true_probs_col):
    y_bin = (true_probs_col > 0).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    return roc_auc_score(y_bin, scores_col)


def table2_auroc_crc(scores_by_sample, gt_by_sample):
    first_info = next(iter(gt_by_sample.values()))
    all_classes = first_info["classes"]
    keep_classes = [
        c for c in all_classes if c not in CRC_EXCLUDE and c.lower() != "background"
    ]
    keep_idx = {c: all_classes.index(c) for c in keep_classes}

    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        true_probs = info["true_probs"]

        aucs = {}
        for cls in keep_classes:
            j = keep_idx[cls]
            s = sc[cls].values if cls in sc.columns else np.zeros(len(true_probs))
            aucs[cls] = presence_auroc(s, true_probs[:, j])
        per_ds_class.append(aucs)

    return _aggregate(per_ds_class, keep_classes)


def table2_auroc_lizard(scores_by_sample, gt_by_sample):
    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig_classes = info["classes"]
        counts = info["counts"]

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

        epi_score = sc["Epithelial"].values
        leuk_score = sum(sc[c].values for c in LIZARD_LEUKOCYTE_ORIG if c in sc.columns)
        fibro_score = sc["Connective tissue"].values
        merged_scores = np.column_stack([epi_score, leuk_score, fibro_score])

        dominant = np.array(orig_classes)[counts.argmax(axis=1)]
        keep_mask = np.array([d not in LIZARD_DROP for d in dominant])

        aucs = {}
        for j, cls in enumerate(LIZARD_REDUCED_CLASSES):
            aucs[cls] = presence_auroc(
                merged_scores[keep_mask, j], merged_probs[keep_mask, j]
            )
        per_ds_class.append(aucs)

    return _aggregate(per_ds_class, LIZARD_REDUCED_CLASSES)


def table2_auroc_pannuke(scores_by_sample, gt_by_sample):
    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig_classes = info["classes"]
        counts = info["counts"]

        keep_classes = [c for c in orig_classes if c not in PANNUKE_DROP]
        keep_idx = [orig_classes.index(c) for c in keep_classes]
        kept_counts = counts[:, keep_idx]
        kept_probs = kept_counts / (kept_counts.sum(axis=1, keepdims=True) + 1e-12)

        dominant = np.array(orig_classes)[counts.argmax(axis=1)]
        keep_mask = np.array([d not in PANNUKE_DROP for d in dominant])

        sc_cols = [c for c in keep_classes if c in sc.columns]
        sc_kept = sc[sc_cols].values

        aucs = {}
        for j, cls in enumerate(keep_classes):
            aucs[cls] = presence_auroc(sc_kept[keep_mask, j], kept_probs[keep_mask, j])
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


models = list(snakemake.params.models)
seed_labels = list(snakemake.params.seed_labels)
assert len(models) == len(seed_labels)

crc_gt = load_gt_files(snakemake.input.crc_gt)
lizard_gt = load_gt_files(snakemake.input.lizard_gt)
pannuke_gt = load_gt_files(snakemake.input.pannuke_gt)

crc_per_model = scores_by_model(snakemake.input.crc_scores, models)
lizard_per_model = scores_by_model(snakemake.input.lizard_scores, models)
pannuke_per_model = scores_by_model(snakemake.input.pannuke_scores, models)

rows = []
per_class_rows = []

for model, seed_label in zip(models, seed_labels):
    macro, pc = table2_auroc_crc(crc_per_model[model], crc_gt)
    rows.append({"seed": seed_label, "benchmark": "CRC_13class", "auroc": macro})
    for cls, auc in pc.items():
        per_class_rows.append(
            {"seed": seed_label, "benchmark": "CRC", "class": cls, "auroc": auc}
        )

    macro, pc = table2_auroc_lizard(lizard_per_model[model], lizard_gt)
    rows.append({"seed": seed_label, "benchmark": "Lizard_3class", "auroc": macro})
    for cls, auc in pc.items():
        per_class_rows.append(
            {"seed": seed_label, "benchmark": "Lizard", "class": cls, "auroc": auc}
        )

    macro, pc = table2_auroc_pannuke(pannuke_per_model[model], pannuke_gt)
    rows.append({"seed": seed_label, "benchmark": "PanNuke_4class", "auroc": macro})
    for cls, auc in pc.items():
        per_class_rows.append(
            {"seed": seed_label, "benchmark": "PanNuke", "class": cls, "auroc": auc}
        )

pd.DataFrame(rows).to_csv(snakemake.output.macro, index=False)
pd.DataFrame(per_class_rows).to_csv(snakemake.output.per_class, index=False)
