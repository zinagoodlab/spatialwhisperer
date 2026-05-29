"""
Compute reduced-class Lizard/PanNuke/CRC AUROC with per-dataset averaging
(macro per dataset, then mean across datasets) across training seeds.

Inputs/outputs: see compute_reduced_class_table2_style.py docstring.
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
        gt_labels = list(counts_nobg.columns[counts_nobg.values.argmax(axis=1)])
        gt[sample] = gt_labels
    return gt


def scores_by_model(score_files, models):
    groups = {m: {} for m in models}
    for f in map(Path, score_files):
        sample = f.stem.replace("_patch_scores_seed0", "")
        for m in models:
            if f"/{m}/" in str(f):
                groups[m][sample] = pd.read_csv(f)
                break
    return groups


def per_class_auroc(scores_array, gt_labels, class_names):
    gt_to_idx = {c: i for i, c in enumerate(class_names)}
    gt_onehot = np.zeros((len(gt_labels), len(class_names)))
    for i, lab in enumerate(gt_labels):
        if lab in gt_to_idx:
            gt_onehot[i, gt_to_idx[lab]] = 1
    aucs = {}
    for j, cls in enumerate(class_names):
        y_bin = gt_onehot[:, j]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs[cls] = np.nan
        else:
            aucs[cls] = roc_auc_score(y_bin, scores_array[:, j])
    return aucs


def eval_lizard_reduced_per_dataset(scores_by_sample, gt_by_sample):
    per_dataset_macro = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]

        epi = sc["Epithelial"].values
        leuk = sum(sc[c].values for c in LIZARD_LEUKOCYTE_ORIG if c in sc.columns)
        fibro = sc["Connective tissue"].values
        merged_scores = np.column_stack([epi, leuk, fibro])

        gt_merged, scores_merged = [], []
        for i, lab in enumerate(gt):
            if lab in LIZARD_DROP:
                continue
            if lab in LIZARD_LEUKOCYTE_ORIG:
                gt_merged.append("Leukocyte")
            elif lab == "Connective tissue":
                gt_merged.append("Fibroblast")
            else:
                gt_merged.append(lab)
            scores_merged.append(merged_scores[i])

        if len(gt_merged) < 5:
            continue
        scores_arr = np.vstack(scores_merged)
        aucs = per_class_auroc(scores_arr, gt_merged, LIZARD_REDUCED_CLASSES)
        macro = np.nanmean(list(aucs.values()))
        if not np.isnan(macro):
            per_dataset_macro.append(macro)

    return float(np.mean(per_dataset_macro)) if per_dataset_macro else np.nan


def eval_pannuke_reduced_per_dataset(scores_by_sample, gt_by_sample):
    per_dataset_macro = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]

        keep_cols = [
            c for c in sc.columns if c not in PANNUKE_DROP and c.lower() != "background"
        ]
        keep_cols = [c for c in keep_cols if c in PANNUKE_REDUCED_CLASSES]
        sc_kept = sc[keep_cols].values

        gt_filt, scores_filt = [], []
        for i, lab in enumerate(gt):
            if lab in PANNUKE_DROP or lab.lower() == "background":
                continue
            if lab in PANNUKE_REDUCED_CLASSES:
                gt_filt.append(lab)
                scores_filt.append(sc_kept[i])

        if len(gt_filt) < 5:
            continue
        scores_arr = np.vstack(scores_filt)
        aucs = per_class_auroc(scores_arr, gt_filt, PANNUKE_REDUCED_CLASSES)
        macro = np.nanmean(list(aucs.values()))
        if not np.isnan(macro):
            per_dataset_macro.append(macro)

    return float(np.mean(per_dataset_macro)) if per_dataset_macro else np.nan


def eval_crc_per_dataset(scores_by_sample, gt_by_sample):
    first_sc = next(iter(scores_by_sample.values()))
    orig_classes = list(first_sc.columns)
    keep_classes = [
        c for c in orig_classes if c not in CRC_EXCLUDE and c.lower() != "background"
    ]

    per_dataset_macro = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample][keep_classes].values

        gt_filt, scores_filt = [], []
        for i, lab in enumerate(gt):
            if lab in CRC_EXCLUDE or lab.lower() == "background":
                continue
            gt_filt.append(lab)
            scores_filt.append(sc[i])

        if len(gt_filt) < 5:
            continue
        scores_arr = np.vstack(scores_filt)
        aucs = per_class_auroc(scores_arr, gt_filt, keep_classes)
        macro = np.nanmean(list(aucs.values()))
        if not np.isnan(macro):
            per_dataset_macro.append(macro)

    return float(np.mean(per_dataset_macro)) if per_dataset_macro else np.nan


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
for model, seed_label in zip(models, seed_labels):
    auc = eval_crc_per_dataset(crc_per_model[model], crc_gt)
    rows.append({"seed": seed_label, "benchmark": "CRC_13class", "auroc": auc})

    auc = eval_lizard_reduced_per_dataset(lizard_per_model[model], lizard_gt)
    rows.append({"seed": seed_label, "benchmark": "Lizard_3class", "auroc": auc})

    auc = eval_pannuke_reduced_per_dataset(pannuke_per_model[model], pannuke_gt)
    rows.append({"seed": seed_label, "benchmark": "PanNuke_4class", "auroc": auc})

pd.DataFrame(rows).to_csv(snakemake.output.table, index=False)
