"""
Compute reduced-class Lizard/PanNuke/CRC AUROC across training seeds,
pooling scores globally (not per-sample averaged).
- Lizard: merge Neutrophil+Lymphocyte+Eosinophil → Leukocyte, drop Plasma (3 classes)
- PanNuke: drop Dead Cells (4 classes)
- CRC: exclude Background + Other cells (13 classes)

Inputs/outputs: see compute_reduced_class_table2_style.py docstring.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

LIZARD_LEUKOCYTE_ORIG = {"Neutrophil", "Lymphocyte", "Eosinophil"}
LIZARD_DROP = {"Plasma"}
LIZARD_REDUCED_CLASSES = ["Epithelial", "Leukocyte", "Fibroblast"]

PANNUKE_ORIG_CLASSES = [
    "Epithelial",
    "Dead Cells",
    "Connective/Soft tissue cells",
    "Inflammatory",
    "Neoplastic cells",
]
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


def load_ground_truth(h5ad_files):
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


def compute_metrics(scores_array, gt_labels, class_names):
    n_classes = len(class_names)
    pred_idx = scores_array.argmax(axis=1)
    pred_labels = [class_names[i] for i in pred_idx]
    f1 = f1_score(
        gt_labels, pred_labels, labels=class_names, average="macro", zero_division=0
    )
    accuracy = float(np.mean([g == p for g, p in zip(gt_labels, pred_labels)]))
    gt_to_idx = {c: i for i, c in enumerate(class_names)}
    gt_onehot = np.zeros((len(gt_labels), n_classes))
    for i, lab in enumerate(gt_labels):
        if lab in gt_to_idx:
            gt_onehot[i, gt_to_idx[lab]] = 1
    valid = (gt_onehot.sum(axis=0) > 0) & (gt_onehot.sum(axis=0) < len(gt_labels))
    if valid.sum() >= 2:
        rocauc = roc_auc_score(
            gt_onehot[:, valid],
            scores_array[:, valid],
            average="macro",
            multi_class="ovr",
        )
    else:
        rocauc = float("nan")
    return {"f1": f1, "auc": rocauc, "accuracy": accuracy}


def evaluate_reduced_lizard(scores_by_sample, gt_by_sample):
    all_gt, all_scores = [], []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        epithelial_score = sc["Epithelial"].values
        leukocyte_score = sum(
            sc[c].values for c in LIZARD_LEUKOCYTE_ORIG if c in sc.columns
        )
        fibroblast_score = sc["Connective tissue"].values
        merged_scores = np.column_stack(
            [epithelial_score, leukocyte_score, fibroblast_score]
        )
        for i, lab in enumerate(gt):
            if lab in LIZARD_DROP:
                continue
            if lab in LIZARD_LEUKOCYTE_ORIG:
                all_gt.append("Leukocyte")
            elif lab == "Connective tissue":
                all_gt.append("Fibroblast")
            else:
                all_gt.append(lab)
            all_scores.append(merged_scores[i])
    all_scores = np.vstack(all_scores)
    return compute_metrics(all_scores, all_gt, LIZARD_REDUCED_CLASSES)


def evaluate_reduced_pannuke(scores_by_sample, gt_by_sample):
    all_gt, all_scores = [], []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        keep_cols = [c for c in PANNUKE_ORIG_CLASSES if c not in PANNUKE_DROP]
        sc_kept = sc[keep_cols].values
        for i, lab in enumerate(gt):
            if lab in PANNUKE_DROP:
                continue
            all_gt.append(lab)
            all_scores.append(sc_kept[i])
    all_scores = np.vstack(all_scores)
    return compute_metrics(all_scores, all_gt, PANNUKE_REDUCED_CLASSES)


def evaluate_crc(scores_by_sample, gt_by_sample):
    all_gt, all_scores = [], []
    first_sample = next(iter(scores_by_sample.values()))
    orig_classes = list(first_sample.columns)
    keep_classes = [
        c for c in orig_classes if c not in CRC_EXCLUDE and c.lower() != "background"
    ]
    keep_idx = [orig_classes.index(c) for c in keep_classes]

    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        sc_kept = sc.iloc[:, keep_idx].values
        for i, lab in enumerate(gt):
            if lab in CRC_EXCLUDE or lab.lower() == "background":
                continue
            all_gt.append(lab)
            all_scores.append(sc_kept[i])
    all_scores = np.vstack(all_scores)
    return compute_metrics(all_scores, all_gt, keep_classes)


models = list(snakemake.params.models)
seed_labels = list(snakemake.params.seed_labels)
assert len(models) == len(seed_labels)

crc_gt = load_ground_truth(snakemake.input.crc_gt)
lizard_gt = load_ground_truth(snakemake.input.lizard_gt)
pannuke_gt = load_ground_truth(snakemake.input.pannuke_gt)

crc_per_model = scores_by_model(snakemake.input.crc_scores, models)
lizard_per_model = scores_by_model(snakemake.input.lizard_scores, models)
pannuke_per_model = scores_by_model(snakemake.input.pannuke_scores, models)

results = []
for model, seed_label in zip(models, seed_labels):
    r = evaluate_reduced_lizard(lizard_per_model[model], lizard_gt)
    results.append(
        {
            "seed": seed_label,
            "benchmark": "Lizard_3class",
            "f1": r["f1"],
            "auc": r["auc"],
            "accuracy": r["accuracy"],
        }
    )

    r = evaluate_reduced_pannuke(pannuke_per_model[model], pannuke_gt)
    results.append(
        {
            "seed": seed_label,
            "benchmark": "PanNuke_4class",
            "f1": r["f1"],
            "auc": r["auc"],
            "accuracy": r["accuracy"],
        }
    )

    r = evaluate_crc(crc_per_model[model], crc_gt)
    results.append(
        {
            "seed": seed_label,
            "benchmark": "CRC_13class",
            "f1": r["f1"],
            "auc": r["auc"],
            "accuracy": r["accuracy"],
        }
    )

pd.DataFrame(results).to_csv(snakemake.output.table, index=False)
