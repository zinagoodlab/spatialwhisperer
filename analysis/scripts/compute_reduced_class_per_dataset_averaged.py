#!/usr/bin/env python3
"""
Compute reduced-class Lizard/PanNuke/CRC AUROC with per-dataset averaging
(matching Table 2 methodology), across training seeds.

For each dataset:
  1. Apply class merging/dropping
  2. Compute per-class AUROC within dataset
  3. Average across classes → per-dataset macro AUROC
Then average across datasets → final macro AUROC.

Usage (on Sherlock):
    python compute_reduced_class_per_dataset_averaged.py
"""

import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path(
    "/home/groups/zinaida/moritzs/cellwhisperer_private/results/pathocell_evaluation"
)
H5AD_BASE = Path(
    "/home/groups/zinaida/moritzs/cellwhisperer_private/resources/pathocell/processed"
)

MODELS = {
    "seed0_orig": "spotwhisperer_cellxgene_census__archs4_geo__hest1k",
    "seed0_retrained": "spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed0",
    "seed1": "spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed1",
    "seed2": "spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed2",
}

# --- Lizard reduced config ---
LIZARD_LEUKOCYTE_ORIG = {"Neutrophil", "Lymphocyte", "Eosinophil"}
LIZARD_DROP = {"Plasma"}
LIZARD_REDUCED_CLASSES = ["Epithelial", "Leukocyte", "Fibroblast"]

# --- PanNuke reduced config ---
PANNUKE_DROP = {"Dead Cells"}
PANNUKE_REDUCED_CLASSES = [
    "Epithelial",
    "Connective/Soft tissue cells",
    "Inflammatory",
    "Neoplastic cells",
]

# --- CRC config ---
CRC_EXCLUDE = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}


def load_gt_for_dir(h5ad_dir, pattern="*_patch.h5ad"):
    gt = {}
    for f in sorted(h5ad_dir.glob(pattern)):
        sample = f.stem.replace("_patch", "")
        adata = anndata.read_h5ad(str(f))
        counts_df = adata.obsm["cell_type_counts_coarse"]
        non_bg = [c for c in counts_df.columns if c.lower() != "background"]
        counts_nobg = counts_df[non_bg]
        gt_labels = list(counts_nobg.columns[counts_nobg.values.argmax(axis=1)])
        gt[sample] = gt_labels
    return gt


def load_scores(score_dir, seed_suffix="seed0"):
    scores_by_sample = {}
    for f in sorted(score_dir.glob(f"*_scores_{seed_suffix}.csv")):
        sample = f.stem.replace(f"_patch_scores_{seed_suffix}", "")
        scores_by_sample[sample] = pd.read_csv(f)
    return scores_by_sample


def per_class_auroc(scores_array, gt_labels, class_names):
    """Compute per-class one-vs-rest AUROC."""
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
    """Per-dataset averaged macro AUROC for Lizard 3-class."""
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

    return np.mean(per_dataset_macro) if per_dataset_macro else np.nan


def eval_pannuke_reduced_per_dataset(scores_by_sample, gt_by_sample):
    """Per-dataset averaged macro AUROC for PanNuke 4-class."""
    per_dataset_macro = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        gt = gt_by_sample[sample]
        sc = scores_by_sample[sample]

        keep_cols = [
            c for c in sc.columns if c not in PANNUKE_DROP and c.lower() != "background"
        ]
        # Only keep columns that are in PANNUKE_REDUCED_CLASSES
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

    return np.mean(per_dataset_macro) if per_dataset_macro else np.nan


def eval_crc_per_dataset(scores_by_sample, gt_by_sample):
    """Per-dataset averaged macro AUROC for CRC (excl Background + Other cells)."""
    # Determine class names from first sample
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

    return np.mean(per_dataset_macro) if per_dataset_macro else np.nan


if __name__ == "__main__":
    print("Loading ground truth...")
    lizard_gt = load_gt_for_dir(H5AD_BASE / "lizard")
    pannuke_gt = load_gt_for_dir(H5AD_BASE / "pannuke")
    crc_gt = load_gt_for_dir(H5AD_BASE)

    rows = []
    for seed_label, model_name in MODELS.items():
        model_dir = BASE / model_name
        print(f"\n{'=' * 60}")
        print(f"  {seed_label} ({model_name})")
        print(f"{'=' * 60}")

        # CRC
        crc_scores = load_scores(model_dir)
        if crc_scores:
            auc = eval_crc_per_dataset(crc_scores, crc_gt)
            print(f"  CRC 13-class (per-dataset avg): AUROC = {auc:.4f}")
            rows.append({"seed": seed_label, "benchmark": "CRC_13class", "auroc": auc})

        # Lizard
        liz_dir = model_dir / "lizard"
        if liz_dir.exists():
            liz_scores = load_scores(liz_dir)
            if liz_scores:
                auc = eval_lizard_reduced_per_dataset(liz_scores, lizard_gt)
                print(f"  Lizard 3-class (per-dataset avg): AUROC = {auc:.4f}")
                rows.append(
                    {"seed": seed_label, "benchmark": "Lizard_3class", "auroc": auc}
                )

        # PanNuke
        pan_dir = model_dir / "pannuke"
        if pan_dir.exists():
            pan_scores = load_scores(pan_dir)
            if pan_scores:
                auc = eval_pannuke_reduced_per_dataset(pan_scores, pannuke_gt)
                print(f"  PanNuke 4-class (per-dataset avg): AUROC = {auc:.4f}")
                rows.append(
                    {"seed": seed_label, "benchmark": "PanNuke_4class", "auroc": auc}
                )

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print("SUMMARY (per-dataset averaged, reduced classes)")
    print(f"{'=' * 60}")
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        print(f"\n{bench}:")
        for _, row in sub.iterrows():
            print(f"  {row['seed']:20s}: AUROC = {row['auroc']:.4f}")
        new_seeds = sub[sub["seed"].isin(["seed0_retrained", "seed1", "seed2"])][
            "auroc"
        ].values
        if len(new_seeds) >= 2:
            print(
                f"  Mean (new seeds):    AUROC = {np.mean(new_seeds):.4f} +/- {np.std(new_seeds):.4f}"
            )

    out_path = BASE / "seed_variance" / "reduced_class_per_dataset_averaged.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
