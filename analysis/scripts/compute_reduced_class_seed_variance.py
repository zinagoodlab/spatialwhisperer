#!/usr/bin/env python3
"""
Compute reduced-class Lizard/PanNuke AUROC across training seeds.

Matches the paper's reporting methodology from analyze_baselines.py:
- Lizard: merge Neutrophil+Lymphocyte+Eosinophil → Leukocyte, drop Plasma (3 classes)
- PanNuke: drop Dead Cells (4 classes)
- Scores are pooled globally across all samples (not per-sample averaged)
- CRC: exclude Background + Other cells (13 classes)

Usage (on Sherlock):
    python compute_reduced_class_seed_variance.py
"""

import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "results/pathocell_evaluation"
H5AD_BASE = PROJECT_ROOT / "resources/pathocell/processed"

MODELS = {
    "seed0_orig": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k",
    "seed0_retrained": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k_seed0",
    "seed1": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k_seed1",
    "seed2": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k_seed2",
}

# --- Lizard config ---
LIZARD_ORIG_CLASSES = [
    "Neutrophil",
    "Epithelial",
    "Lymphocyte",
    "Plasma",
    "Eosinophil",
    "Connective tissue",
]
LIZARD_LEUKOCYTE_ORIG = {"Neutrophil", "Lymphocyte", "Eosinophil"}
LIZARD_DROP = {"Plasma"}
LIZARD_REDUCED_CLASSES = ["Epithelial", "Leukocyte", "Fibroblast"]

# --- PanNuke config ---
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

# --- CRC config ---
CRC_EXCLUDE = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}


def load_ground_truth(h5ad_dir, pattern="*_patch.h5ad"):
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
        df = pd.read_csv(f)
        scores_by_sample[sample] = df
    return scores_by_sample


def compute_metrics(scores_array, gt_labels, class_names):
    n_classes = len(class_names)
    pred_idx = scores_array.argmax(axis=1)
    pred_labels = [class_names[i] for i in pred_idx]
    f1 = f1_score(
        gt_labels, pred_labels, labels=class_names, average="macro", zero_division=0
    )
    accuracy = np.mean([g == p for g, p in zip(gt_labels, pred_labels)])
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

    # Per-class AUROC
    per_class_auc = {}
    for j, cls in enumerate(class_names):
        y_bin = gt_onehot[:, j]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auc[cls] = float("nan")
        else:
            per_class_auc[cls] = roc_auc_score(y_bin, scores_array[:, j])

    return {
        "f1": f1,
        "auc": rocauc,
        "accuracy": accuracy,
        "per_class_auc": per_class_auc,
    }


def evaluate_reduced_lizard(scores_by_sample, gt_by_sample):
    all_gt = []
    all_scores = []
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
    all_gt = []
    all_scores = []
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
    """CRC: exclude Background + Other cells, keep all other classes."""
    all_gt = []
    all_scores = []
    # Determine class names from first sample
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


if __name__ == "__main__":
    print("Loading ground truth...")
    lizard_gt = load_ground_truth(H5AD_BASE / "lizard")
    pannuke_gt = load_ground_truth(H5AD_BASE / "pannuke")
    crc_gt = load_ground_truth(H5AD_BASE)  # CRC h5ads are in the root processed dir

    results = []

    for seed_label, model_name in MODELS.items():
        model_dir = BASE / model_name
        print(f"\n{'=' * 60}")
        print(f"Model: {seed_label} ({model_name})")
        print(f"{'=' * 60}")

        # Lizard
        lizard_score_dir = model_dir / "lizard"
        if lizard_score_dir.exists():
            scores = load_scores(lizard_score_dir)
            if scores:
                r = evaluate_reduced_lizard(scores, lizard_gt)
                print(
                    f"  Lizard (3-class): F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}"
                )
                for cls, auc in r["per_class_auc"].items():
                    print(f"    {cls:20s}: {auc:.4f}")
                results.append(
                    {
                        "seed": seed_label,
                        "benchmark": "Lizard_3class",
                        "f1": r["f1"],
                        "auc": r["auc"],
                        "accuracy": r["accuracy"],
                    }
                )

        # PanNuke
        pannuke_score_dir = model_dir / "pannuke"
        if pannuke_score_dir.exists():
            scores = load_scores(pannuke_score_dir)
            if scores:
                r = evaluate_reduced_pannuke(scores, pannuke_gt)
                print(
                    f"  PanNuke (4-class): F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}"
                )
                for cls, auc in r["per_class_auc"].items():
                    print(f"    {cls:35s}: {auc:.4f}")
                results.append(
                    {
                        "seed": seed_label,
                        "benchmark": "PanNuke_4class",
                        "f1": r["f1"],
                        "auc": r["auc"],
                        "accuracy": r["accuracy"],
                    }
                )

        # CRC
        crc_score_dir = model_dir
        crc_scores = load_scores(crc_score_dir)
        if crc_scores:
            r = evaluate_crc(crc_scores, crc_gt)
            print(
                f"  CRC (excl Bg+Other): F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}"
            )
            results.append(
                {
                    "seed": seed_label,
                    "benchmark": "CRC_13class",
                    "f1": r["f1"],
                    "auc": r["auc"],
                    "accuracy": r["accuracy"],
                }
            )

    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        print(f"\n{bench}:")
        for _, row in sub.iterrows():
            print(f"  {row['seed']:20s}: AUC={row['auc']:.4f}  F1={row['f1']:.4f}")
        aucs = sub["auc"].values
        if len(aucs) >= 2:
            # Exclude seed0_orig for apples-to-apples variance (different config)
            new_seeds = sub[sub["seed"] != "seed0_orig"]["auc"].values
            if len(new_seeds) >= 2:
                print(
                    f"  Mean (new seeds):    AUC={np.mean(new_seeds):.4f} +/- {np.std(new_seeds):.4f}"
                )

    # Save
    out_path = BASE / "seed_variance" / "reduced_class_seed_variance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
