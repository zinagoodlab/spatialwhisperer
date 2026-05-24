#!/usr/bin/env python3
"""
Compute reduced-class AUROC with Table 2 aggregation strategy:
  1. For each (dataset, class): compute one-vs-rest AUROC using presence-based binary labels
  2. Average across datasets within each class → per-class AUROC
  3. Average across classes → final macro AUROC

Presence-based: a patch is "positive" for class j if it contains ANY cells of class j
(i.e., true_probs[:, j] > 0), matching compute_pathocell_metrics_from_scores.py line 155.

Usage (on Sherlock):
    python compute_reduced_class_table2_style.py
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
    """Load ground truth: returns dict of sample -> {classes, counts, true_probs, gt_labels}."""
    gt = {}
    for f in sorted(h5ad_dir.glob(pattern)):
        sample = f.stem.replace("_patch", "")
        adata = anndata.read_h5ad(str(f))
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


def load_scores(score_dir, seed_suffix="seed0"):
    scores_by_sample = {}
    for f in sorted(score_dir.glob(f"*_scores_{seed_suffix}.csv")):
        sample = f.stem.replace(f"_patch_scores_{seed_suffix}", "")
        scores_by_sample[sample] = pd.read_csv(f)
    return scores_by_sample


def presence_auroc(scores_col, true_probs_col):
    """Compute AUROC using presence-based binary labels (matching the pipeline)."""
    y_bin = (true_probs_col > 0).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    return roc_auc_score(y_bin, scores_col)


def table2_auroc_crc(scores_by_sample, gt_by_sample):
    """CRC: presence-based AUROC, exclude Background + Other cells, classes→datasets→mean."""
    # Determine keep classes from first sample
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
    """Lizard 3-class: merge leukocytes, drop Plasma, presence-based AUROC."""
    per_ds_class = []
    for sample in sorted(gt_by_sample.keys()):
        if sample not in scores_by_sample:
            continue
        info = gt_by_sample[sample]
        sc = scores_by_sample[sample]
        orig_classes = info["classes"]
        counts = info["counts"]

        # Build merged counts: Epithelial, Leukocyte (sum of Neutro+Lympho+Eosino), Fibroblast
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
        epi_score = sc["Epithelial"].values
        leuk_score = sum(sc[c].values for c in LIZARD_LEUKOCYTE_ORIG if c in sc.columns)
        fibro_score = sc["Connective tissue"].values
        merged_scores = np.column_stack([epi_score, leuk_score, fibro_score])

        # Filter rows where dominant class is Plasma (dropped)
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
    """PanNuke 4-class: drop Dead Cells, presence-based AUROC."""
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

        # Filter rows where dominant class is Dead Cells
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
    """Classes → datasets → mean aggregation."""
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
    macro = np.mean(valid) if valid else np.nan
    return macro, per_class_mean


if __name__ == "__main__":
    print("Loading ground truth...")
    lizard_gt = load_gt_for_dir(H5AD_BASE / "lizard")
    pannuke_gt = load_gt_for_dir(H5AD_BASE / "pannuke")
    crc_gt = load_gt_for_dir(H5AD_BASE)

    rows = []
    per_class_rows = []

    for seed_label, model_name in MODELS.items():
        model_dir = BASE / model_name
        print(f"\n{'=' * 60}")
        print(f"  {seed_label} ({model_name})")
        print(f"{'=' * 60}")

        # CRC
        crc_scores = load_scores(model_dir)
        if crc_scores:
            macro, pc = table2_auroc_crc(crc_scores, crc_gt)
            print(f"  CRC 13-class: AUROC = {macro:.4f}")
            rows.append(
                {"seed": seed_label, "benchmark": "CRC_13class", "auroc": macro}
            )
            for cls, auc in pc.items():
                per_class_rows.append(
                    {"seed": seed_label, "benchmark": "CRC", "class": cls, "auroc": auc}
                )

        # Lizard
        liz_dir = model_dir / "lizard"
        if liz_dir.exists():
            liz_scores = load_scores(liz_dir)
            if liz_scores:
                macro, pc = table2_auroc_lizard(liz_scores, lizard_gt)
                print(f"  Lizard 3-class: AUROC = {macro:.4f}")
                for cls, auc in pc.items():
                    print(f"    {cls:20s}: {auc:.4f}")
                rows.append(
                    {"seed": seed_label, "benchmark": "Lizard_3class", "auroc": macro}
                )
                for cls, auc in pc.items():
                    per_class_rows.append(
                        {
                            "seed": seed_label,
                            "benchmark": "Lizard",
                            "class": cls,
                            "auroc": auc,
                        }
                    )

        # PanNuke
        pan_dir = model_dir / "pannuke"
        if pan_dir.exists():
            pan_scores = load_scores(pan_dir)
            if pan_scores:
                macro, pc = table2_auroc_pannuke(pan_scores, pannuke_gt)
                print(f"  PanNuke 4-class: AUROC = {macro:.4f}")
                for cls, auc in pc.items():
                    print(f"    {cls:35s}: {auc:.4f}")
                rows.append(
                    {"seed": seed_label, "benchmark": "PanNuke_4class", "auroc": macro}
                )
                for cls, auc in pc.items():
                    per_class_rows.append(
                        {
                            "seed": seed_label,
                            "benchmark": "PanNuke",
                            "class": cls,
                            "auroc": auc,
                        }
                    )

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print("SUMMARY (Table 2 style: presence-based AUROC, classes → datasets → mean)")
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

    out_path = BASE / "seed_variance" / "reduced_class_table2_style.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    pc_df = pd.DataFrame(per_class_rows)
    pc_path = BASE / "seed_variance" / "reduced_class_table2_style_per_class.csv"
    pc_df.to_csv(pc_path, index=False)
    print(f"Saved per-class to {pc_path}")
