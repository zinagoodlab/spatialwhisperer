"""Compute PanNuke cell type prediction metrics stratified by disease state.

Classifies each patch as malignant, benign, or ambiguous following PLIP's
strategy (Huang et al., 2023):
  - Malignant: >= 10 neoplastic cells AND >= 30% of total cells are neoplastic
  - Benign: 0 neoplastic cells
  - Ambiguous: in between (discarded)

Then computes per-class AUROC (and other metrics) separately for the malignant
and benign groups. Dead Cells class is dropped (consistent with the paper's
reduced-class PanNuke evaluation). For the benign group, Neoplastic cells is
additionally excluded (no ground-truth positives possible).

Inputs (Snakemake):
  - snakemake.input.scores: list of per-batch score CSVs
  - snakemake.input.adatas: list of per-batch h5ad files (patch-level)

Outputs:
  - snakemake.output.per_class: CSV with per-class metrics for each disease state
  - snakemake.output.summary: CSV with macro-averaged metrics per disease state
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PLIP thresholds ──────────────────────────────────────────────────────────
NEO_MIN_COUNT = 10
NEO_MIN_FRAC = 0.3

# Classes to drop globally (too few dominant patches, per paper)
DROP_CLASSES = {"Dead Cells"}
# Additionally dropped in the benign stratum (no ground-truth positives)
DROP_IN_BENIGN = {"Neoplastic cells"}

NEOPLASTIC_COL = "Neoplastic cells"


# ── Helpers ──────────────────────────────────────────────────────────────────


def classify_patches(adata):
    """Return a Series with values 'malignant', 'benign', or 'ambiguous'."""
    ct = adata.obsm["cell_type_counts"]
    non_bg = [c for c in ct.columns if c.lower() != "background"]
    total = ct[non_bg].sum(axis=1)
    neo = ct[NEOPLASTIC_COL]
    frac = neo / (total + 1e-12)

    state = pd.Series("ambiguous", index=adata.obs_names)
    state[neo == 0] = "benign"
    state[(neo >= NEO_MIN_COUNT) & (frac >= NEO_MIN_FRAC)] = "malignant"
    return state


def compute_per_class_auroc(true_probs, scores, classes):
    """Compute hard AUROC (presence-based) per class.

    Returns a dict {class_name: auroc} with NaN for degenerate classes.
    """
    results = {}
    for j, cls in enumerate(classes):
        y_soft = true_probs[:, j]
        y_bin = (y_soft > 0).astype(int)
        s = scores[:, j]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            results[cls] = np.nan
        else:
            results[cls] = roc_auc_score(y_bin, s)
    return results


def compute_metrics_for_group(true_probs, scores, classes):
    """Compute per-class and macro metrics for a group of patches."""
    ct_to_idx = {ct: i for i, ct in enumerate(classes)}
    obs_labels = np.array([classes[i] for i in true_probs.argmax(axis=1)])
    y_true_idx = true_probs.argmax(axis=1)
    pred_top1_idx = scores.argmax(axis=1)

    # Softmax
    m = scores - scores.max(axis=1, keepdims=True)
    expm = np.exp(m)
    pred_probs = expm / (expm.sum(axis=1, keepdims=True) + 1e-12)

    rows = []
    for j, cls in enumerate(classes):
        y_bin_hard = (y_true_idx == j).astype(int)
        y_pred_bin = (pred_top1_idx == j).astype(int)
        s = scores[:, j]

        tp = int(((y_bin_hard == 1) & (y_pred_bin == 1)).sum())
        fp = int(((y_bin_hard == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_bin_hard == 1) & (y_pred_bin == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else np.nan
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        accuracy = float((y_true_idx == pred_top1_idx).mean())

        # Hard AUROC (presence-based)
        y_soft = true_probs[:, j]
        y_bin_presence = (y_soft > 0).astype(int)
        if y_bin_presence.sum() == 0 or y_bin_presence.sum() == len(y_bin_presence):
            rocauc = np.nan
        else:
            rocauc = roc_auc_score(y_bin_presence, s)

        n_dominant = int(y_bin_hard.sum())
        rows.append(
            {
                "class_label": cls,
                "f1": f1,
                "precision": precision,
                "accuracy": accuracy,
                "rocauc": rocauc,
                "n_dominant_patches": n_dominant,
            }
        )

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

score_fps = sorted(Path(p) for p in snakemake.input.scores)
adata_fps = sorted(Path(p) for p in snakemake.input.adatas)


# Build a mapping from batch name -> filepath for both scores and adatas
def batch_name_from_adata(fp):
    return fp.stem.replace("_patch", "")


def batch_name_from_scores(fp):
    return fp.stem.replace("_patch_scores_seed0", "")


adata_map = {batch_name_from_adata(fp): fp for fp in adata_fps}
score_map = {batch_name_from_scores(fp): fp for fp in score_fps}

# Collect all patches with their disease state, ground truth, and scores
all_true_probs = []
all_scores = []
all_states = []
classes = None

for batch in sorted(adata_map.keys()):
    if batch not in score_map:
        logger.warning("No scores for batch %s, skipping", batch)
        continue

    adata = ad.read_h5ad(adata_map[batch])
    sdf = pd.read_csv(score_map[batch])

    # Ground truth: cell type counts excluding background
    counts_df = adata.obsm["cell_type_counts_coarse"].copy()
    all_classes = list(counts_df.columns)
    bg_idx = [i for i, c in enumerate(all_classes) if c.lower() == "background"]
    keep = [i for i in range(len(all_classes)) if i not in bg_idx]
    all_classes = [all_classes[i] for i in keep]
    counts_df = counts_df.iloc[:, keep]

    if classes is None:
        classes = all_classes
    assert classes == all_classes, f"Class mismatch in {batch}"

    true_probs = counts_df.values.astype(float)
    true_probs = true_probs / (true_probs.sum(axis=1, keepdims=True) + 1e-12)

    # Align score columns
    try:
        sdf = sdf[classes]
    except KeyError:
        sdf = sdf.iloc[:, : len(classes)].copy()
        sdf.columns = classes
    scores = sdf.values.astype(float)

    assert len(scores) == len(true_probs), f"Row mismatch in {batch}"

    state = classify_patches(adata)

    all_true_probs.append(true_probs)
    all_scores.append(scores)
    all_states.append(state.values)

all_true_probs = np.concatenate(all_true_probs, axis=0)
all_scores = np.concatenate(all_scores, axis=0)
all_states = np.concatenate(all_states, axis=0)

logger.info(
    "Loaded %d patches: %d malignant, %d benign, %d ambiguous",
    len(all_states),
    (all_states == "malignant").sum(),
    (all_states == "benign").sum(),
    (all_states == "ambiguous").sum(),
)

# ── Compute stratified metrics ───────────────────────────────────────────────

per_class_rows = []
summary_rows = []

for stratum in ["malignant", "benign"]:
    mask = all_states == stratum
    n_patches = int(mask.sum())
    if n_patches == 0:
        logger.warning("No %s patches, skipping", stratum)
        continue

    tp = all_true_probs[mask]
    sc = all_scores[mask]

    # Determine which classes to keep for this stratum
    drop = set(DROP_CLASSES)
    if stratum == "benign":
        drop |= DROP_IN_BENIGN
    keep_idx = [i for i, c in enumerate(classes) if c not in drop]
    kept_classes = [classes[i] for i in keep_idx]

    tp_kept = tp[:, keep_idx]
    # Re-normalize true probs after dropping classes
    tp_kept = tp_kept / (tp_kept.sum(axis=1, keepdims=True) + 1e-12)
    sc_kept = sc[:, keep_idx]

    df = compute_metrics_for_group(tp_kept, sc_kept, kept_classes)
    df["disease_state"] = stratum
    df["n_patches_total"] = n_patches
    per_class_rows.append(df)

    # Macro averages (only over non-NaN classes)
    summary_rows.append(
        {
            "disease_state": stratum,
            "n_patches": n_patches,
            "n_classes": len(kept_classes),
            "classes": ", ".join(kept_classes),
            "f1_macroAvg": float(np.nanmean(df["f1"])),
            "precision_macroAvg": float(np.nanmean(df["precision"])),
            "accuracy": float(df["accuracy"].iloc[0]),  # same for all classes
            "rocauc_macroAvg": float(np.nanmean(df["rocauc"])),
        }
    )

per_class_df = pd.concat(per_class_rows, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

# ── Save outputs ─────────────────────────────────────────────────────────────

out_per_class = Path(snakemake.output.per_class)
out_summary = Path(snakemake.output.summary)
out_per_class.parent.mkdir(parents=True, exist_ok=True)
out_summary.parent.mkdir(parents=True, exist_ok=True)

per_class_df.to_csv(out_per_class, index=False)
summary_df.to_csv(out_summary, index=False)

logger.info("Per-class metrics saved to %s", out_per_class)
logger.info("Summary metrics saved to %s", out_summary)

# ── Print summary ────────────────────────────────────────────────────────────

print("\n=== Disease-Stratified PanNuke Metrics ===\n")
print(summary_df.to_string(index=False))
print()
print("Per-class AUROC by disease state:")
pivot = per_class_df.pivot_table(
    index="class_label", columns="disease_state", values="rocauc"
)
print(pivot.to_string())
print()
