"""Compute tissue type prediction metrics from per-batch tissue type scores.

Aggregates patch-level scores to sample-level (mean logits across patches from the
same original PanNuke sample), then computes per-tissue AUROC and confusion matrix.

Optionally filters to benign-only patches (0 neoplastic cells) following PLIP's
definition, controlled by snakemake.params.benign_only.

Inputs (Snakemake):
  - snakemake.input.scores: list of per-batch tissue type score CSVs
  - snakemake.input.mapping: patch-to-sample-to-tissue mapping CSV
  - snakemake.input.adatas: list of per-batch h5ad files (for neoplastic cell counts)

Params:
  - snakemake.params.benign_only: if True, restrict to benign patches only

Outputs:
  - snakemake.output.per_class: CSV with per-tissue-type metrics
  - snakemake.output.summary: CSV with macro-averaged summary
  - snakemake.output.confusion: CSV confusion matrix (true vs predicted tissue type)
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import roc_auc_score, confusion_matrix as sk_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEOPLASTIC_COL = "Neoplastic cells"

# ── Load mapping ─────────────────────────────────────────────────────────────
mapping = pd.read_csv(snakemake.input.mapping)

# ── Load tissue type score CSVs and join with mapping ────────────────────────
score_fps = sorted(Path(p) for p in snakemake.input.scores)
adata_fps = sorted(Path(p) for p in snakemake.input.adatas)

# Build batch_name -> adata filepath mapping for benign filtering
adata_map = {}
for fp in adata_fps:
    bn = fp.stem.replace("_patch", "")
    adata_map[bn] = fp

all_rows = []  # (sample_name, tissue_type, scores_array, is_benign)

for score_fp in score_fps:
    batch_name = score_fp.stem.replace("_tissue_scores", "")
    sdf = pd.read_csv(score_fp)
    tissue_types = list(sdf.columns)

    # Get mapping rows for this batch
    batch_mapping = mapping[mapping["batch_name"] == batch_name].reset_index(drop=True)
    assert len(batch_mapping) == len(sdf), (
        f"Mapping/score length mismatch for {batch_name}: {len(batch_mapping)} vs {len(sdf)}"
    )

    # Determine benign status per patch if needed
    is_benign = np.ones(len(sdf), dtype=bool)  # default: all included
    if snakemake.params.benign_only and batch_name in adata_map:
        adata = ad.read_h5ad(adata_map[batch_name])
        ct = adata.obsm["cell_type_counts"]
        neo = ct[NEOPLASTIC_COL].values
        is_benign = neo == 0

    for i in range(len(sdf)):
        all_rows.append(
            {
                "sample_name": batch_mapping.iloc[i]["sample_name"],
                "tissue_type": batch_mapping.iloc[i]["tissue_type"],
                "scores": sdf.iloc[i].values.astype(float),
                "is_benign": is_benign[i],
            }
        )

df = pd.DataFrame(all_rows)

# Filter to benign-only if requested
if snakemake.params.benign_only:
    n_before = len(df)
    df = df[df["is_benign"]].reset_index(drop=True)
    logger.info("Benign-only filter: %d -> %d patches", n_before, len(df))

logger.info(
    "Total patches: %d, samples: %d, tissue types: %d",
    len(df),
    df["sample_name"].nunique(),
    df["tissue_type"].nunique(),
)

# ── Aggregate to sample level (mean logits) ──────────────────────────────────
sample_scores = defaultdict(list)
sample_tissue = {}
for _, row in df.iterrows():
    sn = row["sample_name"]
    sample_scores[sn].append(row["scores"])
    sample_tissue[sn] = row["tissue_type"]

sample_names = sorted(sample_scores.keys())
n_samples = len(sample_names)
n_classes = len(tissue_types)

scores_matrix = np.zeros((n_samples, n_classes))
true_labels = []
for i, sn in enumerate(sample_names):
    scores_matrix[i] = np.mean(sample_scores[sn], axis=0)
    true_labels.append(sample_tissue[sn])

true_labels = np.array(true_labels)
# Normalize tissue type names in true labels to match score columns.
# types.npy uses e.g. "Adrenal_gland", "HeadNeck", "Bile-duct".
# Score columns use lowercase with underscores/hyphens replaced by spaces,
# plus special case: "HeadNeck" -> "head and neck".
_TISSUE_NORM = {"headneck": "head and neck"}


def _normalize_tissue(t):
    s = t.lower().replace("_", " ").replace("-", " ")
    return _TISSUE_NORM.get(s, s)


true_labels_norm = np.array([_normalize_tissue(t) for t in true_labels])

logger.info("Sample-level: %d samples, %d classes", n_samples, n_classes)
logger.info(
    "Tissue distribution:\n%s", pd.Series(true_labels_norm).value_counts().to_string()
)

# ── Softmax for probabilities ────────────────────────────────────────────────
m = scores_matrix - scores_matrix.max(axis=1, keepdims=True)
expm = np.exp(m)
pred_probs = expm / (expm.sum(axis=1, keepdims=True) + 1e-12)

pred_labels = np.array([tissue_types[i] for i in scores_matrix.argmax(axis=1)])

# ── Per-class AUROC (one-vs-rest) ────────────────────────────────────────────
per_class_rows = []
for j, cls in enumerate(tissue_types):
    y_bin = (true_labels_norm == cls).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int((1 - y_bin).sum())
    s = scores_matrix[:, j]

    if n_pos == 0 or n_neg == 0:
        rocauc = np.nan
    else:
        rocauc = roc_auc_score(y_bin, s)

    # Accuracy and F1 for this class (top-1)
    y_pred_bin = (pred_labels == cls).astype(int)
    tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else np.nan
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else np.nan

    per_class_rows.append(
        {
            "tissue_type": cls,
            "n_samples": n_pos,
            "rocauc": rocauc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    )

per_class_df = pd.DataFrame(per_class_rows)

# ── Confusion matrix ─────────────────────────────────────────────────────────
# Use only classes present in ground truth
present_classes = sorted(set(true_labels_norm))
cm = sk_confusion_matrix(true_labels_norm, pred_labels, labels=present_classes)
cm_df = pd.DataFrame(cm, index=present_classes, columns=present_classes)
cm_df.index.name = "true_tissue"

# ── Summary ──────────────────────────────────────────────────────────────────
accuracy = float((true_labels_norm == pred_labels).mean())
valid_aurocs = per_class_df["rocauc"].dropna()

summary = pd.DataFrame(
    [
        {
            "n_samples": n_samples,
            "n_patches": len(df),
            "n_classes": n_classes,
            "benign_only": bool(snakemake.params.benign_only),
            "accuracy": accuracy,
            "f1_macroAvg": float(np.nanmean(per_class_df["f1"])),
            "rocauc_macroAvg": float(valid_aurocs.mean())
            if len(valid_aurocs) > 0
            else np.nan,
            "n_classes_with_auroc": int(len(valid_aurocs)),
        }
    ]
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pc = Path(snakemake.output.per_class)
out_sum = Path(snakemake.output.summary)
out_cm = Path(snakemake.output.confusion)
out_pc.parent.mkdir(parents=True, exist_ok=True)

per_class_df.to_csv(out_pc, index=False)
summary.to_csv(out_sum, index=False)
cm_df.to_csv(out_cm)

logger.info(
    "Accuracy: %.3f, Macro AUROC: %.3f, Macro F1: %.3f",
    accuracy,
    summary["rocauc_macroAvg"].iloc[0],
    summary["f1_macroAvg"].iloc[0],
)
print(
    "\n=== Tissue Type Prediction (benign_only=%s) ===" % snakemake.params.benign_only
)
print(summary.to_string(index=False))
print("\nPer-class AUROC:")
print(per_class_df[["tissue_type", "n_samples", "rocauc", "f1"]].to_string(index=False))
print("\nConfusion matrix saved to", out_cm)
