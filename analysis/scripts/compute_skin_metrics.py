"""Macro one-vs-rest AUROC for the Kriegsmann skin benchmark.

Reads the per-patch score CSV written by `score_kriegsmann_skin.py` and
computes per-class AUROC, the macro mean, and an argmax confusion matrix.

Inputs (Snakemake):
  - snakemake.input.scores: per-patch CSV

Outputs:
  - snakemake.output.per_class: CSV with per-class AUROC + support counts
  - snakemake.output.summary: CSV with the macro AUROC and totals
  - snakemake.output.confusion: square confusion matrix (true vs argmax-predicted)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scores_df = pd.read_csv(snakemake.input.scores)
meta_cols = {"patch_file", "true_class_raw", "true_class"}
class_cols = [c for c in scores_df.columns if c not in meta_cols]

y_true = scores_df["true_class"].values
y_score = scores_df[class_cols].to_numpy(dtype=float)

per_class_rows = []
for i, c in enumerate(class_cols):
    y_bin = (y_true == c).astype(int)
    n_pos = int(y_bin.sum())
    if n_pos == 0 or n_pos == len(y_bin):
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_bin, y_score[:, i]))
    per_class_rows.append({"class": c, "auroc": auroc, "n_pos": n_pos, "n_total": len(y_bin)})

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(snakemake.output.per_class, index=False)

valid = per_class_df["auroc"].dropna()
macro = float(valid.mean())
summary = pd.DataFrame(
    {
        "metric": ["macro_avg_rocauc", "n_classes_evaluated", "n_classes_total", "n_patches"],
        "value": [macro, len(valid), len(class_cols), len(y_true)],
    }
)
summary.to_csv(snakemake.output.summary, index=False)

preds = np.array(class_cols)[y_score.argmax(axis=1)]
conf = (
    pd.crosstab(pd.Series(y_true, name="true"), pd.Series(preds, name="pred"))
    .reindex(index=class_cols, columns=class_cols, fill_value=0)
)
conf.to_csv(snakemake.output.confusion)

logger.info(
    "Macro one-vs-rest AUROC = %.4f over %d/%d classes (n=%d patches)",
    macro,
    len(valid),
    len(class_cols),
    len(y_true),
)
