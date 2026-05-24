"""Aggregate per-config PathoCellBench metrics into a single CSV.

Mirrors the lambda_ablation aggregation: one row per ablation config, columns
following the convention established in
`experiments/lambda_ablation/results_crc_pathocell.csv`.

Run on Sherlock (so it can read the full results tree):

    cd /home/groups/zinaida/moritzs/cellwhisperer_private
    conda run -n cellwhisperer python \
        analysis/experiments/freezing_encoder_appendix/aggregate_results.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_DIR = Path("/home/groups/zinaida/moritzs/cellwhisperer_private")
RESULTS_DIR = PROJECT_DIR / "results/pathocell_evaluation"
OUT_PATH = (
    PROJECT_DIR
    / "analysis/experiments/freezing_encoder_appendix/results_crc_pathocell.csv"
)

# Configs we actually trained. `ull` is reported as N/A in the appendix table
# (Geneformer-unfrozen would not fit in H100 80 GB GPU memory at any batch
# size that produced stable training; see SUMMARY.md for details).
CONFIGS = ["baseline", "lll", "llu", "uce"]


def aggregate_one(config: str) -> dict:
    """Aggregate a single config's metrics into one row."""
    summary_dir = RESULTS_DIR / f"spotwhisperer_fe_{config}" / "summary"

    per_class = pd.read_csv(summary_dir / "patch_per_class_metrics_from_scores.csv")
    per_dataset = pd.read_csv(summary_dir / "patch_per_dataset_metrics_from_scores.csv")

    return {
        "config": config,
        # primary reviewer-facing metric: per-class AUROC (already dataset-averaged)
        # then averaged across classes
        "macro_auroc_classwise_dataset_avg": per_class["rocauc"].mean(),
        "macro_soft_auroc_classwise_dataset_avg": per_class["soft_rocauc"].mean(),
        # alternate aggregation: per-dataset macro then averaged across datasets
        "macro_auroc_datasetwise_class_avg": per_dataset["rocauc_macroAvg"].mean(),
        "macro_f1_datasetwise_class_avg": per_dataset["f1_macroAvg"].mean(),
        "macro_precision_datasetwise_class_avg": per_dataset["precision_macroAvg"].mean(),
        "macro_recall_at_5_datasetwise_class_avg": per_dataset["recall_at_5_macroAvg"].mean(),
        # calibration-style metrics, dataset-averaged
        "mean_cross_entropy": per_dataset["mean_cross_entropy"].mean(),
        "mean_js_divergence": per_dataset["mean_js_divergence"].mean(),
    }


rows = [aggregate_one(c) for c in CONFIGS]
df = pd.DataFrame(rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False, float_format="%.6f")
logging.info(f"Wrote {OUT_PATH}")
logging.info(df.to_string(index=False))
