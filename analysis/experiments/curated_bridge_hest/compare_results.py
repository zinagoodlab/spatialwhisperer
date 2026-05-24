"""
Compare HEST benchmark results: bridge (quilt1m) vs bridge (quilt1m_curated).
Both models trained on G<->T + T<->I only (no hest1k), evaluated on G<->I retrieval.

Run locally after results are available:
    pixi run python src/spotwhisperer_eval/experiments/curated_bridge_hest/compare_results.py
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[4]  # cellwhisperer root

DATASETS = [
    "IDC",
    "PRAD",
    "PAAD",
    "SKCM",
    "COAD",
    "READ",
    "CCRCC",
    "HCC",
    "LUNG",
    "LYMPH_IDC",
]
MODELS = {
    "Bridge (uncurated)": "spotwhisperer_cellxgene_census__archs4_geo__quilt1m",
    "Bridge (curated)": "spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated",
}
METRICS = [
    "test_retrieval/transcriptome_image/rocauc_macroAvg",
    "test_retrieval/transcriptome_image/f1_macroAvg",
    "test_retrieval/image_transcriptome/rocauc_macroAvg",
    "test_retrieval/image_transcriptome/f1_macroAvg",
]
CSV_LOGS = PROJECT_DIR / "results/spotwhisperer_eval/csv_logging"

rows = []
for model_label, model_name in MODELS.items():
    for dataset in DATASETS:
        metrics_path = (
            CSV_LOGS / f"hest_eval___{model_name}___{dataset}" / "metrics.csv"
        )
        if not metrics_path.exists():
            print(f"MISSING: {metrics_path}")
            continue
        df = pd.read_csv(metrics_path)
        # Get last row (epoch-level summary)
        last = df.dropna(
            subset=[c for c in METRICS if c in df.columns], how="all"
        ).iloc[-1]
        for metric in METRICS:
            if metric in df.columns:
                rows.append(
                    {
                        "model": model_label,
                        "dataset": dataset,
                        "metric": metric.split("/")[-2] + "/" + metric.split("/")[-1],
                        "value": last[metric],
                    }
                )

results = pd.DataFrame(rows)
if results.empty:
    print("No results found. Ensure HEST evals have been run.")
    raise SystemExit(1)

# Pivot for comparison
pivot = results.pivot_table(
    index=["dataset", "metric"], columns="model", values="value"
)
pivot["delta"] = pivot["Bridge (curated)"] - pivot["Bridge (uncurated)"]
pivot = pivot.sort_values(["metric", "dataset"])

print("\n=== Per-dataset comparison ===")
print(pivot.to_string())

# Summary statistics
print("\n=== Mean across datasets ===")
summary = pivot.groupby("metric").mean()
print(summary.to_string())

# Save
output_dir = Path(__file__).parent
pivot.to_csv(output_dir / "comparison_per_dataset.csv")
summary.to_csv(output_dir / "comparison_summary.csv")
print(f"\nSaved to {output_dir}")
