"""
Aggregate per-class CRC metrics across training seeds into mean +/- std table.

Inputs (Snakemake):
- snakemake.input.seed0_per_class: CSV for the existing seed-0 model
- snakemake.input.seeded_per_class: list of CSVs for additional seeded models
- snakemake.params.seed0: the seed value for the existing model (int)
- snakemake.params.extra_seeds: list of additional seed values

Outputs:
- snakemake.output.table: CSV with columns: class_label, rocauc_mean, rocauc_std, rocauc_str, ...
"""

import pandas as pd

all_dfs = []

# Seed 0 (existing model, no _seed suffix)
df0 = pd.read_csv(snakemake.input.seed0_per_class)
df0["training_seed"] = snakemake.params.seed0
all_dfs.append(df0)

# Additional seeds
for fp, seed in zip(snakemake.input.seeded_per_class, snakemake.params.extra_seeds):
    df = pd.read_csv(fp)
    df["training_seed"] = seed
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

# Exclude classes not used in Table 2 (matching plot_pathocell_baselines_vs_trimodal.py)
EXCLUDE_CLASSES = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}
combined = combined[~combined["class_label"].isin(EXCLUDE_CLASSES)]

METRICS = ["rocauc", "f1", "precision", "accuracy", "soft_rocauc"]
available_metrics = [m for m in METRICS if m in combined.columns]

grouped = combined.groupby("class_label")
rows = []
for cls, grp in grouped:
    row = {"class_label": cls}
    for m in available_metrics:
        vals = grp[m].dropna()
        row[f"{m}_mean"] = vals.mean()
        row[f"{m}_std"] = vals.std()
        row[f"{m}_str"] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
    rows.append(row)

result = pd.DataFrame(rows)

# Append mean row (macro-average across classes, matching Table 2)
mean_row = {"class_label": "mean"}
for m in available_metrics:
    col_mean = result[f"{m}_mean"]
    mean_row[f"{m}_mean"] = col_mean.mean()
    col_std = result[f"{m}_std"]
    mean_row[f"{m}_std"] = col_std.mean()
    mean_row[f"{m}_str"] = f"{col_mean.mean():.3f} +/- {col_std.mean():.3f}"
result = pd.concat([result, pd.DataFrame([mean_row])], ignore_index=True)

result.to_csv(snakemake.output.table, index=False)
