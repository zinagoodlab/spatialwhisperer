#!/usr/bin/env python3
"""
Build per-class AUROC comparison table for trimodal ablation models on CRC benchmark.
Produces a CSV and a LaTeX snippet matching tab:pathocell_benchmark format.

Inputs (Snakemake):
- snakemake.input.per_class_csvs: list of per-class metric CSVs (one per model)
- snakemake.params.model_labels: display names matching the CSV order

Outputs:
- snakemake.output.csv_table: CSV with rows=cell types + Mean, columns=models
- snakemake.output.latex_table: LaTeX table snippet (bold best, underline second-best)
"""

from pathlib import Path
import pandas as pd

per_class_fps = [Path(p) for p in snakemake.input.per_class_csvs]
model_labels = snakemake.params.model_labels
csv_out = Path(snakemake.output.csv_table)
tex_out = Path(snakemake.output.latex_table)

METRIC = "rocauc"


def read_per_class(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    label_col = next(
        (
            c
            for c in ["class_label", "cell_type", "label", "class", "target"]
            if c in df.columns
        ),
        next(c for c in df.columns if df[c].dtype == object),
    )
    return df.rename(columns={label_col: "class_label"})


# Load all models
frames = {}
for fp, label in zip(per_class_fps, model_labels):
    df = read_per_class(fp)
    frames[label] = df.set_index("class_label")[METRIC]

merged = pd.DataFrame(frames)

# Drop non-informative classes
exclude = {
    "Other cells",
    "Background",
    "A sample of Other cells",
    "A sample of Background cells",
}
merged = merged[~merged.index.isin(exclude)]

# Append mean row
merged.loc["Mean"] = merged.mean()

# Write CSV
merged.to_csv(csv_out)

# Write LaTeX table
cols = list(merged.columns)
n_cols = len(cols)

lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"	\centering")
lines.append(
    r"	\caption{Zero-shot cell type prediction (mean AUROC, $n=109$ CRC samples). "
    r"Best per row is \textbf{bold}; second-best is \underline{underlined}.}"
)
lines.append(r"	\begin{tabular}{l " + "r" * n_cols + "}")
lines.append(r"		\toprule")
header = r"		\textbf{Cell Type}"
for c in cols:
    header += rf" & \textbf{{\emph{{{c}}}}}"
header += r" \\"
lines.append(header)
lines.append(r"		\midrule")

for ct in merged.index:
    vals = merged.loc[ct]
    sorted_vals = vals.sort_values(ascending=False)
    best_val = sorted_vals.iloc[0]
    second_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None

    row = f"		{ct}"
    for c in cols:
        v = vals[c]
        cell = f"{v:.3f}"
        if v == best_val:
            cell = rf"\textbf{{{cell}}}"
        elif second_val is not None and v == second_val:
            cell = rf"\underline{{{cell}}}"
        row += f" & {cell}"

    if ct == "Mean":
        # Insert midrule before Mean
        lines.append(r"		\midrule")
    row += r" \\"
    lines.append(row)

lines.append(r"		\bottomrule")
lines.append(r"	\end{tabular}")
lines.append(r"	\label{tab:trimodal_ablation_benchmark}")
lines.append(r"\end{table}")

tex_out.write_text("\n".join(lines) + "\n")
