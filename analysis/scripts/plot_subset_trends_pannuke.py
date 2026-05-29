#!/usr/bin/env python
"""
Variant of plot_subset_trends.py that uses PanNuke tissue-type prediction
(19-organ zero-shot patch classification, run through our own inference
pipeline) instead of the MUSK skin-conditions benchmark for the I->T column.

All other columns (transcriptome-text, transcriptome-image) are unchanged.

Inputs via Snakemake params:
- snakemake.params.benchmarks_dir: base directory for aggregated results
- snakemake.params.pathocell_eval_dir: results/pathocell_evaluation root
- snakemake.params.ratios: list of subsampling ratios, e.g., [1, 8, 64, 512]
- snakemake.params.plot_trimodal_all_subset: bool toggle to include third line

Output:
- snakemake.output.plot: path to save the grid plot SVG
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(snakemake.input.mpl_style)

benchmarks_dir = Path(snakemake.params.benchmarks_dir)
pathocell_eval_dir = Path(snakemake.params.pathocell_eval_dir)
ratios = list(snakemake.params.ratios)
plot_trimodal_all_subset = bool(snakemake.params.plot_trimodal_all_subset)

modality_pairs = ["transcriptome-text", "transcriptome-image", "image-text"]

test_dataset_map = {
    "transcriptome-text": "cellxgene_census__archs4_geo",
    "transcriptome-image": "hest1k",
    "image-text": "quilt1m",
}

metrics_by_pair = {
    "transcriptome-text": [
        "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
    ],
    "transcriptome-image": [
        "hest/overall_performance",
    ],
    "image-text": [
        "pannuke_tissue/rocauc_macroAvg",
    ],
}


def build_combo(modality_pair: str, ratio: int, include_bridge: bool) -> str:
    if modality_pair == "transcriptome-text":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "cellxgene_census__archs4_geo"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k__quilt1m"
            if include_bridge
            else f"cellxgene_census_{suffix}__archs4_geo_{suffix}"
        )
    elif modality_pair == "transcriptome-image":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "hest1k"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census__archs4_geo__hest1k_{suffix}__quilt1m"
            if include_bridge
            else f"hest1k_{suffix}"
        )
    elif modality_pair == "image-text":
        if ratio == 1:
            return (
                "cellxgene_census__archs4_geo__hest1k__quilt1m"
                if include_bridge
                else "quilt1m"
            )
        suffix = f"{ratio}thsub"
        return (
            f"cellxgene_census__archs4_geo__hest1k__quilt1m_{suffix}"
            if include_bridge
            else f"quilt1m_{suffix}"
        )
    else:
        raise ValueError(f"Unknown modality_pair: {modality_pair}")


def build_trimodal_all_subset_combo(ratio: int) -> str:
    if ratio == 1:
        return "cellxgene_census__archs4_geo__hest1k__quilt1m"
    suffix = f"{ratio}thsub"
    return f"cellxgene_census_{suffix}__archs4_geo_{suffix}__hest1k_{suffix}__quilt1m_{suffix}"


def extract_metrics_for_combo(modality_pair: str, combo: str) -> dict:
    out = {}
    ret_path = benchmarks_dir / "retrieval" / combo / "aggregated_retrieval.csv"
    df_ret = pd.read_csv(ret_path, index_col=0)
    ret_row = df_ret.loc[test_dataset_map[modality_pair]]

    cw_path = benchmarks_dir / "retrieval" / combo / "aggregated_cwevals.csv"
    df_cw = pd.read_csv(cw_path, index_col=0, header=None).squeeze("columns")

    if modality_pair == "transcriptome-text":
        for direction in ["left_right", "right_left"]:
            for metric in ["rocauc_macroAvg", "recall_at_50_macroAvg"]:
                key = f"test_retrieval/{direction}/{metric}"
                out[key] = float(ret_row[key])
        for cw_key in [
            "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
            "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
        ]:
            out[cw_key] = float(df_cw.loc[cw_key])

    elif modality_pair == "transcriptome-image":
        hest_path = benchmarks_dir / "hest" / combo / "aggregated_results.json"
        with open(hest_path, "r") as f:
            hest_json = json.load(f)
        out["hest/overall_performance"] = float(hest_json["overall_performance"])

    elif modality_pair == "image-text":
        summary_path = (
            pathocell_eval_dir
            / f"spatialwhisperer_{combo}"
            / "pannuke_summary"
            / "tissue_type_all_summary.csv"
        )
        summary_df = pd.read_csv(summary_path)
        out["pannuke_tissue/rocauc_macroAvg"] = float(
            summary_df["rocauc_macroAvg"].iloc[0]
        )

    return out


def get_baseline_combo(modality_pair: str, model: str) -> str:
    if model == "trimodal":
        return "cellxgene_census__archs4_geo__hest1k__quilt1m"
    if model == "bimodal_bridge":
        if modality_pair == "transcriptome-text":
            return "hest1k__quilt1m"
        if modality_pair == "transcriptome-image":
            return "cellxgene_census__archs4_geo__quilt1m"
        if modality_pair == "image-text":
            return "cellxgene_census__archs4_geo__hest1k"
        raise ValueError(f"Unknown modality_pair: {modality_pair}")
    raise ValueError(f"Unknown baseline model: {model}")


ncols = len(modality_pairs)
nrows = max(len(metrics_by_pair[mp]) for mp in modality_pairs)
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(1.85 * ncols, 1.6 * nrows),
    sharex=False,
    sharey=True,
)

pair_only_color = "#4e79a7"
with_bridge_color = "#e15759"
trimodal_all_color = "#59a14f"

for col, mp in enumerate(modality_pairs):
    metrics = metrics_by_pair[mp]

    pair_only_values_by_metric = {m: [] for m in metrics}
    with_bridge_values_by_metric = {m: [] for m in metrics}
    trimodal_all_values_by_metric = {m: [] for m in metrics}

    for ratio in ratios:
        combo_pair_only = build_combo(mp, ratio, include_bridge=False)
        combo_with_bridge = build_combo(mp, ratio, include_bridge=True)
        vals_pair_only = extract_metrics_for_combo(mp, combo_pair_only)
        vals_with_bridge = extract_metrics_for_combo(mp, combo_with_bridge)
        for m in metrics:
            pair_only_values_by_metric[m].append(vals_pair_only[m])
            with_bridge_values_by_metric[m].append(vals_with_bridge[m])
        if plot_trimodal_all_subset:
            combo_trimodal_all = build_trimodal_all_subset_combo(ratio)
            vals_trimodal_all = extract_metrics_for_combo(mp, combo_trimodal_all)
            for m in metrics:
                trimodal_all_values_by_metric[m].append(vals_trimodal_all[m])

    bridge_combo = get_baseline_combo(mp, model="bimodal_bridge")
    bridge_vals = extract_metrics_for_combo(mp, bridge_combo)

    for row in range(nrows):
        ax = axs[row, col] if nrows > 1 else axs[col]
        if row < len(metrics):
            metric = metrics[row]
            fracs = [1.0 / r for r in ratios]
            order_idx = sorted(range(len(ratios)), key=lambda i: fracs[i])
            x_fracs_sorted = [fracs[i] for i in order_idx]
            y_pair_sorted = [pair_only_values_by_metric[metric][i] for i in order_idx]
            y_bridge_sorted = [
                with_bridge_values_by_metric[metric][i] for i in order_idx
            ]

            x_with_pair = [1 / 4096] + x_fracs_sorted
            y_with_pair = [0.5] + y_pair_sorted
            ax.plot(
                x_with_pair,
                y_with_pair,
                marker="o",
                color=pair_only_color,
                label="pair-only" if (row == 0 and col == 0) else None,
            )
            x_with_bridge = [1 / 4096] + x_fracs_sorted
            y_with_bridge = [bridge_vals[metric]] + y_bridge_sorted
            ax.plot(
                x_with_bridge,
                y_with_bridge,
                marker="o",
                color=with_bridge_color,
                label="with-bridge" if (row == 0 and col == 0) else None,
            )
            if plot_trimodal_all_subset:
                y_trimodal_sorted = [
                    trimodal_all_values_by_metric[metric][i] for i in order_idx
                ]
                ax.plot(
                    x_fracs_sorted,
                    y_trimodal_sorted,
                    marker="o",
                    color=trimodal_all_color,
                    label="trimodal-all-subset" if (row == 0 and col == 0) else None,
                )
            ax.axhline(
                0.5,
                color="#7f7f7f",
                linestyle=":",
                linewidth=1.0,
                label="random baseline" if (row == 0 and col == 0) else None,
            )
            ax.set_ylim(bottom=0)
            ax.set_xscale("log")
            xticks = [1 / 4096] + x_fracs_sorted
            xticklabels = ["0"] + [
                ("1" if ratios[i] == 1 else f"1/{ratios[i]}") for i in order_idx
            ]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.minorticks_off()
            ax.set_xlim(1 / 4096, 1.0)
            ax.set_title(metric)
        else:
            ax.axis("off")
        ax.set_ylabel("")


handles, labels = (
    axs[0, 0].get_legend_handles_labels()
    if nrows > 1
    else axs[0].get_legend_handles_labels()
)
if handles:
    legend_ncol = 3 if plot_trimodal_all_subset else 2
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol)
    plt.subplots_adjust(bottom=0.1)

plt.tight_layout()
out_path = Path(snakemake.output.plot)
plt.savefig(out_path)
plt.savefig(out_path.with_suffix(".svg"))
plt.savefig(out_path.with_suffix(".pdf"))
