from pathlib import Path
import argparse
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.results_root.glob("*/*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "model": metrics["model"],
                "caption_condition": metrics["caption_condition"],
                "macro_auroc_mean": metrics["macro_auroc_mean"],
                "image_to_text_rocauc_macroAvg": metrics["image_to_text"][
                    "rocauc_macroAvg"
                ],
                "text_to_image_rocauc_macroAvg": metrics["text_to_image"][
                    "rocauc_macroAvg"
                ],
                "n_samples": metrics["n_samples"],
            }
        )

    long_df = pd.DataFrame(rows).sort_values(["model", "caption_condition"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_csv.parent / "control_summary_long.csv", index=False)

    summary = (
        long_df.pivot(
            index="model", columns="caption_condition", values="macro_auroc_mean"
        )
        .reset_index()
        .rename(
            columns={
                "original": "original_captions_auroc",
                "curated": "curated_captions_auroc",
            }
        )
    )
    if "original_captions_auroc" in summary and "curated_captions_auroc" in summary:
        summary["delta"] = (
            summary["curated_captions_auroc"] - summary["original_captions_auroc"]
        )
    summary.to_csv(args.output_csv, index=False)

    md_lines = [
        "| Model | Original captions (AUROC) | Curated captions (AUROC) | Delta |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {str(row['model']).upper()} | {row['original_captions_auroc']:.3f} | {row['curated_captions_auroc']:.3f} | {row['delta']:+.3f} |"
        )
    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
