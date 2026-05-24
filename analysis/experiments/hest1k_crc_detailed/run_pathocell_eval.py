"""
Standalone PathoCellBench CRC evaluation script.
Bypasses snakemake DAG resolution issues on SNAP.

Usage:
  python run_pathocell_eval.py --model_path /path/to/model.ckpt --output_dir /path/to/output
"""

import pyarrow  # must be first (glibc compat)
import argparse
import json
import logging
import numpy as np
import pandas as pd
import anndata
import torch
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PathoCellBench CRC datasets (109 total)
DATASETS = [
    "reg006_B",
    "reg014_B",
    "reg022_B",
    "reg030_A",
    "reg037_B",
    "reg046_A",
    "reg056_A",
    "reg007_A",
    "reg015_A",
    "reg023_A",
    "reg030_B",
    "reg038_A",
    "reg047_A",
    "reg058_A",
    "reg007_B",
    "reg016_A",
    "reg023_B",
    "reg031_A",
    "reg039_A",
    "reg048_A",
    "reg059_A",
    "reg001_A",
    "reg008_A",
    "reg016_B",
    "reg024_B",
    "reg031_B",
    "reg039_B",
    "reg048_B",
    "reg059_B",
    "reg001_B",
    "reg008_B",
    "reg017_A",
    "reg025_A",
    "reg032_A",
    "reg040_A",
    "reg049_A",
    "reg060_A",
    "reg002_A",
    "reg009_A",
    "reg017_B",
    "reg025_B",
    "reg032_B",
    "reg040_B",
    "reg050_A",
    "reg060_B",
    "reg002_B",
    "reg009_B",
    "reg018_A",
    "reg026_A",
    "reg033_A",
    "reg041_A",
    "reg050_B",
    "reg061_A",
    "reg003_A",
    "reg010_A",
    "reg018_B",
    "reg026_B",
    "reg033_B",
    "reg041_B",
    "reg051_A",
    "reg062_A",
    "reg003_B",
    "reg010_B",
    "reg019_A",
    "reg027_A",
    "reg034_A",
    "reg042_A",
    "reg051_B",
    "reg063_A",
    "reg004_A",
    "reg011_A",
    "reg020_A",
    "reg027_B",
    "reg035_A",
    "reg042_B",
    "reg052_A",
    "reg064_A",
    "reg004_B",
    "reg011_B",
    "reg020_B",
    "reg028_A",
    "reg035_B",
    "reg043_A",
    "reg052_B",
    "reg065_A",
    "reg005_A",
    "reg012_A",
    "reg021_A",
    "reg028_B",
    "reg036_A",
    "reg044_A",
    "reg053_A",
    "reg066_A",
    "reg005_B",
    "reg012_B",
    "reg021_B",
    "reg029_A",
    "reg036_B",
    "reg045_A",
    "reg054_A",
    "reg067_A",
    "reg006_A",
    "reg013_B",
    "reg022_A",
    "reg029_B",
    "reg037_A",
    "reg045_B",
    "reg055_A",
    "reg068_A",
]


LABEL_REMAP = {
    "lizard": {
        "Neutrophil": "neutrophil",
        "Epithelial": "epithelial",
        "Lymphocyte": "lymphocyte",
        "Plasma": "plasma",
        "Eosinophil": "eosinophil",
        "Connective tissue": "fibroblast",
    },
    "pannuke": {
        "Epithelial": "epithelial",
        "Dead Cells": "dead cells",
        "Neoplastic cells": "cancer cells",
        "Inflammatory": "leukocytes",
        "Connective/Soft tissue cells": "fibroblasts",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--data_dir", required=True, help="Path to resources/pathocell/processed/"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prediction_level", default="patch")
    parser.add_argument(
        "--cell_type_col",
        default="cell_type",
        help="Column in adata.obs for cell type labels",
    )
    parser.add_argument(
        "--benchmark",
        default="crc",
        choices=["crc", "lizard", "pannuke"],
        help="Which benchmark to run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="Limit number of datasets for quick testing",
    )
    args = parser.parse_args()

    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    from cellwhisperer.utils.inference import score_left_vs_right

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    logger.info(f"Loading model from {args.model_path}")
    pl_model, text_processor, transcriptome_processor, image_processor = (
        load_cellwhisperer_model(model_path=args.model_path, eval=True)
    )

    # Discover datasets based on benchmark type
    if args.benchmark == "crc":
        datasets = DATASETS
        sub_dir = data_dir
    elif args.benchmark == "lizard":
        sub_dir = data_dir / "lizard"
        datasets = sorted(
            set(
                p.stem.replace(f"_{args.prediction_level}", "")
                for p in sub_dir.glob(f"*_{args.prediction_level}.h5ad")
            )
        )
    elif args.benchmark == "pannuke":
        sub_dir = data_dir / "pannuke"
        datasets = sorted(
            set(
                p.stem.replace(f"_{args.prediction_level}", "")
                for p in sub_dir.glob(f"*_{args.prediction_level}.h5ad")
            )
        )

    if args.max_datasets:
        datasets = datasets[: args.max_datasets]

    remap = LABEL_REMAP.get(args.benchmark, {})
    logger.info(
        f"Running {args.benchmark} evaluation on {len(datasets)} datasets, label remap: {remap}"
    )

    all_per_class = []
    all_per_dataset = []

    for dataset_name in tqdm(datasets, desc="Evaluating"):
        adata_path = sub_dir / f"{dataset_name}_{args.prediction_level}.h5ad"
        image_path = sub_dir / f"{dataset_name}_{args.prediction_level}.tiff"

        if not adata_path.exists() or not image_path.exists():
            logger.warning(f"Skipping {dataset_name}: missing files")
            continue

        adata = anndata.read_h5ad(adata_path)
        adata.uns["image_path"] = str(image_path)

        # Get cell type labels and apply remap
        raw_types = sorted(adata.obs[args.cell_type_col].unique().tolist())
        cell_types = [remap.get(t, t) for t in raw_types]
        # Also remap the obs column for AUROC computation
        adata.obs["_eval_cell_type"] = adata.obs[args.cell_type_col].map(
            lambda x: remap.get(x, x)
        )

        # Score: image (left) vs text labels (right)
        try:
            model = pl_model.model
            scores, _ = score_left_vs_right(
                left_input=adata,
                right_input=cell_types,
                model=model,
                logit_scale=model.discriminator.temperature.exp(),
                average_mode=None,
                batch_size=128,
                use_image_data=True,
            )
            scores = scores.numpy()
            # scores shape is (n_classes, n_cells) — transpose to (n_cells, n_classes)
            if scores.shape[0] == len(cell_types) and scores.shape[1] != len(
                cell_types
            ):
                scores = scores.T
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # scores is (n_cells, n_classes) numpy array
        scores_df = pd.DataFrame(scores, columns=cell_types, index=adata.obs_names)
        scores_df.to_csv(
            output_dir
            / f"{dataset_name}_{args.prediction_level}_scores_seed{args.seed}.csv"
        )

        # Compute per-class AUROC (presence-based)
        from sklearn.metrics import roc_auc_score

        true_labels = adata.obs["_eval_cell_type"].values
        per_class = {}
        for j, ct in enumerate(cell_types):
            y_bin = (true_labels == ct).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                per_class[ct] = {"rocauc": np.nan}
            else:
                per_class[ct] = {"rocauc": float(roc_auc_score(y_bin, scores[:, j]))}

        per_class_df = pd.DataFrame(per_class).T
        per_class_df["dataset"] = dataset_name
        all_per_class.append(per_class_df)

        macro_auroc = float(np.nanmean([v["rocauc"] for v in per_class.values()]))
        all_per_dataset.append(
            {"dataset": dataset_name, "rocauc_macroAvg": macro_auroc}
        )
        logger.info(f"{dataset_name}: macro AUROC = {macro_auroc:.4f}")

    # Aggregate
    per_dataset_df = pd.DataFrame(all_per_dataset)
    per_dataset_df.to_csv(output_dir / f"per_dataset_metrics.csv", index=False)

    per_class_all = pd.concat(all_per_class)
    per_class_all.to_csv(output_dir / f"per_class_by_dataset_metrics.csv")

    # Global macro AUROC: average per-dataset macro AUROCs
    global_macro_auroc = float(per_dataset_df["rocauc_macroAvg"].mean())

    # Per-class macro: average per-dataset-per-class AUROCs across datasets, then across classes
    per_class_mean = per_class_all.groupby(per_class_all.index)["rocauc"].mean()
    global_per_class_macro_auroc = float(per_class_mean.mean())

    summary = {
        "rocauc_macroAvg": global_macro_auroc,
        "rocauc_per_class_macro": global_per_class_macro_auroc,
        "n_datasets": len(per_dataset_df),
        "per_class_mean_auroc": per_class_mean.to_dict(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"RESULTS ({len(datasets)} datasets)")
    logger.info(f"  Macro AUROC (avg of per-dataset macros): {global_macro_auroc:.4f}")
    logger.info(f"  Per-class macro AUROC: {global_per_class_macro_auroc:.4f}")
    logger.info(f"  Per-class breakdown:")
    for ct, auroc in per_class_mean.items():
        logger.info(f"    {ct}: {auroc:.4f}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
