"""
Plot confusion matrix with hierarchical clustering from PathoCellBench CRC scores.
Aggregates scores across all 109 datasets, then computes confusion matrix from argmax predictions.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import confusion_matrix
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_dir", required=True, help="Directory with per-dataset score CSVs"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory with processed h5ad files"
    )
    parser.add_argument("--output", required=True, help="Output SVG/PNG path")
    parser.add_argument(
        "--cell_type_col", default="cell_type", help="Column for true labels"
    )
    parser.add_argument("--style", default=None, help="Matplotlib style file")
    args = parser.parse_args()

    if args.style:
        matplotlib.style.use(args.style)

    scores_dir = Path(args.scores_dir)
    data_dir = Path(args.data_dir)

    # Collect all scores and true labels
    all_true = []
    all_pred_scores = []
    all_classes = set()

    for csv_path in sorted(scores_dir.glob("*_patch_scores_seed0.csv")):
        dataset_name = csv_path.stem.replace("_patch_scores_seed0", "")
        adata_path = data_dir / f"{dataset_name}_patch.h5ad"

        if not adata_path.exists():
            continue

        import anndata

        adata = anndata.read_h5ad(adata_path)
        scores_df = pd.read_csv(csv_path, index_col=0)

        true_labels = adata.obs[args.cell_type_col].values
        all_true.extend(true_labels)
        all_pred_scores.append(scores_df)
        all_classes.update(true_labels)
        all_classes.update(scores_df.columns)

    # Combine all scores — align columns (each dataset may have a different class subset)
    all_scores = pd.concat(all_pred_scores, axis=0)
    all_true = np.array(all_true)

    # Z-score normalize logits per class (across cells) to remove class-level score bias
    # Only use valid (non-NaN) values per column for mean/std; NaN stays NaN
    scores_values = all_scores.values.copy()
    for j in range(scores_values.shape[1]):
        col = scores_values[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            m, s = col[valid].mean(), col[valid].std()
            if s > 0:
                scores_values[valid, j] = (col[valid] - m) / s
            else:
                scores_values[valid, j] = 0
        # NaN stays NaN — these cells didn't have this class in their dataset

    # For argmax: replace NaN with -inf so they're never selected
    scores_for_argmax = np.where(np.isnan(scores_values), -np.inf, scores_values)

    # Predicted labels from argmax of z-scored logits
    pred_labels = all_scores.columns[scores_for_argmax.argmax(axis=1)]

    # Get sorted class list (union of true and predicted)
    classes = sorted(all_classes)

    # Confusion matrix (rows = true, cols = predicted)
    cm = confusion_matrix(all_true, pred_labels, labels=classes)

    # Normalize by row (true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Hierarchical clustering on rows
    if cm_norm.shape[0] > 2:
        row_linkage = linkage(cm_norm, method="average", metric="cosine")
        row_dendro = dendrogram(row_linkage, no_plot=True)
        row_order = row_dendro["leaves"]
    else:
        row_order = list(range(cm_norm.shape[0]))

    # Reorder both rows and columns by the same clustering
    cm_clustered = cm_norm[row_order][:, row_order]
    classes_ordered = [classes[i] for i in row_order]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_clustered, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes_ordered)))
    ax.set_yticks(range(len(classes_ordered)))
    ax.set_xticklabels(classes_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(classes_ordered, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "PathoCellBench CRC: Confusion Matrix (normalized, hierarchically clustered)"
    )

    # Add text annotations for values > 0.05
    for i in range(len(classes_ordered)):
        for j in range(len(classes_ordered)):
            val = cm_clustered[i, j]
            if val > 0.05:
                color = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=color,
                )

    plt.colorbar(im, ax=ax, shrink=0.8, label="Fraction of true class")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix to {output_path}")

    # Also save the raw confusion matrix as CSV
    cm_df = pd.DataFrame(cm_clustered, index=classes_ordered, columns=classes_ordered)
    cm_df.to_csv(output_path.with_suffix(".csv"))
    print(f"Saved CSV to {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
