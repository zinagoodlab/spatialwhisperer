"""
Create label mapping from CellXGene Census cell types to PathoCellBench CRC cell types.

Uses regex word-boundary matching: keywords must appear as whole words or at
word boundaries in the training cell type name. Checked in priority order
(more specific categories first) to avoid ambiguous matches.
"""

import pandas as pd
import re

CRC_CELL_TYPES = [
    "B cells",
    "T cells",
    "Tumor cells",
    "Stroma",
    "Smooth muscle",
    "Nerves",
    "Plasma cells",
    "Granulocytes",
    "Macrophages/Monocytes",
    "NK cells",
    "Endothelial",
    "Muscle",
    "Epithelial",
    "Neuroendocrine",
    "Dead cells",
]


def _make_pattern(keywords):
    """Compile keywords into a single regex with word boundaries."""
    if not keywords:
        return None
    escaped = [re.escape(kw) for kw in keywords]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# Order matters: more specific categories first.
# Each keyword is matched with word boundaries (\b) so e.g. "T cell" won't
# match "goblet cell" or "mast cell".
MAPPING_RULES_ORDERED = [
    (
        "Plasma cells",
        _make_pattern(
            [
                "plasma cell",
                "plasmablast",
            ]
        ),
    ),
    (
        "NK cells",
        _make_pattern(
            [
                "natural killer",
                "NK cell",
                "NK T cell",
                "innate lymphoid",
            ]
        ),
    ),
    (
        "B cells",
        _make_pattern(
            [
                # \bB cell\b won't match "club cell" or "goblet cell"
                "B cell",
                "B-cell",
                "B lymphocyte",
                "pro-B",
                "pre-B",
                "marginal zone B",
                "follicular B",
                "germinal center B",
            ]
        ),
    ),
    (
        "T cells",
        _make_pattern(
            [
                # \bT cell\b won't match "goblet cell", "mast cell", "fat cell"
                "T cell",
                "T-cell",
                "T lymphocyte",
                "thymocyte",
                "thymus-derived",
                "alpha-beta T",
                "gamma-delta T",
                "regulatory T",
                "CD4-positive",
                "CD8-positive",
                "T-helper",
                "T follicular",
                "cytotoxic T",
                "helper T",
                "memory T",
                "naive T",
                "mature T",
                "effector T",
                "exhausted T",
                "pro-T",
                "mucosal invariant T",
            ]
        ),
    ),
    (
        "Granulocytes",
        _make_pattern(
            [
                "neutrophil",
                "eosinophil",
                "basophil",
                "mast cell",
                "granulocyte",
            ]
        ),
    ),
    (
        "Macrophages/Monocytes",
        _make_pattern(
            [
                "macrophage",
                "monocyte",
                "dendritic cell",
                "kupffer",
                "langerhans",
                "microglia",
                "histiocyte",
                "osteoclast",
            ]
        ),
    ),
    (
        "Neuroendocrine",
        _make_pattern(
            [
                "neuroendocrine",
                "enteroendocrine",
                "chromaffin",
            ]
        ),
    ),
    (
        "Nerves",
        _make_pattern(
            [
                "neuron",
                "neural progenitor",
                "neural cell",
                "glial cell",
                "Schwann",
                "astrocyte",
                "oligodendrocyte",
                "radial glial",
            ]
        ),
    ),
    (
        "Smooth muscle",
        _make_pattern(
            [
                "smooth muscle",
            ]
        ),
    ),
    (
        "Muscle",
        _make_pattern(
            [
                "muscle",
                "myofibroblast",
                "myocyte",
                "cardiomyocyte",
                "myoblast",
            ]
        ),
    ),
    (
        "Endothelial",
        _make_pattern(
            [
                "endothelial",
            ]
        ),
    ),
    (
        "Epithelial",
        _make_pattern(
            [
                "epithelial",
                "epithelium",
                "enterocyte",
                "goblet",
                "paneth",
                "colonocyte",
                "keratinocyte",
                "basal cell",
                "secretory cell",
                "club cell",
                "ciliated cell",
                "ciliated epithelial",
                "hepatocyte",
                "cholangiocyte",
                "podocyte",
                "tubular cell",
                "tubule epithelial",
                "tuft cell",
            ]
        ),
    ),
    (
        "Stroma",
        _make_pattern(
            [
                "fibroblast",
                "stromal",
                "mesenchymal stem cell",
                "pericyte",
                "stellate cell",
            ]
        ),
    ),
    # No good mapping from normal tissue:
    ("Tumor cells", None),
    ("Dead cells", None),
]

# Load existing label transfer to get the list of training cell types
ts_mapping = pd.read_csv(snakemake.input.transfered_labels)
training_cell_types = ts_mapping["training_cell_type"].tolist()

# Map each training cell type to a CRC category using regex matching
rows = []
for training_ct in training_cell_types:
    matched_eval = "none"
    for eval_ct, pattern in MAPPING_RULES_ORDERED:
        if pattern is not None and pattern.search(training_ct):
            matched_eval = eval_ct
            break
    rows.append(
        {
            "training_cell_type": training_ct,
            "evaluation_cell_type": matched_eval,
        }
    )

df = pd.DataFrame(rows)
n_mapped = (df["evaluation_cell_type"] != "none").sum()
print(f"Mapped {n_mapped}/{len(df)} training cell types to CRC categories")
for cat in CRC_CELL_TYPES:
    n = (df["evaluation_cell_type"] == cat).sum()
    if n > 0:
        print(f"  {cat}: {n} types")

df.to_csv(snakemake.output.label_mapping, index=False)
