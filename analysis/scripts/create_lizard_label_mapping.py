"""
Map CellXGene Census `cell_type` labels (482 classes used by the Geneformer
classifier in the two-stage baseline) onto the 6 Lizard cell-type classes:
    Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue

Mirrors the regex-word-boundary approach used in
`create_pathocell_label_mapping.py`; only the rule set differs.
The reduced 3-class scoring (Epithelial, Leukocyte, Fibroblast) is applied
later by `compute_reduced_class_table2_style.py`.
"""

import pandas as pd
import re

LIZARD_CELL_TYPES = [
    "Neutrophil",
    "Epithelial",
    "Lymphocyte",
    "Plasma",
    "Eosinophil",
    "Connective tissue",
]


def _make_pattern(keywords):
    if not keywords:
        return None
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# Order matters: more specific categories first.
MAPPING_RULES_ORDERED = [
    (
        "Plasma",
        _make_pattern(
            [
                "plasma cell",
                "plasmablast",
            ]
        ),
    ),
    (
        "Neutrophil",
        _make_pattern(
            [
                "neutrophil",
            ]
        ),
    ),
    (
        "Eosinophil",
        _make_pattern(
            [
                "eosinophil",
            ]
        ),
    ),
    (
        "Lymphocyte",
        _make_pattern(
            [
                # Lizard's Lymphocyte covers all T/B/NK lineages.
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
                "B cell",
                "B-cell",
                "B lymphocyte",
                "pro-B",
                "pre-B",
                "marginal zone B",
                "follicular B",
                "germinal center B",
                "natural killer",
                "NK cell",
                "NK T cell",
                "innate lymphoid",
                "lymphocyte",
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
        "Connective tissue",
        _make_pattern(
            [
                "fibroblast",
                "stromal",
                "mesenchymal stem cell",
                "pericyte",
                "stellate cell",
                "myofibroblast",
            ]
        ),
    ),
]

ts_mapping = pd.read_csv(snakemake.input.transfered_labels)
training_cell_types = ts_mapping["training_cell_type"].tolist()

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
print(f"Mapped {n_mapped}/{len(df)} training cell types to Lizard categories")
for cat in LIZARD_CELL_TYPES:
    n = (df["evaluation_cell_type"] == cat).sum()
    if n > 0:
        print(f"  {cat}: {n} types")

df.to_csv(snakemake.output.label_mapping, index=False)
