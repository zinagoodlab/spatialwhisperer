"""
Map CellXGene Census `cell_type` labels (482 classes used by the Geneformer
classifier in the two-stage baseline) onto the 5 PanNuke cell-type classes:
    Epithelial, Dead Cells, Connective/Soft tissue cells, Inflammatory,
    Neoplastic cells

Mirrors the regex-word-boundary approach used in
`create_pathocell_label_mapping.py`. The reduced 4-class scoring (drops
Dead Cells) is applied later by `compute_reduced_class_table2_style.py`.

Two notes:
- "Inflammatory" subsumes the union of CRC's Macrophages/Monocytes, T cells,
  B cells, NK cells, Granulocytes, Plasma cells, Dendritic cells, etc.
- "Neoplastic cells" has no good mapping from CellXGene Census normal-tissue
  labels; left unmapped (analogous to Tumor cells in the CRC mapping).
"""

import pandas as pd
import re

PANNUKE_CELL_TYPES = [
    "Epithelial",
    "Dead Cells",
    "Connective/Soft tissue cells",
    "Inflammatory",
    "Neoplastic cells",
]


def _make_pattern(keywords):
    if not keywords:
        return None
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# Order matters: more specific categories first. Inflammatory is broad, so
# Epithelial / Connective tissue are checked before it to avoid false hits.
MAPPING_RULES_ORDERED = [
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
                "alveolar",
                "ductal cell",
                "acinar cell",
                "luminal cell",
                "mammary epithelial",
            ]
        ),
    ),
    (
        "Connective/Soft tissue cells",
        _make_pattern(
            [
                "fibroblast",
                "stromal",
                "mesenchymal stem cell",
                "pericyte",
                "stellate cell",
                "myofibroblast",
                "smooth muscle",
                "muscle cell",
                "myocyte",
                "cardiomyocyte",
                "myoblast",
                "endothelial",
                "vascular",
            ]
        ),
    ),
    (
        "Inflammatory",
        _make_pattern(
            [
                # Whole immune lineage maps to "Inflammatory" in PanNuke.
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
                "macrophage",
                "monocyte",
                "dendritic cell",
                "kupffer",
                "langerhans",
                "microglia",
                "histiocyte",
                "osteoclast",
                "neutrophil",
                "eosinophil",
                "basophil",
                "mast cell",
                "granulocyte",
                "plasma cell",
                "plasmablast",
            ]
        ),
    ),
    # No good mapping from normal-tissue CellXGene types:
    ("Neoplastic cells", None),
    ("Dead Cells", None),
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
print(f"Mapped {n_mapped}/{len(df)} training cell types to PanNuke categories")
for cat in PANNUKE_CELL_TYPES:
    n = (df["evaluation_cell_type"] == cat).sum()
    if n > 0:
        print(f"  {cat}: {n} types")

df.to_csv(snakemake.output.label_mapping, index=False)
