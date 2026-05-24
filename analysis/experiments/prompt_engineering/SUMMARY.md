# Prompt Engineering Experiment for Zero-Shot Cell Type Classification

## Status

**Complete.** Controller job 19682379 finished 2026-03-27.

### TODO

1. Decide whether to include in the appendix or just the rebuttal letter
2. Write the corresponding paper text

## Motivation (Reviewer Request)

> For the first experiment, the text prompts used for zero-shot classification
> are the raw class labels (Appendix B shows that formulating a simple phrase did
> not improve the results). However, is it possible to formulate more
> sophisticated prompts that could improve the results? [...] different prompt
> designs could be optimal [...] and a similar alignment as in the text
> harmonization experiments (Section 4.3) could be considered.

We address this in two parts:

1. **Prior experiment (branch `spatialwhisperer-revision`)**: A label
   sensitivity analysis on the CRC benchmark tested 12 synonym/domain-specific
   remappings for problematic cell types (Nerves, NK cells, Granulocytes).
   Result: label naming has minimal impact (AUROC range 0.639–0.647 across 12
   term sets; original = 0.643).

2. **This experiment**: We test 4 prompt *templates* that wrap every class label
   in a contextualizing sentence. Two templates are image-focused (the kind of
   prompt a pathology VLM would be trained on), two are transcriptomics-focused
   (reflecting the gene-expression–text alignment in our transitive training).

## Setup

| Parameter | Value |
|---|---|
| Model | `spotwhisperer_cellxgene_census__archs4_geo__hest1k` (bimodal bridge) |
| Benchmark | CRC (PathoCellBench, 109 datasets, 15 cell types) |
| Seeds | 1 (seed 0; consistent with baseline) |
| Prediction level | Patch |
| Templates | 4 (see below) + original (raw labels) |

### Prompt Templates

| Template ID | Template | Rationale |
|---|---|---|
| `original` | `{label}` (raw class name) | Baseline (already computed) |
| `hne_tissue_image` | `An H&E stained tissue image showing {label}` | Image-centric; natural pathology caption style |
| `histopath_image_of` | `histopathology image of {label}` | CONCH/MUSK benchmark template; common in pathology VLMs |
| `tissue_sample_containing` | `a tissue sample containing {label}` | Transcriptomics-aligned; emphasizes sample/tissue context |
| `gene_expression_profiling` | `cells classified as {label} by gene expression profiling` | Explicitly transcriptomics-aligned; mirrors transitive training signal |

### CRC Class Names (PathoCellBench)

The 15 class labels used as `{label}` in each template:

Adipocytes, B cells, Background, Dendritic cells, Granulocytes,
Macrophages/Monocytes, NK cells, Nerves, Other cells, Plasma cells, Smooth
muscle, Stroma, T cells, Tumor cells, Vasculature/Lymphatics

## Implementation

Self-contained experiment under `src/spotwhisperer_eval/experiments/prompt_engineering/`.
Does **not** modify the main pipeline.

### File layout

```
src/spotwhisperer_eval/experiments/prompt_engineering/
├── SUMMARY.md                           # This file
├── Snakefile                            # Self-contained; invoked independently
└── scripts/
    └── prompt_template_summary.py       # Collects per-template JSONs → comparison CSV
```

### How it works

- Reuses the existing `pathocell_cell_type_prediction.py.ipynb` notebook via
  the `label_remap` parameter (notebook path: `../../notebooks/`).
- Reuses the existing `compute_pathocell_metrics_from_scores.py` script for
  per-template metric aggregation (script path: `../../scripts/`).
- For each template, a `label_remap` dict maps every original class name to its
  template-wrapped form. The text encoder receives the full template string.
- Scores CSVs use original class names as column headers (not template-wrapped);
  only the text embedding changes.
- 109 CRC datasets x 4 templates x 1 seed = 436 prediction jobs, plus 4
  aggregation jobs and 1 summary job.

### How to run

From Sherlock, within a SLURM job:

```bash
cd ~/cellwhisperer_private/src/spotwhisperer_eval/experiments/prompt_engineering
conda activate cellwhisperer
snakemake --profile sm7_slurm --cores 4
```

### Output paths (on Sherlock)

```
results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k/
├── prompt_hne_tissue_image/              # Per-dataset predictions
├── prompt_histopath_image_of/
├── prompt_tissue_sample_containing/
├── prompt_gene_expression_profiling/
├── prompt_hne_tissue_image_summary/      # Aggregated metrics
├── prompt_histopath_image_of_summary/
├── prompt_tissue_sample_containing_summary/
├── prompt_gene_expression_profiling_summary/
└── prompt_template_summary/
    └── patch_prompt_comparison.csv       # Final comparison table
```

## Results

### Part 1: Label sensitivity (12 synonym remappings, 15-dataset subset)

From branch `spatialwhisperer-revision`. Tested renaming individual
hard-to-classify cell types (Nerves, NK cells, Granulocytes) to domain-specific
synonyms.

| Term set | AUROC (macro) | F1 (macro) | Soft AUROC (macro) |
|---|---|---|---|
| original | 0.643 | 0.069 | 0.467 |
| nk_natural_killer_cells | 0.643 | 0.070 | 0.467 |
| nerves_nerve_fibers | 0.643 | 0.069 | 0.466 |
| nerves_enteric_neurons | 0.643 | 0.067 | 0.467 |
| nerves_schwann_cells | 0.640 | 0.065 | 0.464 |
| nerves_neural_cells | 0.643 | 0.070 | 0.466 |
| nerves_ganglion_cells | 0.639 | 0.068 | 0.464 |
| nerves_glial_cells | 0.640 | 0.070 | 0.463 |
| nerves_peripheral_nerves | 0.641 | 0.072 | 0.465 |
| nerves_all | 0.641 | 0.065 | 0.464 |
| granulocytes_neutrophils | 0.642 | 0.066 | 0.468 |
| granulocytes_eosinophils | 0.647 | 0.063 | 0.466 |

Label synonyms produce <1% AUROC variation, ruling out label naming as the
cause of low performance on rare cell types (Nerves, NK cells, Granulocytes).

### Part 2: Prompt template experiment — bimodal bridge model

**Note on class set:** The paper reports AUROC averaged over 13 cell types
(excluding Background and Other cells). The `compute_pathocell_metrics_from_scores.py`
script includes all 15 classes. The table below reports **13-class AUROC**
(consistent with the paper) recomputed from the per-class-by-dataset CSVs.

Results across all 109 CRC datasets (sorted by AUROC, 13 classes):

| Template | AUROC (macro, 13-class) |
|---|---|
| `tissue_sample_containing` | **0.647** |
| `gene_expression_profiling` | 0.633 |
| `hne_tissue_image` | 0.633 |
| `original` (baseline) | 0.630 |
| `histopath_image_of` | 0.628 |

### Part 3: Best template on trimodal curated model

Ran `tissue_sample_containing` (best template from Part 2) on
`spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_curated`.
Controller job 19747145, completed 2026-03-28.

| Model | AUROC (raw labels) | AUROC (best template) | Delta |
|---|---|---|---|
| bimodal bridge | 0.630 | **0.647** | +1.7 pp |
| trimodal curated | 0.645 | 0.642 | -0.3 pp |

The best template improves the bimodal bridge (+1.7 pp) but has no meaningful
effect on the trimodal curated model (-0.3 pp), consistent with the latter
already being better aligned through its richer training data.

## Conclusions

The best-performing template (`tissue_sample_containing`) yields a modest +1.7 pp
AUROC gain for the bimodal bridge model. The other three templates produce smaller
or negligible improvements (+0.3 pp or less). Across models, the effect vanishes
for the trimodal curated model, which is already stronger with raw labels.

Critically, these gains are small and consistent with the label sensitivity
finding (Part 1, <1% AUROC variation across 12 synonym sets). This confirms
that **prompt phrasing has limited impact on zero-shot performance**.

For the rebuttal: this result shows that performance is robust to prompt design.
Even purpose-designed prompts (including CONCH/MUSK's own recommended template)
yield only marginal gains, ruling out suboptimal prompting as an explanation for
the observed limitations.
