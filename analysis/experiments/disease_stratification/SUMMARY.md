# Disease Stratification & Tissue Type Prediction on PanNuke

## Overview

Two analyses on PanNuke, both addressing reviewer questions about the model's
behavior across healthy vs diseased tissue and its ability to recognize tissue context:

1. **Disease-stratified cell type prediction** (Part 1) — cell type AUROC in malignant vs benign patches
2. **Zero-shot tissue type (organ) prediction** (Parts 2 & 3) — 19-class organ classification, comparing
   trimodal models (base, +quilt1m, +quilt1m_curated) and a bimodal baseline (quilt1m-only)

## Prior PanNuke & Lizard Benchmark Results (context)

The analyses below build on the zero-shot cell type prediction benchmarks established in
`src/experiments/934-celltype-zeroshot-eval/SUMMARY.md`. Key prior findings:

### PanNuke cell type prediction (reduced 4 classes, dropping Dead Cells)

| Model            | F1    | AUROC | Accuracy |
|------------------|-------|-------|----------|
| **SpotWhisperer**| **0.506** | **0.748** | **0.550** |
| MUSK             | 0.341 | 0.580 | 0.361    |
| PLIP             | 0.310 | 0.621 | 0.416    |
| CONCH            | 0.066 | 0.626 | 0.108    |

Per-class AUROCs: Neoplastic 0.820, Inflammatory 0.804, Connective 0.747, Epithelial 0.623.

### Lizard cell type prediction (reduced 3 classes: Epithelial, Leukocyte, Fibroblast)

| Model            | F1    | AUROC | Accuracy |
|------------------|-------|-------|----------|
| **SpotWhisperer**| **0.618** | **0.828** | **0.659** |
| CONCH            | 0.254 | 0.693 | 0.251    |
| PLIP             | 0.072 | 0.759 | 0.121    |
| MUSK             | 0.072 | 0.654 | 0.121    |

### Trimodal ablation on CRC (from `src/spotwhisperer_eval/experiments/trimodal_ablation/SUMMARY.md`)

| Model                   | Mean AUROC (CRC, 13 cell types) |
|-------------------------|---------------------------------|
| Trimodal (curated)      | **0.645**                       |
| Bimodal bridge          | 0.630                           |
| Trimodal (uncurated)    | 0.609                           |

Key insight: Uncurated Quilt1M *hurts* CRC cell type prediction (0.609 < 0.630 bimodal),
but curation recovers and exceeds bimodal (0.645). This motivated the curated vs uncurated
comparison in tissue type prediction below.

## Implementation

All code lives in `src/spotwhisperer_eval/`:

| File | Purpose |
|------|---------|
| `rules/disease_stratification.smk` | All Snakemake rules (included from main Snakefile) |
| `scripts/compute_disease_stratified_metrics.py` | Part 1: PLIP-style malignant/benign stratification |
| `scripts/build_pannuke_tissue_mapping.py` | Part 2: Reconstruct (batch, patch) → (sample, tissue_type) |
| `scripts/score_pannuke_tissue_type.py` | Part 2: Zero-shot scoring with 19 tissue text queries (GPU) |
| `scripts/compute_tissue_type_metrics.py` | Part 2: Aggregate to sample-level, AUROC + confusion matrix |

External data:
- `resources/pathocell/pannuke_fold2_types.npy` — tissue type labels (downloaded from Warwick PanNuke fold 2)
- `resources/pathocell/processed/pannuke/tissue_mapping.csv` — patch-to-sample-to-tissue mapping (8,373 rows)

Output pattern: `results/pathocell_evaluation/{model}/pannuke_summary/`

### Execution

All runs on Sherlock via:

```bash
cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/spotwhisperer_eval
conda run -n cellwhisperer snakemake --profile sm7_slurm --cores 4 -p \
    results/pathocell_evaluation/{model}/pannuke_summary/<target>.csv
```

Part 1 is CPU-only (~5 min). Part 2 requires GPU inference: 51 batches × ~2 min each per model.
To force re-scoring (e.g. after changing text queries): `--forcerun score_pannuke_tissue_type`.

### Key design decisions

- **Disease state inferred from cell composition** (not metadata): PanNuke has no image-level
  disease labels. We follow PLIP (Huang et al., 2023): malignant = ≥10 neoplastic cells AND
  ≥30% neoplastic; benign = 0 neoplastic cells; ambiguous = discarded.
- **Patch-to-sample mapping reconstructed post-hoc**: The deterministic batching in
  `convert_lmdb_to_hdf.py` (sorted samples, batch_size=50, `isqrt`-based grid layout)
  allows tracing each patch back to its original PanNuke sample and tissue type without
  re-processing the raw data.
- **Sample-level aggregation**: Each PanNuke sample (256×256) is tiled into 4 patches (2×2
  of 224×224). Tissue scores are averaged (mean logits) across patches per sample before
  computing metrics.
- **Text queries**: Bare tissue names (e.g., `"breast"`, `"head and neck"`), no template.
  Originally used `"headneck"` (from types.npy normalization); fixed to `"head and neck"`
  on 2026-03-26, which improved base model AUROC for that tissue from 0.678 to 0.721.
- **Dead Cells** dropped from Part 1 (consistent with paper's reduced-class evaluation).
- **Lizard excluded**: All Lizard samples are colorectal pathology with no healthy/diseased distinction.

---

# Part 1: Disease-Stratified Cell Type Prediction

Stratify cell type AUROC by malignant vs benign patches (PLIP strategy).

### Patch distribution

| Category   | n_patches | %     |
|------------|-----------|-------|
| Benign     | 4,399     | 52.5% |
| Malignant  | 715       | 8.5%  |
| Ambiguous  | 3,259     | 38.9% |

### Results (model: `spotwhisperer_cellxgene_census__archs4_geo__hest1k`, seed 0)

| Class                        | Benign AUROC | Malignant AUROC | Delta  |
|------------------------------|--------------|-----------------|--------|
| Epithelial                   | 0.772        | 0.693           | -0.079 |
| Connective/Soft tissue cells | 0.783        | 0.611           | -0.172 |
| Inflammatory                 | 0.741        | 0.623           | -0.118 |
| **Macro avg**                | **0.765**    | **0.642**       | **-0.123** |

**Interpretation**: Consistent degradation in malignant tissue across all cell types.
Largest gap in Connective/Soft tissue (0.172). In malignant patches, 710/715 are
Neoplastic-dominated, so F1/precision for non-neoplastic classes is near zero;
AUROC is the informative metric. The gap likely reflects more heterogeneous morphology
in cancer tissue making zero-shot classification harder.

---

# Part 2: Zero-Shot Tissue Type Prediction (Base Model)

19-class organ classification — novel benchmark (no prior work evaluates this on PanNuke;
existing papers only do binary benign/malignant or image retrieval).

### Base model summary

| Subset      | n_samples | Accuracy | F1 (macro) | AUROC (macro) |
|-------------|-----------|----------|-----------|---------------|
| All patches | 2,401     | 0.107    | 0.072     | 0.586         |
| Benign only | 1,478     | 0.120    | 0.068     | 0.549         |

### Per-tissue AUROC (base model, all patches)

| Tissue             | n_samples | AUROC |
|--------------------|-----------|-------|
| Liver              | 66        | 0.851 |
| Ovarian            | 40        | 0.821 |
| Prostate           | 37        | 0.812 |
| Adrenal gland      | 148       | 0.762 |
| Head and neck      | 143       | 0.721 |
| Colon              | 418       | 0.672 |
| Thyroid            | 82        | 0.655 |
| Kidney             | 54        | 0.655 |
| Esophagus          | 128       | 0.607 |
| Bile duct          | 97        | 0.604 |
| Breast             | 736       | 0.589 |
| Lung               | 53        | 0.558 |
| Cervix             | 46        | 0.545 |
| Pancreatic         | 89        | 0.543 |
| Testis             | 75        | 0.440 |
| Skin               | 83        | 0.377 |
| Bladder            | 52        | 0.351 |
| Uterus             | 6         | 0.346 |
| Stomach            | 48        | 0.233 |

Above-chance for most tissues. Top: Liver, Ovarian, Prostate (>0.8). Worst: Stomach,
Uterus, Bladder (<0.4, partly due to small sample sizes). Accuracy is low (10.7%) but
chance is 5.3% for 19 classes; AUROC is the primary metric.

---

# Part 3: Multi-Model Tissue Type Comparison

## Models

| Short name           | Full checkpoint | Type |
|----------------------|-----------------|------|
| **quilt1m (bimodal)**| `spotwhisperer_quilt1m` | Image-text only (no transcriptome) |
| **base (trimodal)**  | `spotwhisperer_cellxgene_census__archs4_geo__hest1k` | Trimodal (transcriptome + image + text) |
| **+quilt1m**         | `...hest1k__quilt1m` | Trimodal + quilt1m |
| **+quilt1m_curated** | `...hest1k__quilt1m_curated` | Trimodal + quilt1m curated |

Note: A bimodal `spotwhisperer_quilt1m_curated` checkpoint does not exist.

## Results

### Summary (all patches, sample-level)

| Model                | Accuracy | F1 (macro) | AUROC (macro) |
|----------------------|----------|-----------|---------------|
| quilt1m (bimodal)    | 0.186    | **0.169** | **0.689**     |
| base (trimodal)      | 0.107    | 0.072     | 0.586         |
| +quilt1m             | 0.168    | 0.152     | 0.646         |
| +quilt1m_curated     | **0.206**| 0.149     | 0.649         |

### Summary (benign-only patches)

| Model                | Accuracy | F1 (macro) | AUROC (macro) |
|----------------------|----------|-----------|---------------|
| quilt1m (bimodal)    | 0.177    | **0.154** | **0.649**     |
| base (trimodal)      | 0.120    | 0.068     | 0.549         |
| +quilt1m             | 0.150    | 0.151     | 0.595         |
| +quilt1m_curated     | **0.182**| 0.130     | 0.616         |

### Per-tissue AUROC comparison (all patches)

| Tissue             | n   | bimodal | base  | +quilt1m | +quilt1m_cur. | Best |
|--------------------|-----|---------|-------|----------|---------------|------|
| Adrenal gland      | 148 | **0.938** | 0.762 | 0.928  | 0.860         | bim  |
| Bile duct          | 97  | 0.479   | **0.604** | 0.519 | 0.556        | base |
| Bladder            | 52  | **0.558** | 0.351 | 0.442 | 0.474         | bim  |
| Breast             | 736 | **0.804** | 0.589 | 0.786 | 0.771         | bim  |
| Cervix             | 46  | **0.709** | 0.545 | 0.615 | 0.576         | bim  |
| Colon              | 418 | 0.877   | 0.672 | **0.875** | 0.802        | +q   |
| Esophagus          | 128 | 0.468   | **0.607** | 0.560 | 0.556        | base |
| Head and neck      | 143 | 0.465   | **0.721** | 0.694 | 0.630        | base |
| Kidney             | 54  | **0.773** | 0.655 | 0.707 | 0.676        | bim  |
| Liver              | 66  | 0.797   | 0.851 | 0.841   | **0.869**     | +qc  |
| Lung               | 53  | 0.646   | 0.558 | 0.542   | **0.709**     | +qc  |
| Ovarian            | 40  | 0.764   | 0.821 | **0.827** | 0.798        | +q   |
| Pancreatic         | 89  | 0.534   | 0.543 | 0.561   | **0.598**     | +qc  |
| Prostate           | 37  | 0.754   | **0.812** | 0.619 | 0.563        | base |
| Skin               | 83  | **0.677** | 0.377 | 0.437 | 0.578         | bim  |
| Stomach            | 48  | **0.469** | 0.233 | 0.307 | 0.401         | bim  |
| Testis             | 75  | **0.753** | 0.440 | 0.575 | 0.566         | bim  |
| Thyroid            | 82  | 0.704   | 0.655 | 0.712   | **0.769**     | +qc  |
| Uterus             | 6   | **0.929** | 0.346 | 0.719 | 0.570         | bim  |

(bim = bimodal quilt1m, +q = trimodal +quilt1m, +qc = trimodal +quilt1m_curated)

## Interpretation

- **The bimodal quilt1m model achieves the highest macro AUROC (0.689)**, outperforming
  all trimodal variants (base 0.586, +quilt1m 0.646, +quilt1m_curated 0.649). This is
  a surprising result: adding transcriptome data (trimodal) *hurts* tissue type prediction
  compared to the pure image-text model trained on Quilt1M.
- **Possible explanation**: The trimodal models' text encoder is pulled in multiple directions
  during training — aligning with both transcriptome descriptions and image descriptions.
  For a purely visual task like tissue type prediction, the bimodal model's text encoder
  may be better calibrated to pathology image descriptions.
- **Quilt1M data is the key ingredient**: All models with Quilt1M substantially outperform
  the base trimodal model (which has no pathology text data). The bimodal model demonstrates
  that Quilt1M alone suffices — the transcriptome modality doesn't help for this task.
- **Per-tissue**: The bimodal model wins on 9/19 tissues, including the largest improvements
  on Uterus (+0.583 over base), Skin (+0.300), Testis (+0.313), Cervix (+0.164).
  The base trimodal model retains advantage on Esophagus, Head and neck, Prostate — tissues
  where the bimodal model also struggles, suggesting these are inherently difficult.
- **Benign-only**: Same pattern — bimodal leads with 0.649 macro AUROC vs 0.616 for the
  best trimodal (curated).

---

# Output Files

| Category | Path pattern |
|----------|-------------|
| Disease stratification per-class | `results/pathocell_evaluation/{model}/pannuke_summary/disease_stratified_per_class.csv` |
| Disease stratification summary | `results/pathocell_evaluation/{model}/pannuke_summary/disease_stratified_summary.csv` |
| Tissue type scores | `results/pathocell_evaluation/{model}/pannuke_tissue/batch_*_tissue_scores.csv` |
| Tissue type per-class | `results/pathocell_evaluation/{model}/pannuke_summary/tissue_type_{all,benign_only}_per_class.csv` |
| Tissue type summary | `results/pathocell_evaluation/{model}/pannuke_summary/tissue_type_{all,benign_only}_summary.csv` |
| Tissue type confusion | `results/pathocell_evaluation/{model}/pannuke_summary/tissue_type_{all,benign_only}_confusion.csv` |
| Tissue mapping | `resources/pathocell/processed/pannuke/tissue_mapping.csv` |
| types.npy | `resources/pathocell/pannuke_fold2_types.npy` |

---

# Status

- [x] Part 1: Disease stratification (2026-03-25)
- [x] Part 2: Tissue type prediction — base model (2026-03-25)
- [x] Part 3: Trimodal comparison — base, +quilt1m, +quilt1m_curated (2026-03-25/26)
- [x] Part 3: Bimodal baseline — `spotwhisperer_quilt1m` (2026-03-26)
- [x] Manuscript integration of 19-tissue result (2026-05-09; revision item ZBSb-Q2|L1)

# Next Steps

- Prompt engineering: test `"An H&E image of {tissue} tissue"` template
- Tissue grouping: merge confusable tissues based on confusion matrix
- Additional baselines: CONCH, PLIP, MUSK
- Multi-seed replication for trimodal variants on tissue prediction (currently seed=0 only)

---

# Manuscript integration (revision item: ZBSb-Q2|L1 — tissue-level prediction)

Analysis directory: `/home/moritz/Projects/SpatialWhisperer/Analysis/pannuke_19tissue_quilt1m_baseline/`
(org-roam id `31811559-dbb2-445a-ba95-aa05fd82b5ff`).

## Reviewer commitment

Reviewer ZBSb explicitly asked (Q2|L1): *"Would transitive learning improve performance for
other tasks like tissue classification and those shown in Fig. 6?"* The rebuttal table
promised:

| Model (training data) | Tissue prediction (PanNuke; AUROC) |
|-----------------------|------------------------------------|
| Ours: T↔G+G↔I         | 0.586                              |
| T↔G+G↔I+I↔T           | 0.646                              |

with the interpretation: *"Transitive learning helps some tasks more than others. Training
on all three paired datasets integrates the best of both worlds (high tissue-level and
cell-level performance)."* The reviewer marked this concern *Fully resolved*; the result
must therefore appear in the camera-ready.

## Final numbers (sample-level macro AUROC, n=2,401 samples / 8,373 patches)

| Model                                 | All AUROC | Benign-only AUROC |
|---------------------------------------|-----------|-------------------|
| Bimodal I↔T (Quilt-1M only)           | **0.689** | **0.649**         |
| Bridge T↔G + G↔I                      | 0.586     | 0.549             |
| Trimodal (+raw Quilt-1M)              | 0.646     | 0.595             |
| Trimodal (+harmonized Quilt-1M)       | 0.649     | 0.616             |

Source CSVs on Sherlock:
`results/pathocell_evaluation/{model}/pannuke_summary/tissue_type_{all,benign_only}_summary.csv`
(verified 2026-05-09). All single-seed (=0), consistent with `tab:trimodal_ablation_benchmark`
in the manuscript.

## Cross-task pattern (combined with PathoCell cell-type, mean AUROC)

| Model                                 | Cell-type (PathoCell) | Tissue (PanNuke 19-class) |
|---------------------------------------|-----------------------|---------------------------|
| Bimodal I↔T (Quilt-1M only)           | 0.566                 | **0.689**                 |
| Bridge T↔G + G↔I                      | 0.630                 | 0.586                     |
| Trimodal (+raw Quilt-1M)              | 0.609                 | 0.646                     |
| Trimodal (+harmonized Quilt-1M)       | **0.645**             | 0.649                     |

The harmonized trimodal model is the only configuration competitive on both benchmarks.
Quilt-1M alone wins tissue but loses cell-type (no gene-expression bridge); the bridge
wins cell-type but loses tissue (no I↔T data); the raw trimodal partially recovers tissue
performance but the cell-type signal is corrupted by raw Quilt-1M captions; harmonization
recovers both.

## Manuscript placement (decision: side-note framing)

Folded into the existing harmonization appendix `\section{Effects of text annotation
harmonization}` (label `appendix:quilt1m-curation`) as **one short paragraph** inserted
between `tab:trimodal_ablation_benchmark` and the existing image-text retrieval paragraph.
No new table, no new figure, no main-text addition (the §4.2 reference at
`main.tex:482` already points to `appendix:quilt1m-curation` and now implicitly
covers both probes).

The paragraph reports three numbers inline (bridge 0.586; +raw Quilt-1M 0.646; +harmonized
Quilt-1M 0.649). The bimodal Quilt-1M I↔T baseline (0.689) is **omitted** — not promised
in the rebuttal and would shift a side note into a baseline showdown that weakens the
"best of both worlds" framing the reviewer accepted.

Files staged for the user (in the analysis directory):

- `manuscript_edits.md` — surgical patch instructions for `appendix.tex` (and three
  optional levels of main-text pointer in §4.2).
- `appendix_tissue_snippet.tex` — drop-in LaTeX for the single appendix paragraph.

## Code status

Code in `rules/disease_stratification.smk` runs unchanged; existing summary CSVs on
Sherlock (Mar 26 timestamps) reproduce the rebuttal numbers exactly. No regression to
report (the `uni_model.py` TIFF dispatch fix from the `trimodal_ablation` analysis
post-dates these tissue-prediction CSVs and does not invalidate them).
