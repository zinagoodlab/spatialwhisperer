# Annotation Noise Robustness Experiment

**Reviewer concern:** "The method relies on gene expression as a bridge to cell-type text, but such annotations are often noisy and reference-dependent in practice. How robust is the proposed framework to this kind of annotation uncertainty?"

**Goal:** Measure how robust the trimodal model is to noisy cell type annotations in the G↔T training data by randomly swapping text annotations at controlled rates and evaluating on PathoCellBench (CRC, 13 cell types, 109 samples).

**Status:** Not yet started.

---

## Design

We corrupt the `cellxgene_census` G↔T dataset by randomly reassigning the `natural_language_annotation` field across data points at rates of {1%, 2%, 5%}. Each corrupted sample receives the annotation from another randomly selected sample. This simulates misannotation / reference-dependent labeling errors that propagate through the LLM curation step.

We train 4 models (0%, 1%, 2%, 5% noise) on `cellxgene_census + hest1k` for 1 epoch and evaluate all on PathoCellBench (CRC patch-level, AUROC). The 0% model serves as the baseline.

**Why 1 epoch:** The full 4-epoch training is expensive (~70h GPU). 1-epoch models are sufficient to demonstrate the *relative* degradation trend, which is the signal we need for the reviewer response. The absolute numbers will be lower than the paper's main results, but the comparison is fair since all noise levels use the same epoch budget.

**Why cellxgene_census only (no archs4_geo):** Training with archs4_geo requires ~250GB memory (vs ~150GB without). Dropping it halves memory and significantly reduces training time while keeping the core G↔T dataset that provides the majority of training pairs.

---

## Implementation Plan

### Step 1: Create noise-corrupted cellxgene_census datasets

- [ ] Write `scripts/corrupt_annotations.py` -- a Snakemake script that:
  1. Loads `results/cellxgene_census/full_data.h5ad`
  2. Randomly selects `noise_fraction` of observations
  3. For selected observations, replaces `natural_language_annotation` with the annotation from a randomly chosen *different* observation (permutation-based swap)
  4. Saves to `results/cellxgene_census_{noise_pct}pct_noise/full_data.h5ad`
  5. Uses seed=0 for reproducibility

- [ ] Add Snakemake rule `corrupt_annotations` in `rules/annotation_noise.smk`:
  - Input: `results/cellxgene_census/full_data.h5ad`
  - Output: `results/cellxgene_census_{noise_pct}pct_noise/full_data.h5ad`
  - Wildcard: `noise_pct` in {1, 2, 5}
  - Params: `noise_fraction = int(wildcards.noise_pct) / 100`, `seed = 0`

### Step 2: Train 4 models (1 epoch each)

- [ ] Add Snakemake rule `train_annotation_noise_model` in `rules/annotation_noise.smk`:
  - Reuses the existing `cellwhisperer fit` CLI
  - Overrides `--trainer.max_epochs 1`
  - Dataset combos:
    - `cellxgene_census__hest1k` (0% noise baseline)
    - `cellxgene_census_1pct_noise__hest1k` (1% noise)
    - `cellxgene_census_2pct_noise__hest1k` (2% noise)
    - `cellxgene_census_5pct_noise__hest1k` (5% noise)
  - Output: `results/models/jointemb/spotwhisperer_annotation_noise_{noise_pct}pct.ckpt`
  - Resources: same as `train_spotwhisperer` but with `--trainer.max_epochs 1`
  - The 0% baseline model name: `spotwhisperer_annotation_noise_0pct.ckpt`

### Step 3: Evaluate all 4 models on PathoCellBench (CRC)

- [ ] Reuse the existing `pathocell_cell_type_prediction` and `pathocell_metrics_from_scores` rules
  - Model wildcards: `spotwhisperer_annotation_noise_{0,1,2,5}pct`
  - Prediction level: `patch`
  - Produces per-class AUROC across 109 CRC samples

### Step 4: Plot degradation curve

- [ ] Add `scripts/plot_annotation_noise_robustness.py`:
  - X-axis: noise fraction (0%, 1%, 2%, 5%)
  - Y-axis: mean AUROC (macro-averaged across 13 cell types)
  - Also show per-cell-type lines or a shaded band
  - Output: `results/pathocell_evaluation/comparison/patch/plots/annotation_noise_robustness.svg` + `.csv`

### Step 5: Summary target rule

- [ ] Add `rule annotation_noise_all` target in `rules/annotation_noise.smk` that requests the plot and all metrics CSVs

---

## Files to create/modify

| File | Action |
|------|--------|
| `rules/annotation_noise.smk` | **Create** -- all rules for this experiment |
| `scripts/corrupt_annotations.py` | **Create** -- annotation corruption script |
| `scripts/plot_annotation_noise_robustness.py` | **Create** -- degradation curve plot |
| `Snakefile` | **Modify** -- add `include: "rules/annotation_noise.smk"` |

---

## Expected outcome

- If AUROC degrades gracefully (e.g. <2% drop at 5% noise), this demonstrates strong robustness to annotation uncertainty.
- If AUROC degrades sharply, we can still characterize the sensitivity and discuss it honestly -- which itself is a contribution (understanding failure modes, as the theory in Section 3.4 predicts).

---

## Next steps (after this experiment)

- [ ] Repeat with `archs4_geo` included in training (`cellxgene_census_{noise}__archs4_geo__hest1k`) to test whether additional G↔T data buffers against noise
- [ ] Test coarser noise variants: replace fine-grained cell type labels with their parent ontology terms (e.g. "CD8+ cytotoxic T cell" → "T cell" → "lymphocyte") to simulate reference-dependent granularity differences
- [ ] Test annotation replicate variance: train with different LLM-generated text replicates (already stored in `obsm["natural_language_annotation_replicates"]`) to measure sensitivity to text generation stochasticity
