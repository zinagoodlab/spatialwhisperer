# Seed Variance Analysis

## Motivation
Reviewer requested variance estimates for Table 2 results (CRC PathoCellBench).
Quote: "Can the authors provide variance estimates across multiple seeds?"

## Design
- **Existing model**: `spotwhisperer_cellxgene_census__archs4_geo__hest1k.ckpt` (seed 0)
  - Already has CRC eval results in `results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k/summary/`
- **New models to train**: seeds 1 and 2, producing:
  - `spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed1.ckpt`
  - `spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed2.ckpt`
- **Evaluation**: CRC PathoCellBench only (109 datasets, patch-level)
- **Output**: `results/pathocell_evaluation/seed_variance/seed_variance_table.csv`
  - Per-cell-type AUROC mean +/- std across 3 training seeds

## Implementation
- `rules/seed_analysis.smk`: training rule (`train_spotwhisperer_seeded`) + aggregation rule (`seed_variance_table`)
- `seed_training_config.yaml`: copy of `base_config.yaml` with two changes:
  - `callbacks: [ModelCheckpoint(save_last=true)]` (required since the hardcoded callback was removed from `training.py` in commit `1e24aea5`)
  - `use_cache: true` (so the second sequential training run reuses cached data from the first)
- `scripts/compute_seed_variance_table.py`: reads per-class CSVs from 3 seeds, computes mean/std
- Seeded models use `{model}_seed{N}` naming which flows through existing eval pipeline unchanged
- `ruleorder: train_spotwhisperer_seeded > train_spotwhisperer` to resolve wildcard ambiguity

## How to run
```bash
# From src/spotwhisperer_eval/ on Sherlock
snakemake --profile sm7_slurm --jobs 1 seed_analysis_all
```

## Progress Log

### 2026-03-24: First attempt
- Submitted controller job 19472299 on Sherlock
- Training jobs 19472334 (seed 1) and 19472338 (seed 2) started on sh04-02n05 (H100)
- **Failed after ~4h** with `FileNotFoundError`: the preprocessed `.pt` cache for `cellxgene_census` (geneformer base variant) had been purged from `/scratch/`
- W&B logged partial run: seed 1 = `wuu8b2lb`

### 2026-03-25: Cache regeneration
- The `.pt` files are the training samples read by the dataloader; they're generated from processed h5ad files during a preprocessing step
- The check for missing individual `.pt` files in `jointemb.py:475` was gated behind `if False` for performance reasons
- Temporarily changed to `if True` to enable regeneration
- Submitted adhoc regeneration job 19499106: `cellwhisperer fit` with `--trainer.limit_train_batches 1 --trainer.max_epochs 1` and only `cellxgene_census` dataset
- **Completed in 57 min**, restored 370k `.pt` files to `/scratch/groups/zinaida/moritzs/cellwhisperer/data_loading/cellxgene_census_/geneformer_dmis-lab__biobert-v1.1_uni2_100_False___/`
- Reverted `jointemb.py:475` back to `if False`

### 2026-03-25: Second attempt
- Submitted controller job **19503226** (`seed_var_ctrl2`) on sh02-09n49 (CPU, 7-day limit)
- Training jobs on sh04-02n05 (H100):
  - **19503307** (seed 1): W&B https://wandb.ai/single-cellm/SpatialWhisperer/runs/40g6plrd
  - **19503308** (seed 2): W&B https://wandb.ai/single-cellm/SpatialWhisperer/runs/jax2w3mn
- Training completed all 4 epochs (~28h), loss converged well (train ~1.05, val ~3.84)
- **Failed**: `MissingOutputException` — no `.ckpt` files produced

### Root cause analysis
- The `after_fit` method in `training.py` copies from `trainer.checkpoint_callback.last_model_path` to the target path
- `checkpoint_callback.last_model_path` was empty because no `ModelCheckpoint` callback was instantiated
- In `base_config.yaml`, `callbacks: null` means no callbacks are created
- The original model (Dec 25) was trained with an older code version (pre-commit `1e24aea5`, Feb 11) that had a **hardcoded** `ModelCheckpoint(save_last=True)` in `cli_main()`. That hardcoded callback was removed in `1e24aea5` and the comment says configs should specify callbacks explicitly — but `base_config.yaml` was never updated.
- The `train_spotwhisperer` rule in `training.smk` has the same latent bug: any new training with the current code + `base_config.yaml` would fail to save checkpoints.

### 2026-03-26: Third attempt (current)
- Created `seed_training_config.yaml`: copy of `base_config.yaml` with:
  - `callbacks: [ModelCheckpoint(save_last=true)]` — fixes the missing checkpoint issue
  - `use_cache: true` — enables caching so the second sequential run reuses preprocessed data
- Updated `seed_analysis.smk` to use `seed_training_config.yaml` instead of `base_config.yaml`
- Submitted with `--jobs 1` so seed 1 trains first, then seed 2 (sequential, reusing cache)
- Controller job **19651962** (`seed_var_ctrl3`)
- Controller logs: `/scratch/users/moritzs/seed_variance_controller3_19651962.{out,err}`

### 2026-03-28: CRC variance table complete; follow-up runs submitted

**CRC variance table done** — see Results section below.

**Seed 0 retrain** (to check reproducibility vs. original model):
- Original model was trained Dec 25 with old code (hardcoded ModelCheckpoint, no caching)
- Retraining seed 0 with `seed_training_config.yaml` to get apples-to-apples comparison
- Controller: **19807502** (`seed0_retrain`)
- Training job: **19807584** on H100
- Target: `spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed0.ckpt` + CRC eval

**Lizard/PanNuke evals for seed1+seed2** (in parallel):
- Models already exist from CRC runs; only eval pipeline runs
- Controller: **19807576** (`liz_pan_eval`)
- 246 jobs: 140 Lizard + 102 PanNuke predictions + 4 metrics aggregations
- Targets:
  - `{seed1,seed2}/lizard_summary/patch_per_class_metrics_from_scores.csv`
  - `{seed1,seed2}/pannuke_summary/patch_per_class_metrics_from_scores.csv`

## Upcoming: quilt1m_curated seed analysis

The `spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_curated` model (Jan 15) was also
trained with seed 0 (same `train_spotwhisperer` rule, `SEEDS[0]=0`). After current runs complete:
- Train seed1 and seed2 for `cellxgene_census__archs4_geo__hest1k__quilt1m_curated`
- Eval all 3 seeds on CRC, Lizard, PanNuke
- Aggregate variance table

### 2026-03-28: All evals complete

- Seed 0 retrained (use_cache=true): CRC AUROC = 0.629, nearly identical to original 0.630 → **use_cache has no effect**
- Lizard and PanNuke evals for seed1+seed2 complete (70/70 and 51/51 each)
- Seed 0 CRC eval for retrained model complete (109/109)

### 2026-03-29: Methodology investigation and baseline comparison

**Key discovery**: Table 2 uses **presence-based** binary labels for AUROC (`true_probs > 0`, not dominant-class argmax).
This was the source of discrepancy between our scripts and the reported numbers.
Fixed in `compute_reduced_class_table2_style.py` — now reproduces Table 2 exactly (seed0 CRC = 0.630).

**Baseline CSVs**: extracted PLIP/CONCH/MUSK logits from Animesh's `VLM_benchmarks.zip` into
`src/spotwhisperer_eval/static/baselines_animesh/`. Updated `pathocell_benchmark.smk` to reference
them via `BASELINES_DIR`.

## Status
- [x] Implementation complete (rules + script)
- [x] Cache regeneration (370k .pt files restored)
- [x] Root cause identified (missing ModelCheckpoint callback)
- [x] Fix: created `seed_training_config.yaml` with ModelCheckpoint + caching
- [x] Training seed 1 (Mar 28 04:39, 1.5GB checkpoint)
- [x] Training seed 2 (Mar 27, 1.5GB checkpoint)
- [x] CRC eval for seed 1 (109/109)
- [x] CRC eval for seed 2 (109/109)
- [x] Seed 0 retrained with new config → CRC 0.629 ≈ original 0.630 (use_cache has no effect)
- [x] CRC eval for retrained seed 0 (109/109)
- [x] Lizard eval for seed 1 (70/70)
- [x] Lizard eval for seed 2 (70/70)
- [x] PanNuke eval for seed 1 (51/51)
- [x] PanNuke eval for seed 2 (51/51)
- [x] Baseline comparison (PLIP/CONCH/MUSK) with Table 2 methodology
- [ ] quilt1m_curated: train seed1+seed2, eval CRC/Lizard/PanNuke
- [ ] Retrain seed=1 with use_cache=false (ablation; likely unnecessary given seed0 result)

## Results

### Main result: Table 2 methodology (presence-based AUROC, classes→datasets→mean, reduced classes)

`terms2` (original):

| Benchmark | PLIP | CONCH | MUSK | Ours seed0 | Ours seed1 | Ours seed2 | Ours mean ± std |
|---|---|---|---|---|---|---|---|
| **CRC** (13-class) | 0.468 | 0.507 | 0.581 | **0.630** | 0.614 | 0.620 | **0.621 ± 0.008** |
| **Lizard** (3-class) | 0.528 | 0.581 | 0.563 | **0.764** | 0.732 | 0.752 | **0.749 ± 0.016** |
| **PanNuke** (4-class) | 0.588 | 0.624 | 0.530 | **0.689** | 0.687 | 0.686 | **0.687 ± 0.001** |

`terms1` (added 2026-05-09 — matches the prompts used in the manuscript's `tab:pathocell_benchmark`):

| Benchmark | PLIP terms1 | CONCH terms1 | MUSK terms1 | Ours seed0 | Ours mean ± std |
|---|---|---|---|---|---|
| **CRC** (13-class) | 0.488 | 0.546 | 0.563 | **0.630** | **0.621 ± 0.008** |
| **Lizard** (3-class) | 0.528 | 0.615 | 0.556 | **0.764** | **0.749 ± 0.016** |
| **PanNuke** (4-class) | 0.584 | 0.604 | 0.512 | **0.689** | **0.687 ± 0.001** |

(Sanity check: `terms1` PathoCell CONCH=0.546, PLIP=0.488 reproduces the manuscript table exactly.)

- **Seed 0** = original model (trained Dec 25, `use_cache=false`, W&B `8vwy3s34`)
- **Seeds 1, 2** = retrained with `seed_training_config.yaml` (`use_cache=true`)
- Retrained seed 0 (use_cache=true) gives CRC 0.629 ≈ original 0.630, confirming no config effect
- Mean ± std uses sample std (ddof=1) over {seed0_orig, seed1, seed2}; supersedes the earlier `± 0.006 / 0.010 / 0.001` triplet (population std + a stale Lizard mean)
- Margins over baselines (0.04–0.18) are much larger than seed variance (0.001–0.016)

### Manuscript follow-up (2026-05-09)

Added `terms1` baseline scoring + merged 7-method × 20-class appendix table consolidation in
`Projects/SpatialWhisperer/Analysis/lizard_pannuke_baselines_terms1_multiseed/`.
Recomputed abstract relative-gain claims (10.3% / 21.9% / 13.9% vs strongest CONCH/PLIP/MUSK terms1
baseline per benchmark; replaces 9.1% / 15.4% / 19.5% from the camera-ready). See
`manuscript_edits.md` in that analysis directory for the surgical edits.

### Aggregation methodology notes
- **CRC**: 13 cell types (excl Background + Other cells), presence-based AUROC, per-class averaged across 109 datasets, then averaged across classes
- **Lizard**: 3-class reduced (Neutrophil+Lymphocyte+Eosinophil → Leukocyte, drop Plasma), same AUROC scheme, 70 datasets
- **PanNuke**: 4-class reduced (drop Dead Cells), same AUROC scheme, 51 datasets
- Baseline prompt set selectable via `BASELINE_TERMS=terms1|terms2` (default `terms2`)

### Scripts
- `scripts/compute_reduced_class_table2_style.py` — our model (all seeds), exact Table 2 methodology
- `scripts/compute_baselines_table2_style.py` — PLIP/CONCH/MUSK baselines
- `scripts/compute_seed_variance_table.py` — snakemake-integrated per-class variance (CRC only)
- `scripts/compute_reduced_class_seed_variance.py` — global-pooling variant (different numbers)
- `scripts/compute_reduced_class_per_dataset_averaged.py` — per-dataset macro-averaged variant

### Output files
- `results/pathocell_evaluation/seed_variance/reduced_class_table2_style.csv` — our model results
- `results/pathocell_evaluation/seed_variance/reduced_class_table2_style_per_class.csv` — per-class breakdown
- `results/pathocell_evaluation/seed_variance/baselines_table2_style.csv` — baseline results
- `results/pathocell_evaluation/seed_variance/seed_variance_table.csv` — snakemake-generated CRC variance
