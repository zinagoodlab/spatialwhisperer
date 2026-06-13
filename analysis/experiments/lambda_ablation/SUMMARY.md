# Lambda Ablation Study

## Motivation
Reviewer asked: "How sensitive is the overall model performance to the choice of the weight
parameters (λ₁, λ₂, λ₃) for the different terms in the composite loss function? Could you
provide a brief ablation study varying these weights?"

## Design
We approximate different λ ratios by varying the **dataset composition** (duplicating
datasets to increase their share in each batch), while fixing the total number of
gradient steps via `max_steps: 4000` (≈1 epoch of the baseline combo).

Since we train on `cellxgene_census, archs4_geo, hest1k` (3-dataset combo, no quilt1m),
only two modality-pair losses are active:
- λ_{I↔G} (image ↔ transcriptome, served by hest1k, ~921K pairs)
- λ_{T↔G} (text ↔ transcriptome, served by cellxgene_census + archs4_geo, ~1.08M pairs)

### Configurations

All runs: `max_steps: 4000`, `max_epochs: -1`, LR 1e-5, batch_size 512, seed 0.

| Config | dataset_names | Ratio (T↔G : I↔G) | Effective λ |
|--------|---------------|-------------------|-------------|
| `baseline_1ep` | archs4_geo,cellxgene_census,hest1k | 54:46 (≈1:1) | λ_{T↔G}≈λ_{I↔G} |
| `image_heavy` | archs4_geo,cellxgene_census,hest1k,hest1k | 37:63 (≈1:2) | λ_{I↔G}≈2·λ_{T↔G} |
| `text_heavy` | archs4_geo,archs4_geo,cellxgene_census,cellxgene_census,hest1k | 70:30 (≈2:1) | λ_{T↔G}≈2·λ_{I↔G} |

### Why max_steps instead of max_epochs
Duplicating datasets changes the epoch length, so a "1-epoch" run would mean different
numbers of gradient updates. `max_steps: 4000` ensures all three runs see exactly
the same number of optimizer steps — the only variable is batch composition.

## Implementation
- `rules/lambda_ablation.smk`: training rule + `lambda_ablation_all` target
- `lambda_training_config.yaml`: copy of `base_config.yaml` with two fixes:
  - `callbacks: [ModelCheckpoint(save_last=true)]` (required since the hardcoded callback was removed in commit `1e24aea5`)
  - `use_cache: true` (reuse preprocessed `.pt` cache across runs)
- `experiments/lambda_ablation/delta_config/*.yaml`: per-config overrides (max_steps, dataset_names)
- Snakefile: `include: "rules/lambda_ablation.smk"` (line 29)

### Output paths
- Models: `results/models/jointemb/spatialwhisperer_lambda_{baseline_1ep,image_heavy,text_heavy}.ckpt`
- Eval: `results/pathocell_evaluation/spatialwhisperer_lambda_{config}/summary/patch_per_class_metrics_from_scores.csv`
- WandB: project `SpatialWhisperer`, entity `single-cellm`, run names `lambda_ablation_{baseline_1ep,image_heavy,text_heavy}`

## How to run
```bash
snakemake --snakefile src/spatialwhisperer_eval/Snakefile --profile sm7_slurm lambda_ablation_all
```

## Progress Log

### 2026-03-26: First attempt
- Submitted 3 training jobs (19639847/19639851/19639858) via tmux `lambda_submit`
- **Issue identified**: `base_config.yaml` has `callbacks: null`, meaning no `ModelCheckpoint`
  is created. The `--last_model_path` flag relies on `trainer.checkpoint_callback.last_model_path`
  which would be empty → training completes but no `.ckpt` is saved (same bug as seed_variance).
- Also `use_cache: false` in base_config means no benefit from existing preprocessed data cache.
- **Cancelled** jobs 19639847/19639851/19639858 before they wasted GPU hours.

### 2026-03-26: Second attempt
- Created `lambda_training_config.yaml` (copy of `base_config.yaml` with ModelCheckpoint + use_cache fixes)
- Updated `lambda_ablation.smk` to reference `lambda_training_config.yaml` instead of `base_config.yaml`
- Resubmitted via tmux `lambda_submit`
- Training jobs: **19652844** (baseline_1ep), **19652845** (image_heavy), **19652848** (text_heavy)
- Controller log: `~/lambda_ablation_submit2.log`

### 2026-03-27: Training completed, first eval attempt failed
- All three trainings completed successfully and produced checkpoints:
  - `spatialwhisperer_lambda_baseline_1ep.ckpt`
  - `spatialwhisperer_lambda_image_heavy.ckpt`
  - `spatialwhisperer_lambda_text_heavy.ckpt`
- Partial CRC eval progress before failure:
  - `text_heavy`: 12 score files
  - `image_heavy`: 2 score files
  - `baseline_1ep`: 0 score files
- Root cause: multiple `pathocell_cell_type_prediction` jobs were submitted with **50 GB** RAM and were OOM-killed (`State=OUT_OF_MEMORY`, exit `0:125`, `MaxRSS` ~51 GB)
- Snakemake controller exited after these failed jobs; tmux session `lambda_submit` remained open but was no longer running the workflow

### 2026-03-27: Eval restart
- Verified via Sherlock dry-run that the current `pathocell_cell_type_prediction` rule now requests **150 GB** RAM (`mem_mb=150000`)
- Unlocked the working directory and restarted the lambda eval controller in tmux session `lambda_resume`
- New controller log: `~/lambda_ablation_resume.log`
- Restart uses a private cache dir on scratch via `XDG_CACHE_HOME=/scratch/users/moritzs/.cache` to avoid the Snakemake `/tmp/snakemake` permission issue
- Dry-run/launch shows only the remaining missing outputs are targeted: **313** `pathocell_cell_type_prediction` jobs + **3** `pathocell_metrics_from_scores` jobs

### 2026-03-28: Switched from tmux controller to SLURM conductor job
- The tmux-based controller failed again after progressing to ~53% because the Snakemake SLURM helper crashed while polling job status (`BlockingIOError: [Errno 11] Resource temporarily unavailable`)
- Added `src/spatialwhisperer_eval/experiments/lambda_ablation/sherlock_controller.sh`, a long-running `sbatch` conductor job that:
  - activates the `cellwhisperer` conda environment
  - uses `XDG_CACHE_HOME=/scratch/users/moritzs/.cache`
  - runs `snakemake --unlock`
  - resumes `lambda_ablation_all` under `sm7_slurm`
- First conductor submission (**19833919**) failed immediately because `set -u` conflicted with a conda activation hook on Sherlock (`ADDR2LINE: unbound variable`)
- Fixed the script to use `set -eo pipefail` and resubmitted as **19833948**
- Current progress at handoff: `baseline_1ep=60`, `image_heavy=63`, `text_heavy=60` completed score files

## Status (2026-03-28)
- [x] Experiment designed and implemented
- [x] Fixed checkpoint saving (lambda_training_config.yaml with ModelCheckpoint + use_cache)
- [x] All three training runs completed and checkpoints saved
- [x] First eval failure diagnosed: 50 GB `pathocell` jobs OOM-killed
- [x] Eval controller restarted with current 150 GB rule in tmux session `lambda_resume`
- [x] Replaced fragile tmux controller with Sherlock SLURM conductor script
- [x] CRC PathoCellBench eval completed under conductor job `19833948`
- [x] Metrics aggregation completed

## Results

Reviewer-facing macro AUROC should use the **classwise dataset-averaged** convention:

1. compute per-dataset per-class AUROC
2. average AUROC across datasets within each class
3. average across classes

This is reported below as `macro_auroc_classwise_dataset_avg` and matches the mean of
`patch_per_class_metrics_from_scores.csv` (equivalently: group
`patch_per_class_by_dataset_metrics_from_scores.csv` by `class_label`, average, then
average across classes). It is **not** the pooled/global AUROC, and it is also distinct
from averaging per-dataset macro AUROCs (`macro_auroc_datasetwise_class_avg`).

Saved table: `src/spatialwhisperer_eval/experiments/lambda_ablation/results_crc_pathocell.csv`

| config | macro_auroc_classwise_dataset_avg | macro_soft_auroc_classwise_dataset_avg | macro_auroc_datasetwise_class_avg | macro_f1_datasetwise_class_avg | macro_precision_datasetwise_class_avg | macro_recall_at_5_datasetwise_class_avg |
|--------|-----------------------------------|----------------------------------------|-----------------------------------|--------------------------------|----------------------------------------|------------------------------------------|
| baseline_1ep | 0.623577 | 0.482361 | 0.638873 | 0.079515 | 0.151138 | 0.551594 |
| image_heavy | 0.620423 | 0.490869 | 0.630006 | 0.104734 | 0.180076 | 0.571620 |
| text_heavy | 0.605036 | 0.484784 | 0.613052 | 0.104798 | 0.177800 | 0.567051 |

Interpretation:
- The main macro AUROC metric is fairly stable across weighting choices, with the best setting (`baseline_1ep`) at 0.6236 and the worst (`text_heavy`) at 0.6050, a spread of ~0.0185 absolute.
- Overweighting image-transcriptome pairs (`image_heavy`) slightly reduces macro AUROC versus baseline but improves soft AUROC, F1, precision, recall@5, and calibration-style losses (cross-entropy / JS divergence).
- Overweighting text-transcriptome pairs (`text_heavy`) is the weakest setting on macro AUROC in this CRC benchmark.

## Caveat / follow-up TODO

- The current analysis is flawed because only the balanced case effectively sees the full baseline dataset, while the duplicated-dataset settings are constrained to the same `max_steps=4000` budget.
- Follow-up TODO: rerun the lambda ablation with **one full epoch per configuration** (instead of a fixed 4000 steps) so each weighting condition sees its own full training data distribution once.

## Full-epoch rerun

To address the caveat above, a second round was set up with distinct config names so the
original fixed-step outputs remain intact:

- `baseline_full_epoch`
- `image_heavy_full_epoch`
- `text_heavy_full_epoch`

These use one full epoch per configuration (`max_epochs: 1`) rather than `max_steps: 4000`.

### Added files
- `src/spatialwhisperer_eval/experiments/lambda_ablation/delta_config/baseline_full_epoch.yaml`
- `src/spatialwhisperer_eval/experiments/lambda_ablation/delta_config/image_heavy_full_epoch.yaml`
- `src/spatialwhisperer_eval/experiments/lambda_ablation/delta_config/text_heavy_full_epoch.yaml`
- `src/spatialwhisperer_eval/experiments/lambda_ablation/sherlock_controller_full_epoch.sh`

### Snakemake target
- `lambda_ablation_full_epoch_all`

This target retrains the three full-epoch models and runs the full CRC PathoCellBench
evaluation again, writing to new model/result paths such as:

- `results/models/jointemb/spatialwhisperer_lambda_baseline_full_epoch.ckpt`
- `results/pathocell_evaluation/spatialwhisperer_lambda_baseline_full_epoch/...`

### Full-epoch progress log

- First conductor (`19891296`) completed training successfully but eval jobs failed immediately
  due to a `TypeError` in `pathocell_benchmark.smk:928` (`BASELINES_DIR = srcdir(...)` returns
  `str`, not `Path`; `str / str` fails). Fixed by wrapping in `_Path()`.
- Second conductor (`19926260`) ran eval to completion for `image_heavy` and `text_heavy`,
  but the `baseline_full_epoch` summary job was stuck on the `normal` partition (missing
  `partition=cmackall` in the `pathocell_metrics_from_scores` rule resources). Fixed.
- Third conductor (`20022223`) completed the last summary job.

### Full-epoch results (2026-03-30)

Saved table: `src/spatialwhisperer_eval/experiments/lambda_ablation/results_crc_pathocell_full_epoch.csv`

| config | macro_auroc (class→dataset avg) | macro_soft_auroc | macro_auroc (dataset→class avg) | macro_f1 | macro_precision | macro_recall@5 | cross_entropy | JS_divergence |
|--------|--------------------------------|-----------------|-------------------------------|----------|----------------|---------------|--------------|--------------|
| baseline_full_epoch | 0.6285 | 0.4850 | 0.6440 | 0.0819 | 0.1544 | 0.5041 | 2.8585 | 0.3352 |
| image_heavy_full_epoch | **0.6391** | **0.4945** | **0.6547** | **0.1066** | **0.1835** | **0.5827** | **2.5990** | **0.3036** |
| text_heavy_full_epoch | 0.6346 | 0.4924 | 0.6484 | 0.0994 | 0.1834 | 0.5542 | 2.6671 | 0.3152 |

Interpretation (full-epoch, corrected analysis):
- With full-epoch training, **image_heavy is now the best configuration** across all metrics,
  including the reviewer-facing `macro_auroc_classwise_dataset_avg` (0.6391 vs 0.6285 baseline).
- The ranking flipped compared to the fixed-step analysis: the baseline is now the weakest on
  macro AUROC, while both weighted configurations outperform it.
- This makes sense: in the fixed-step analysis, only the baseline saw its full dataset; now that
  all configs see their full data once, the duplicated-dataset configs benefit from the extra
  emphasis on their respective modality pairs.
- The spread across all three configs remains small: **~0.011** absolute on macro AUROC,
  indicating that the model is **not highly sensitive to λ weighting**.

## Next steps

The current experiments still have a confound: the fixed-step runs use the same number of
gradient updates but don't see all data; the full-epoch runs see all data but have different
numbers of gradient updates. A proper design needs both:

1. All runs must see all data at least once (no modality pair is undersampled).
2. All runs must train for the same number of steps (no run gets more gradient updates).

**Option A (dataset duplication + max_steps):** Calculate the number of steps needed for
the *largest* dataset mix to complete one full pass (i.e. 1 epoch of the most-duplicated
config). Use that as `max_steps` for all three runs. The smaller configs will cycle through
their data more than once, but the step count and the "all data seen" constraints are both
satisfied.

**Option B (implement per-pair lambdas directly):** Add `lambda_transcriptome_text`,
`lambda_transcriptome_image`, `lambda_text_image` to ClipLoss config. Weight each modality
pair's loss contribution directly in `ClipLoss.forward()`. This is cleaner, avoids the
dataset-duplication hack entirely, and gives exact control over λ ratios with identical
data and step counts across all runs.

## Per-pair λ sweep (Option B, 2026-05-14 → 2026-05-16)

Chosen approach for the appendix.

> **Reproducing this experiment requires applying `perpair_lambdas.patch`
> before training.** The patch lives in this directory and modifies
> `src/cellwhisperer/jointemb/loss/{config.py,losses.py}` to expose
> per-pair λ coefficients. The change is intentionally not merged into the
> main code path to keep the production training loop unchanged; it is
> appendix-only. Apply with:
>
> ```bash
> git apply src/spatialwhisperer_eval/experiments/lambda_ablation/perpair_lambdas.patch
> ```
>
> Revert with `git checkout HEAD -- src/cellwhisperer/jointemb/loss/` after
> the sweep finishes.

Implementation:

- `perpair_lambdas.patch` adds three dataclass fields to `LossConfig`
  (`clip_lambda_transcriptome_text`, `clip_lambda_transcriptome_image`,
  `clip_lambda_text_image`, all default `1.0`) and rewrites
  `ClipLoss.forward()` to return `(sum_p λ_p L_p) / (sum_p λ_p)` over the
  *active* pairs (logits non-None and λ > 0). Dividing by the active-λ sum
  keeps gradient magnitude comparable to the previous equal-weight mean,
  so LR does not need retuning. With defaults (all λ = 1) the loss
  numerically matches the prior `total_loss / num_valid_pairs` exactly
  (verified by a local sanity check).
- 5 new delta configs in `delta_config/perpair_*.yaml`. All use
  `dataset_names: archs4_geo,cellxgene_census,hest1k` (same as main-text Trimodal),
  `max_epochs: 1`, identical step budget. Only the per-pair lambdas vary.
- Snakemake target `lambda_ablation_perpair_all` (uses the existing
  `train_spatialwhisperer_lambda` and `pathocell_cell_type_prediction` rules).
- Controller script `sherlock_controller_perpair.sh`. Job 25029903 ran 1d 15h 11m on
  Sherlock cmackall; completed cleanly (exit 0:0). No job failures during the sweep.

Because Quilt-1M is excluded, `L_{T<->I}` is dormant and the effective knob is the
single ratio λ_{T<->G} : λ_{I<->G}.

### Results (`results_crc_pathocell_perpair.csv`)

| config | λ_{T<->G} : λ_{I<->G} | macro_auroc (class→dataset) | macro_soft_auroc | macro_auroc (dataset→class) | macro_f1 | macro_precision | macro_recall@5 | cross_entropy | JS_divergence |
|--------|----------------------|-----------------------------|------------------|-----------------------------|----------|-----------------|----------------|---------------|---------------|
| perpair_text_heavy_4x   | 4:1 | 0.6240 | 0.4858 | 0.6351 | 0.1064 | 0.1830 | 0.5750 | 2.5868 | 0.3042 |
| perpair_text_heavy_2x   | 2:1 | **0.6241** | 0.4876 | **0.6356** | 0.1036 | 0.1750 | 0.5652 | 2.5967 | 0.3068 |
| perpair_balanced        | 1:1 | 0.6234 | **0.4889** | 0.6354 | 0.1034 | 0.1775 | 0.5621 | 2.5964 | 0.3069 |
| perpair_image_heavy_2x  | 1:2 | 0.6222 | 0.4891 | 0.6338 | 0.1022 | 0.1786 | 0.5650 | 2.5940 | 0.3067 |
| perpair_image_heavy_4x  | 1:4 | 0.6199 | 0.4889 | 0.6315 | 0.1053 | 0.1753 | 0.5671 | 2.5877 | 0.3056 |

Interpretation (clean isolation of loss weighting from data/step confounds):

- **Spread on macro AUROC across the full 16× ratio range (1:4 → 4:1) is ~0.0042
  absolute** (0.6199 → 0.6241). All five configurations sit within a band narrower
  than typical seed variance, demonstrating that the model is robust to the choice
  of composite-loss weights.
- Mild monotone trend: pulling weight away from the T<->G pair (image-heavy 1:4) is
  the weakest on macro AUROC, while text-heavy 2:1 is marginally the best.
- The `perpair_balanced` (1:1) point is essentially tied with text_heavy_2x and
  validates the main-paper choice of equal weights as a defensible default.
- Comparison to the earlier confounded designs: `perpair_balanced` (0.6234) ≈
  `baseline_1ep` (0.6236) but lower than `baseline_full_epoch` (0.6285). The
  earlier full-epoch *image_heavy* (0.6391) drew its win from seeing 2× as many
  images, not from the loss weighting itself — exactly the confound this sweep
  was designed to remove.

Conclusion for the appendix: the composite loss is insensitive to the choice of
λ over a 16× range when data and step count are held constant; the equal-weight
setting used in the main paper is justified.
