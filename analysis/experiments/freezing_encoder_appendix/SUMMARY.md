# Freezing / Encoder-choice Appendix

## Motivation

Reviewer commitment in `SpatialWhisperer_ICML/rebuttal/final.org` lines 1051-1053:
> We will provide an expanded version of this analysis (multiple seeds, 2 more
> lambdas) as well as additional ablations on encoder modules and freezing
> configurations in the revised manuscript.

The lambda + multi-seed halves are tracked separately; this experiment covers
the *encoder modules* and *freezing configurations* halves only.

Reviewer signals motivating each axis:
- Multiple reviewers note that "heavy backbones are frozen and only projection
  heads and the text encoder are trained for 4 epochs" (`final.org` lines
  149-154, 232-237). The freezing choice is currently asserted, not justified.
- One reviewer flags "encoder choices" as an honest limitation (`final.org`
  line 254).

## Design

Two axes, single seed (seed=0), one full epoch each. The point at the
intersection of both axes (`baseline` = geneformer + LUL) is the main paper
configuration and serves as the shared anchor.

### Freezing axis (encoders fixed at geneformer / bert / uni2)

`locking_mode` is a 3-character string `(transcriptome, text, image)` with
values `L` (locked), `U` (unfrozen, pretrained), `u` (unfrozen, randomly
initialized). See `src/cellwhisperer/jointemb/config.py:35` and
`src/cellwhisperer/jointemb/model.py:189-322`.

| config     | locking_mode | what's trained                                  |
|------------|--------------|-------------------------------------------------|
| `baseline` | LUL          | text tower + projection heads (main paper)      |
| `lll`      | LLL          | projection heads only                           |
| `llu`      | LLU          | image tower + projection heads                  |
| `ull`      | ULL          | transcriptome (bridge) tower + projection heads |

### Encoder axis (freezing fixed at LUL)

| config     | transcriptome model | other towers      |
|------------|---------------------|-------------------|
| `baseline` | geneformer          | bert / uni2 (LUL) |
| `uce`      | uce                 | bert / uni2 (LUL) |

Image and text encoders are intentionally held fixed: the bridge modality is
where the rebuttal text lives and where transitive alignment depends on
encoder geometry. Swapping `uni2` or `bert` would muddy that argument.

### Why one epoch

Matches the lambda full-epoch rerun: cleanly attributable, comparable to the
existing 1-epoch lambda configs, and avoids the step/epoch confound that
affected the lambda fixed-step round.

### Why Quilt-1M is dropped from training

The main-paper "Trimodal" model is the **2-pair** variant (T↔G + G↔I) — the
3-pair variant with Quilt-1M raw scored 0.609 PathoCell and is excluded from
the main-text tables (per the project memory). The training_config therefore
sets `dataset_names: archs4_geo,hest1k,cellxgene_census`. Quilt-1M caches in
the old GROUP_SCRATCH are in a pre-`use_disk_loading` in-memory format
incompatible with the current loader anyway — dropping is the correct call
on both grounds.

## Implementation

- `src/spotwhisperer_eval/freezing_encoder_training_config.yaml`
  — 1-epoch base config, no Quilt-1M, `ModelCheckpoint(save_last=true)`,
  `use_cache: true`.
- `src/spotwhisperer_eval/experiments/freezing_encoder_appendix/delta_config/`
  — `baseline.yaml`, `lll.yaml`, `llu.yaml`, `ull.yaml`, `uce.yaml`,
  `uce_smoke.yaml` (smoke uses `immgen,hest1k_8thsub` and `max_steps: 20`).
- `src/spotwhisperer_eval/rules/freezing_encoder_appendix.smk`
  — `train_spotwhisperer_fe` rule with `mem_mb=900000`,
  `slurm_gres("large", num_cpus=60, time="12:00:00", num_gpus=1)`.
  Targets: `freezing_encoder_appendix_smoke`,
  `freezing_encoder_appendix_all`.
- `src/spotwhisperer_eval/Snakefile`: `include: "rules/freezing_encoder_appendix.smk"`.
- `experiments/freezing_encoder_appendix/sherlock_controller.sh {smoke|full|baseline}`:
  SBATCH wrapper. The `baseline` option targets just the baseline ckpt for
  the controlled single-config production run.
- `aggregate_results.py`: per-config CSV → `results_crc_pathocell.csv`
  (writes the 8-column schema mirroring `lambda_ablation`).
- `appendix_draft.tex`: appendix subsection skeleton with `NN.NN`
  placeholders for the results table + forward-reference text for
  `main.tex` lines 274/289.

### Code changes outside this directory

- `static/UCE/gene_names_full.txt` (new, 19,656 human gene symbols from
  `all_species_gene_dict.json`).
- `static/UCE/gene_names.txt` (untouched, 5,782 CosMx-subset; legacy).
- `config.yaml` `uce_paths`:
  - `gene_names_path: static/UCE/gene_names_full.txt`
  - `checkpoint: "KuanP/uce-cxg-2025-baseline-8l-512d"` (was
    `cosmx_cxg_lbcl_gene_subset_assay_token`)
  - `use_assay_token: false`
- `src/cellwhisperer/jointemb/uce_model.py` `_align_genes`: added
  Ensembl→symbol fallback (~25 % of hest1k samples carry only Ensembl IDs
  in `var.gene_name`). Re-uses `resources/ensembl_gene_symbol_map.csv`
  (the file the Geneformer processor already builds).

### Why this UCE model

KuanP has 5 UCE checkpoints on HF; all share the same architecture
(`vocab_size=145469`, 512-dim, 8 layers) and differ only in training data:

- `uce-cosmx-geneset`, `cosmx_cxg_lbcl_gene_subset_assay_token` — CosMx
  subset, narrow vocab use-case.
- `uce-brain-pilot-8l-512d`, `uce-brain-midtrain-curated-hq-10000` —
  brain-specific.
- **`uce-cxg-2025-baseline-8l-512d`** — CELLxGENE 2025 baseline, broad
  scRNA atlas. Best fit for transitive learning across hest1k +
  archs4_geo + cellxgene_census.

The model card READMEs are unfortunately empty templates; selection was
made by name + naming convention.

## Cache + storage state

Switched `~/cellwhisperer_private/scratch` from
`/scratch/groups/zinaida/moritzs/cellwhisperer/` (GROUP_SCRATCH, hit 20M
inode limit during UCE prep) to `/scratch/users/moritzs/cellwhisperer/`
($SCRATCH, 20M inode budget, ~0 % used at switch). Old symlink preserved
as `scratch.groupscratch.bak`.

### Symlinks rescued from old scratch (no regen needed)

| Dataset             | Bulk `.pt` links | Per-cell dir links |
|---------------------|------------------|--------------------|
| archs4_geo          | 1                | 1                  |
| hest1k              | 384              | 384                |
| cellxgene_census    | 4                | 3                  |
| quilt1m (unused)    | 200,410          | 0                  |
| quilt1m/fullres     | (single symlink to dir, unused) |    |

Targets in old GROUP_SCRATCH were `chmod -R a-w` to prevent accidental
writes through the symlinks. They are owned by `moritzs` so the chmod is
reversible (the wrap-up will need to do this).

### Caches generated locally in new $SCRATCH

- 384 hest1k Geneformer per-cell dirs (the half that had no matching
  per-cell dir in GROUP_SCRATCH) — generated during the baseline run.
- UCE caches for smoke datasets (`immgen`, `hest1k_8thsub`).
- frozen_model cache `.pkl`s (forward outputs of locked towers; rebuilt
  per run anyway).

### Still missing

- UCE caches for the production datasets (`archs4_geo`, `hest1k`,
  `cellxgene_census`). Smoke validated the pipeline on `_8thsub`. First
  run of the `uce` config will materialize these.

## How to run

From a Sherlock login node (after lsync has propagated the code):

```bash
cd ~/cellwhisperer_private/src/spotwhisperer_eval/experiments/freezing_encoder_appendix

sbatch sherlock_controller.sh smoke     # UCE prep + 20-step train, no eval
sbatch sherlock_controller.sh baseline  # train baseline ckpt only (no eval)
sbatch sherlock_controller.sh full      # full sweep + CRC eval (5 trains + 545 evals)
```

The controller submits the Snakemake driver as a CPU-only job; the actual
training jobs are sbatched as separate H100 jobs by Snakemake.

## Reporting metrics

Following the `lambda_ablation` convention:

1. `macro_auroc_classwise_dataset_avg` (reviewer-facing primary).
2. `macro_soft_auroc_classwise_dataset_avg` (multi-positive aware).
3. `macro_f1_datasetwise_class_avg` and
   `macro_recall_at_5_datasetwise_class_avg` for operating-point spread.
4. Wall-clock training time per config — relevant for arguing the LUL
   default on compute grounds.

Aggregated CSV path:
`src/spotwhisperer_eval/experiments/freezing_encoder_appendix/results_crc_pathocell.csv`

## Progress Log

### 2026-05-10
- Surveyed UCE wiring (`config.py:83`, `uce_model.py`, processing pipeline).
- Verified `locking_mode` supports `LUL`, `LLL`, `LLU`, `ULL` end-to-end.
- Scaffolded experiments dir + Snakemake rule + controller.
- Snakemake dry-run on Sherlock confirms targets resolve cleanly:
  smoke = 1 train job, full = 5 train + 545 pathocell + 5 metric jobs.
- Wrote `aggregate_results.py` and `appendix_draft.tex` (placeholders).
- Open issue: UCE citation key (Rosen et al. 2023, "Universal Cell
  Embeddings") is not in `main.bib` or `~/wiki/papers/paperpile.bib`.
  Flagged as `TODO_uce_rosen2023` in the appendix draft.

### 2026-05-11
- First smoke hit GROUP_SCRATCH inode wall (20M, exhausted by UCE per-cell
  prep). Switched scratch symlink to $SCRATCH; wiped orphans.
- Hit Ensembl-only `gene_name` on ~25 % of hest1k samples → added
  Ensembl→symbol fallback in `uce_model.py`.
- Swapped UCE checkpoint + vocab: `KuanP/uce-cxg-2025-baseline-8l-512d`
  + `gene_names_full.txt` (19,656 genes); `use_assay_token: false`.
- Smoke `.ckpt` saved 15:27.
- Symlinked Geneformer caches from GROUP_SCRATCH → $SCRATCH (768 hest1k
  bulk + 446 per-cell dirs + 4 cellxgene + 1 archs4 + quilt1m/fullres);
  targets `chmod -R a-w`.
- 384 hest1k bulk `.pt`s had no matching per-cell dir → deleted to force
  re-prep within the training run.

### 2026-05-12
- **Baseline production ckpt saved 02:36** (1.59 GB) after 8h05m on
  H100/sh04-02n05 with 60 CPUs + 900 GB. 3,655 global steps (1 epoch).
  train/clip_loss=2.074, val/clip_loss=4.222. wandb run `wne74jw3`.
- Maintenance window 08:00–18:00 PDT: Sherlock unreachable.
- Code state captured on branch `camera-ready-experiment` (committed at
  ~08:30).

### 2026-05-13
- Resumed full sweep at 19:00 PDT (post-maintenance). Bumped controller
  to 48h + training rule to 24h time limits (now that the next
  maintenance is many days out).
- **LLL ckpt saved** 00:28 (107 MB, projection-heads-only is much smaller
  than baseline). wandb `bvxsjy4j`.
- **LLU ckpt saved** 11:31 (8.28 GB; UNI2 trainable bloats the ckpt).
  Required `bs=64` + `accumulate_grad_batches=8` (effective batch 512)
  to fit on H100 80 GB (\*deviation from main paper, see asterisk).
- **ULL: not trained** (\*asterisk in results). Geneformer-unfrozen
  forward+backward at bs=512 OOM'd the H100 80 GB; bs=64 OOM'd; bs=32
  OOM'd; bs=16 fit but only with `unlocked_fp16=true`, which produced
  NaN loss at step 649 (fp16/bf16 mixed-precision numerical
  instability). Per user direction, accepting this gap rather than
  trading off model size or precision; report as missing in the table.
- **UCE training in progress** (started 09:55, expected ~8h). UCE prep
  skipped a small number of hest1k samples whose `var.gene_name` is
  neither symbol nor Ensembl (UCENoGeneOverlapError fallback + empty
  bulk-`.pt` sentinel in `prepare_dataset_file`).

#### Asterisks for the appendix table

- `llu`: trained with smaller per-step batch (64 with grad_accum=8;
  effective batch 512). All other configs use bs=512 directly.
- `ull`: not trained — H100 80 GB GPU memory not sufficient for
  Geneformer unfrozen at bs=64+; smaller bs only fit with fp16 which
  introduced NaN loss. Report as N/A in the freezing-axis table.

The delta configs have been reverted to their minimal form (only
`locking_mode` overridden) so the bs/grad-accum changes are not
preserved in the canonical config; the deviation is documented here.

## Resume plan (after 18:00 PDT maintenance ends)

1. Launch the three remaining Geneformer configs back-to-back:
   `sbatch sherlock_controller.sh <lll|llu|ull>` (each takes ~8h with
   the same resources; Geneformer caches are warm, so prep is a no-op).
   Or extend the controller with a single target that fans out the four
   Geneformer configs.
2. Launch the `uce` config — first run will materialize UCE caches for
   archs4_geo + hest1k + cellxgene_census (estimate 2–3 h for prep + 8 h
   for training = ~10 h).
3. PathoCellBench eval on the 5 ckpts (~545 leaf jobs per the dry-run).
4. `python aggregate_results.py` → `results_crc_pathocell.csv`.
5. Merge `appendix_draft.tex` into `SpatialWhisperer_ICML/appendix.tex`
   with the real numbers; resolve `TODO_uce_rosen2023` citation key.
6. Insert forward-references at `main.tex` lines 274 / 289.
7. Revert `~/cellwhisperer_private/scratch` to GROUP_SCRATCH:
   `rm scratch && mv scratch.groupscratch.bak scratch`. Optionally
   `chmod -R u+w` on the old targets to undo the read-only marking.

## Status

- [x] Reviewer concern mapped to specific commitment in `final.org`.
- [x] Code paths surveyed; UCE wiring patched (Ensembl→symbol fallback).
- [x] Experiment scaffolded (configs + rule + controller).
- [x] UCE model + vocab decided
  (`KuanP/uce-cxg-2025-baseline-8l-512d` + 19,656-gene `gene_names_full.txt`).
- [x] Smoke test passed (`spotwhisperer_fe_uce_smoke.ckpt`).
- [x] Geneformer caches symlinked from GROUP_SCRATCH; chmod read-only.
- [x] Baseline production ckpt saved (`spotwhisperer_fe_baseline.ckpt`).
- [x] `lll` ckpt saved (`spotwhisperer_fe_lll.ckpt`, 107 MB).
- [x] `llu` ckpt saved (`spotwhisperer_fe_llu.ckpt`, 8.28 GB — bs=64\*).
- [ ] `ull` — **not trained**; H100 80 GB insufficient for Geneformer
      unfrozen; report as N/A with asterisk.
- [ ] `uce` ckpt trained (in progress, expected ~16:00 PDT 2026-05-13).
- [ ] CRC PathoCellBench eval completed for all 5 configs.
- [ ] `results_crc_pathocell.csv` aggregated.
- [ ] Appendix subsection merged into `SpatialWhisperer_ICML/appendix.tex`.
- [ ] Manuscript edits inserted in `main.tex`.
- [ ] `rebuttal/final.org` updated with the final numbers.
- [ ] Scratch symlink reverted to GROUP_SCRATCH.
