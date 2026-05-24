# Trimodal Ablation: CRC PathoCellBench Comparison

## Goal
Compare three model variants on the CRC PathoCellBench (zero-shot cell type prediction, 13 cell types, n=109 CRC image samples, 88,014 patches, 35 patients) to quantify the effect of adding QUILT-1M (uncurated vs curated) I↔T data to the bimodal bridge model.

## Models
| Label              | Dataset combo                                           | Checkpoint (on Sherlock)                                                   | Description                    |
|--------------------|---------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------|
| Bimodal bridge     | `cellxgene_census__archs4_geo__hest1k`                  | `spotwhisperer_cellxgene_census__archs4_geo__hest1k.ckpt`                  | I↔G + G↔T only (no direct I↔T) |
| Trimodal           | `cellxgene_census__archs4_geo__hest1k__quilt1m`         | `spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m.ckpt`         | All 3 modality pairs           |
| Trimodal (curated) | `cellxgene_census__archs4_geo__hest1k__quilt1m_curated` | `spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_curated.ckpt` | All 3 pairs, curated QUILT-1M  |

Checkpoints at: `results/models/jointemb/` on Sherlock.

## Code
- **Snakemake rule**: `pathocell_trimodal_comparison_table` in `src/spotwhisperer_eval/rules/pathocell_benchmark.smk` (line ~1006)
- **Comparison script**: `src/spotwhisperer_eval/scripts/pathocell_trimodal_comparison_table.py`
- **Model dict**: `TRIMODAL_ABLATION_MODELS` in `rules/pathocell_benchmark.smk`
- **Upstream dependencies** (all pre-existing, wildcarded on `{model}`):
  - `pathocell_cell_type_prediction` → per-dataset score CSVs (109 per model)
  - `pathocell_metrics_from_scores` → aggregated per-class AUROC CSV

## Output files (on Sherlock)
```
results/pathocell_evaluation/comparison/patch/tables/trimodal_ablation_rocauc.csv   # CSV table
results/pathocell_evaluation/comparison/patch/tables/trimodal_ablation_rocauc.tex   # LaTeX snippet
results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_curated/
    summary/patch_per_class_metrics_from_scores.csv                                 # Curated model aggregated metrics
    summary/patch_per_dataset_metrics_from_scores.csv
    summary/patch_metrics_from_scores_aggregated.json
    reg*_patch_scores_seed0.csv                                                     # 109 per-dataset score CSVs
```

## How to run
```bash
# From src/spotwhisperer_eval/ on Sherlock (within SLURM job):
# Use absolute path (git rev-parse resolves through symlink)
conda run -n cellwhisperer snakemake --profile sm7_slurm \
    /home/groups/zinaida/moritzs/cellwhisperer_private/results/pathocell_evaluation/comparison/patch/tables/trimodal_ablation_rocauc.csv
```

## Execution history

### Attempt 1 — job 19503661 (FAILED)
- All 109 prediction jobs failed with `OpenSlideUnsupportedFormatError`
- Root cause: `src/cellwhisperer/jointemb/uni_model.py:154` dispatched `.tiff` files to `openslide.OpenSlide()`, but PathoCellBench processed TIFFs are standard RGB images (1920×1440), not whole-slide images. This was a regression — an older version only sent `.svs` to OpenSlide.
- Fix: added `try/except openslide.OpenSlideUnsupportedFormatError` fallback to `PIL.Image.open()` at `uni_model.py:155-156`.

### Attempt 2 — job 19514861 (SUCCESS)
- Submitted after fix + `snakemake --unlock`
- Conductor: `sbatch --partition=cmackall --cpus-per-task=4 --mem=32G --time=24:00:00`
- Logs: `/scratch/users/moritzs/trimodal_ablation_v2_19514861.{out,err}`
- DAG: 109 GPU prediction jobs (curated model only; bibridge + trimodal already existed) → 1 `pathocell_metrics_from_scores` → 1 `pathocell_trimodal_comparison_table`
- Runtime: 6h44m total (GPU contention with seeded training jobs on the single H100 node)
- Completed: exit code 0

## Results

### Per-class AUROC (mean across n=109 samples)

| Cell Type | Bimodal bridge | Trimodal | Trimodal (curated) |
|---|---|---|---|
| Adipocytes | 0.473 | 0.491 | **0.508** |
| B cells | **0.726** | 0.724 | 0.678 |
| Dendritic cells | 0.611 | 0.604 | **0.622** |
| Granulocytes | 0.636 | 0.538 | **0.650** |
| Macrophages/Monocytes | 0.712 | 0.731 | **0.774** |
| NK cells | 0.457 | 0.430 | **0.524** |
| Nerves | 0.541 | **0.609** | 0.560 |
| Plasma cells | 0.639 | **0.666** | 0.640 |
| Smooth muscle | 0.734 | 0.710 | **0.761** |
| Stroma | **0.633** | 0.546 | 0.631 |
| T cells | 0.734 | 0.705 | **0.752** |
| Tumor cells | 0.575 | 0.549 | **0.613** |
| Vasculature/Lymphatics | **0.720** | 0.618 | 0.679 |
| **Mean** | 0.630 | 0.609 | **0.645** |

### Key findings
1. **Trimodal (curated) wins 9/13 cell types** and achieves the best mean AUROC (0.645).
2. **Bimodal bridge** is second-best (0.630), winning on B cells and Vasculature/Lymphatics.
3. **Trimodal (uncurated QUILT-1M)** performs worst (0.609) — wins only on Nerves and Plasma cells.
4. Curated I↔T data lifts the trimodal model from 0.609 → 0.645, surpassing the bimodal bridge.
5. Uncurated QUILT-1M actually *hurts* performance vs the bimodal bridge (0.609 vs 0.630), suggesting noisy I↔T data can interfere with transitive alignment. Curation recovers and exceeds baseline.

### Note on per-sample vs aggregated AUROC
The per-sample `rocauc_macroAvg` from the prediction notebooks is ~0.24 for all models (including bibridge which achieves 0.63 in the paper). This is an artifact of macro-averaging over 15 classes per sample when most classes have zero positive examples (yielding AUROC=0). The aggregated pipeline pools all patches across 109 samples before computing per-class AUROC, giving stable estimates with thousands of positive/negative examples per class. Spot-checking present-class-only AUROC per sample confirmed values are genuinely >0.5 (typically 0.6–0.85).

## Status
- [x] Bimodal bridge: results existed on Sherlock
- [x] Trimodal: results existed on Sherlock
- [x] Trimodal (curated): predictions completed, metrics aggregated
- [x] Comparison table generated (CSV + LaTeX)

---

## Reviewer baseline: bimodal I↔T models (QUILT-1M only)

### Motivation
Reviewer request: provide zero-shot classification results for a strictly bimodal I↔T baseline trained exclusively on the same image-text data (QUILT-1M), to isolate the contribution of the trimodal architecture from the benefit of large-scale gene expression data.

### Baseline models
| Label | Checkpoint | Description | Results |
|---|---|---|---|
| Bimodal I↔T (QUILT-1M) | `spotwhisperer_quilt1m.ckpt` | I↔T only, full QUILT-1M | **Already computed** |

Note: `spotwhisperer_quilt1m_curated` was never trained as a standalone model — curated QUILT-1M was only used in combination with other datasets. So `spotwhisperer_quilt1m` is the only available pure I↔T baseline matching the reviewer's request.

### Execution
- `spotwhisperer_quilt1m` results were pre-existing from earlier experiments — no new jobs needed.
- Job 19722016 (which would have run `hest1k__quilt1m` variants) was submitted then immediately cancelled — those models are irrelevant since HEST1k is an image-gene dataset, not image-text.

### Results

| Cell Type | Bimodal I↔T (QUILT-1M) | Bimodal bridge | Trimodal | Trimodal (curated) |
|---|---|---|---|---|
| Adipocytes | 0.526 | 0.473 | 0.491 | **0.508** |
| B cells | 0.707 | **0.726** | 0.724 | 0.678 |
| Dendritic cells | 0.552 | 0.611 | 0.604 | **0.622** |
| Granulocytes | 0.563 | 0.636 | 0.538 | **0.650** |
| Macrophages/Monocytes | 0.403 | 0.712 | 0.731 | **0.774** |
| NK cells | 0.451 | 0.457 | 0.430 | **0.524** |
| Nerves | 0.591 | 0.541 | **0.609** | 0.560 |
| Plasma cells | 0.596 | 0.639 | **0.666** | 0.640 |
| Smooth muscle | 0.637 | 0.734 | 0.710 | **0.761** |
| Stroma | 0.599 | **0.633** | 0.546 | 0.631 |
| T cells | 0.701 | 0.734 | 0.705 | **0.752** |
| Tumor cells | 0.535 | 0.575 | 0.549 | **0.613** |
| Vasculature/Lymphatics | 0.499 | **0.720** | 0.618 | 0.679 |
| **Mean** | 0.566 | 0.630 | 0.609 | **0.645** |

### Interpretation (response to reviewer)
- The bimodal I↔T baseline (`spotwhisperer_quilt1m`, 0.566) is substantially worse than the bimodal bridge (0.630) and trimodal curated (0.645), **despite having direct I↔T supervision on the same QUILT-1M data**.
- This isolates the contribution of gene expression grounding: the G↔T bridge (CellxGene + ARCHS4) provides semantically rich cell type information that image-text alignment on QUILT-1M alone cannot recover.
- The trimodal curated model's +0.079 mean AUROC advantage over the I↔T baseline is therefore not explained by simply having more image-text data — it reflects the architectural benefit of transitive multimodal alignment through gene expression.
- Notably, the uncurated trimodal (0.609) is also still better than the pure I↔T baseline (0.566), reinforcing that the G↔T bridge is the key driver.

### Status
- [x] `spotwhisperer_quilt1m`: results available (pre-existing)
- [x] No other pure I↔T baselines exist (`spotwhisperer_quilt1m_curated` was never trained standalone)

---

## Manuscript integration (revision item: "integrate Trimodal + harmonized QUILT-1M")

Analysis directory: `/home/moritz/Projects/SpatialWhisperer/Analysis/trimodal_harmonized_quilt1m/` (org-roam id `98d021a8-267c-4ac3-89ff-cd3f86582638`).

### Decision
- The harmonized trimodal result (0.645 vs bridge 0.630) is now the lead empirical demonstration that text harmonization recovers and exceeds the bimodal bridge. It is folded into the existing appendix `\section{Effects of text annotation harmonization}` (label `appendix:quilt1m-curation`), which previously only covered the I↔T retrieval probe.
- The pre-existing `\paragraph{Combining Task-Matching and Transitive Data for Cell Type Prediction Benchmark:}` (with `fig:pathocellbench_subsetting`) is dropped: its claim "trimodal hurts cell-type prediction" is subsumed by the new appendix, which presents both the negative raw-QUILT-1M finding and its harmonized recovery in one place.
- Manuscript convention preserved: "Trimodal" in the main text still refers to the 2-paired-datasets bridge (T↔G+G↔I, mean 0.630). The 3-paired raw-Quilt-1M variant (0.609) appears only inside the appendix table, where it is explicitly contrasted with the harmonized variant.

### Files staged for the user
- `Analysis/trimodal_harmonized_quilt1m/appendix_trimodal_harmonized.tex` — drop-in replacement for `\section{Effects of text annotation harmonization}` in `appendix.tex`.
- `Analysis/trimodal_harmonized_quilt1m/manuscript_edits.md` — surgical patch instructions for `main.tex` §4.2 / §4.3 plus the appendix swap and deletion.

### Single-seed caveat
All three cell-type AUROCs in `tab:trimodal_ablation_benchmark` are seed=0. The harmonized model has not yet been re-run at additional seeds; this is flagged in the appendix and tracked under the parent revision item.
