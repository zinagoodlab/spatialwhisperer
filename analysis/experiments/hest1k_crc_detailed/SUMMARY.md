# HEST-1K CRC Detailed (Cell-Level Visium HD)

## Goal
Train a trimodal model on cell-level CRC Visium HD data (hest1k_crc_detailed + cellxgene_census) and evaluate on PathoCellBench CRC slides to assess zero-shot cell type prediction from H&E.

## Motivation
The existing HEST-1K training data uses 128um pooled pseudo-spots (~10 cells/spot). For CRC evaluation (PathoCellBench), we need single-cell resolution. Visium HD provides 2um bins that can be aggregated to cell level using CellViT nuclear segmentation (already available in HEST).

## Data Source
Oliveira et al. 2025 (Nat Genet). GEO: GSE280318. All FFPE, 40x, whole-genome (18085 genes).
No cell-type labels deposited (only QC obs columns); Oliveira identified 9 cell types via clustering but didn't release labels. A scRNA-seq reference atlas (260K cells, GSE280311) is available for re-running deconvolution if needed. Not needed for training.

## Datasets

### TENX156 (Stage II-A CRC, Patient 1)
- **205,313 cells** x **18,085 genes**
- 4,785,987 / 8,115,564 2um bins (59%) assigned to cells
- H&E: 40x, 71106x58791 px, 0.27 um/px, FFPE

### TENX128 (10x demo CRC)
- **~222K cells** x **18,085 genes**
- H&E: 40x, 75250x48740 px, 0.27 um/px, FFPE

### Not yet processed (need GEO download, GSE280315_RAW.tar)
- TENX153 (P3 NAT), TENX154 (P5 CRC), TENX155 (P2 CRC)

## Dataset Pipeline: src/datasets/hest1k_crc_detailed

### Rules
1. `download_hest_data`: WSI + CellViT seg + metadata from HuggingFace (`local_dir_use_symlinks=False`)
2. `download_10x_raw`: Raw 2um bins from 10x (MEX format directory; `sc.read_10x_mtx()` + `spatial/tissue_positions.parquet`)
3. `create_cell_level_adata`: expand_nuclei -> read_spots_gdf -> assign_spot_to_cell -> vectorized sum_per_cell -> diagnostics -> h5ad

### Key fixes vs naive HEST usage
- Bypass `iter_hest()` — has `UnboundLocalError` on `tissue_contours_path` when `tissue_seg/` absent
- Vectorized `sum_per_cell` via sparse matmul (seconds vs 44h for HEST's per-cell loop over 205K cells)
- OpenSlide for WSI access in diagnostics (avoids loading 12GB+ image into RAM)

## Training

### Config: src/spotwhisperer_eval/experiments/hest1k_crc_detailed/train_config.yaml
- Datasets: `hest1k_crc_detailed, cellxgene_census`
- Resolution: `detailed_resolution` (0.25 um/px, 224px patches, `cell_level_model: true`)
- Model: UNI2 image + MLP transcriptome + BERT text, locking mode `LUL` (UNI2 frozen)
- 1 epoch, batch_size=128, use_cache=true, use_disk_loading=true

### Completed runs
- **1-GPU (job 18793)**: TENX156 only, 6h7m, WandB run `m4m4`, loss 5.5 -> 0.76
- **4-GPU (job 18806)**: TENX156 + TENX128, ~3h, WandB run `4kx7`, loss 4.2 -> 0.66
- Checkpoints: `results/models/jointemb/crc_visiumhd_{1gpu,4gpu}.ckpt`

## PathoCellBench CRC Evaluation

### Setup
- Model: `crc_visiumhd_4gpu` (TENX156 + TENX128, 427K cells, 1 epoch)
- Eval script: `src/spotwhisperer_eval/experiments/hest1k_crc_detailed/run_pathocell_eval.py`
- 109 CRC TMA datasets from Schuerch et al. 2020 (CODEX-derived ground truth)
- Zero-shot: image -> text similarity scoring per cell

### Fine-grained results (24 classes, 5043 cells total)

| Cell type | AUROC | n cells |
|---|---|---|
| CD3+ T cells | 0.805 | 1 |
| adipocytes | 0.793 | 52 |
| immune cells / vasculature | 0.766 | 9 |
| CD4+ T cells | 0.727 | 20 |
| CD11b+CD68+ macrophages | 0.654 | 8 |
| CD11b+ monocytes | 0.643 | 17 |
| granulocytes | 0.604 | 398 |
| tumor cells | 0.573 | 902 |
| CD4+ T cells CD45RO+ | 0.570 | 321 |
| smooth muscle | 0.513 | 756 |
| plasma cells | 0.492 | 140 |
| CD68+CD163+ macrophages | 0.489 | 1198 |
| CD8+ T cells | 0.466 | 192 |
| vasculature | 0.460 | 109 |
| stroma | 0.418 | 377 |
| B cells | 0.393 | 225 |

Macro AUROC (per-dataset avg): **0.525**. Per-class macro: **0.561**.

### Merged class results (8 classes)

Merged T cell subtypes -> "T cells", macrophage/monocyte/granulocyte/DC subtypes -> "myeloid cells", stroma -> "fibroblasts" (better text query, 0.611 vs 0.354 for "stroma"), vasculature kept as-is. Dropped: dirt, undefined, tumor cells/immune cells, immune cells/vasculature, nerves, adipocytes, immune cells.

Hybrid approach (fine-type scores summed for immune classes, "fibroblasts" query for stroma):

| Class | Pooled AUROC | n cells |
|---|---|---|
| fibroblasts | **0.611** | 377 |
| myeloid cells | **0.591** | 1631 |
| tumor cells | **0.584** | 902 |
| T cells | **0.552** | 539 |
| vasculature | 0.499 | 109 |
| smooth muscle | 0.497 | 756 |
| plasma cells | 0.470 | 140 |
| B cells | 0.328 | 225 |

**Hybrid macro AUROC: 0.517**

### Key analysis insights

#### Class-level score bias (sink effect)
Some text queries produce inherently high scores regardless of actual cell type. Mean score across ALL cells:
- tumor cells: 4.33 (highest — biggest sink)
- adipocytes: 4.12
- granulocytes: 3.95
- B cells: 1.81 (one of the lowest)

This bias means in multi-class argmax, high-bias classes absorb predictions. Column z-score normalization partially corrects this but doesn't fully resolve it because the ranking within each class is also noisy.

#### B cell analysis
- Binary AUROC (B cells vs rest): **0.562** — slight positive signal
- But non-B cells score HIGHER on the "B cells" query than actual B cells (mean 1.90 vs 1.42)
- Only 1 out of 225 B cells gets correctly predicted by argmax (0.4%)
- Top false positives: stroma (90% score above B cell median), CD68+CD163+ macrophages (81%), smooth muscle (75%)
- Text query variants tested: "B cells" (0.562) > "B cells forming dense lymphoid aggregates" (0.547) > "B lymphocytes" (0.530) > "CD20+ B cells" (0.517) > "germinal center B cells" (0.496) > "B cells in lymphoid follicles" (0.484). Plain "B cells" is best — model was trained on simple cellxgene_census cell type names, not tissue-context descriptions.
- CRC B cells often form tertiary lymphoid structures (TLS) visible in H&E, but the model doesn't leverage this — likely because frozen UNI2 patches are too small (224px) for TLS context and 1 epoch is insufficient to learn the association.

#### "fibroblasts" vs "stroma" text query
- "stroma" as text query: AUROC 0.354 (pooled)
- "fibroblasts" as text query: AUROC **0.611** (pooled)
- Large improvement (+0.26) from using a more specific biological term

#### Cross-benchmark results (same model)

| Benchmark | Datasets | Classes | Macro AUROC |
|---|---|---|---|
| PathoCellBench CRC (fine) | 109 | 24 | 0.525 |
| PathoCellBench CRC (coarse) | 109 | 14 | 0.496 |
| Lizard (colon H&E) | 70 | 5 | 0.479 |
| PanNuke (multi-tissue) | 51 | 5 | 0.476 |

- PanNuke colon-only (0.477) vs other tissues (0.487): no tissue-specific benefit
- Lizard by source: CRAG (0.512) > DigestPath/GlaS (have missing classes -> NaN macro)
- Lizard per-class: neutrophil 0.695, lymphocyte 0.498, epithelial 0.430 (CRAG best for most)

### Confusion matrix analysis
- With all 24 fine classes: fibroblasts/tumor cells act as major sinks
- Column z-score normalization reduces but doesn't eliminate sink effect
- With 8 merged classes: tumor cells (0.30 diagonal), myeloid (0.29), fibroblasts (0.36) show some separation; plasma cells (0.10), B cells (0.11) are essentially random
- Best-separable 4 classes: tumor cells, myeloid cells, T cells, smooth muscle (macro AUROC 0.549 restricted to these)

### Root cause of weak performance
- **Frozen UNI2**: image encoder produces generic histology features, not cell-type-discriminative features
- **1 epoch training**: insufficient to learn image-text alignment for cell morphology
- **Text encoder bias**: BERT embeddings for some cell type names (tumor cells, adipocytes) project closer to average image embedding than others (B cells, CD11c+ DCs), creating inherent class score bias
- **Patch size**: 224px context patches may be too small for tissue-level cues (TLS, glandular architecture)
- **Morphological similarity**: lymphocyte subtypes (CD4+, CD8+, B, Tregs) look identical in H&E — discriminating them requires transcriptomic/proteomic information not available at inference

## Infrastructure Pain Points (SNAP/hyperturing2)

### Storage
- `scratch/` and `.pixi/envs/` must NOT live on `/sailhome/` (NFS, 20GB quota)
- **Fix**: symlink both to `/lfs/local/0/moritzs/...`
- Without this: NFS stalls (process in D state, `rpc_wait_bit_killable`), quota exceeded mid-training, `use_disk_loading` appearing 40h slow (actually NFS quota full)

### Pixi env on NFS = hang
- Pixi puts envs in `.pixi/envs/` (project dir) by default -> NFS -> every Python import stalls
- **Fix**: symlink `.pixi/envs` -> SSD, always set `XDG_CACHE_HOME`, `PIXI_CACHE_DIR`, `HF_HOME`

### CUDA PyTorch
- Pixi conda-forge pytorch is CPU-only
- **Fix**: `pixi run pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124` then `pip install "numpy<2"`

### torch 2.6 weights_only=True
- Lightning checkpoint loading and `.pt` file loading fail with `_IncompatibleKeys`
- **Fix**: monkey-patch `torch.load` in `model_io.py`; add `weights_only=False` to all `torch.load` calls in `jointemb.py`

### torch.save of large processed .pt
- With `use_disk_loading=True`, only save `orig_ids` in processed .pt (not full patch tensors)
- See `jointemb.py:~700`

### CUDA OOM at unfreezing
- Batch size 512 OOMs on RTX8000 (44GB) when warmup ends and U towers unfreeze
- **Fix**: batch_size=128

### Multi-GPU DDP
- Needs `--trainer.strategy=ddp_find_unused_parameters_true` (frozen UNI2 has unused params)
- 4-GPU with nproc=4: ~0.15 it/s = 2.8x throughput vs 1-GPU

## Status
- [x] Step 1: Verify H&E quality for all 5 samples (all 40x, 0.27 um/px)
- [x] Step 2: Verify cell-level coordinates + diagnostics plot
- [x] Step 3: Build dataset pipeline (TENX156 + TENX128 done)
- [x] Step 4: Training completed (1-GPU: 6h, 4-GPU: 3h)
- [x] Step 5: PathoCellBench CRC eval (109 datasets) + Lizard + PanNuke
- [x] Step 6: Analysis of class biases, text query optimization, confusion matrices
- [ ] Step 7: Improve: more epochs, unfreeze UNI2, more CRC samples
- [ ] Step 8: Download TENX153/154/155 from GEO (GSE280315_RAW.tar)
