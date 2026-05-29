# PLIP QUILT-1M Harmonization Control

## Status: Complete

## Goal

Section 4.3 of the paper claims that curating QUILT-1M captions to match transcriptomics-style text improves I↔T retrieval because it increases overlap in the shared text modality, enabling better transitive transfer through the gene-expression bridge.

This experiment tests whether that improvement is bridge-specific by running the same evaluation on PLIP (`vinid/plip`), a pathology VLM with no gene-expression bridge. If PLIP does not benefit from curation while the bridge model does, the gain is specific to transitive alignment. If PLIP also benefits, the gain is a generic text-quality effect.

## Results

| Model | Original captions (AUROC) | Curated captions (AUROC) | Delta |
|:--|--:|--:|--:|
| PLIP | 0.769 | 0.849 | +0.079 |
| Our bridge model | 0.645 | 0.695 | +0.050 |

PLIP benefits *more* from curation than the bridge model (+0.079 vs +0.050). The curation gain is therefore **not** specific to bridge-mediated alignment — it is a generic text-quality improvement.

## Caption example

- **Original**: `Normal melanocytes are located along the basal layer.`
- **Curated**: `Normal human epidermal melanocytes isolated from the basal layer of skin tissue.`

## Setup

- Model: `vinid/plip` (HuggingFace CLIPModel)
- Data: 20,000 randomly sampled image-caption pairs (seed=0) from the intersection of original and curated QUILT-1M lookup tables
- Same images in both conditions; only caption text changes
- Metric: image-to-text and text-to-image macro AUROC (via torchmetrics multiclass AUROC, matching the SpotWhisperer pipeline), averaged across both directions
- Cluster: SNAP (`il-interactive`, hyperturing2, RTX 8000)

## Implementation

- Inference script: `src/spatialwhisperer_eval/scripts/plip_quilt1m_harmonization_control.py`
- SNAP controller: `src/spatialwhisperer_eval/experiments/plip_harmonization_control/run_snap_controller.sh`

The script merges the two CSV lookup tables by `image_path`, samples 20k pairs, loads images from zip archives on local SSD, encodes with PLIP, computes torchmetrics multiclass retrieval metrics, and writes results.

## SNAP data paths

- Original lookup: `/dfs/user/moritzs/quilt1m_control/inputs/quilt_1M_lookup.csv`
- Curated lookup: `/dfs/user/moritzs/quilt1m_control/inputs/quilt_1M_lookup_curated.csv`
- Image archives: `/lfs/local/0/moritzs/quilt1m_control/inputs/fullres/*.zip` (also on `/dfs`)
- Output (CPU run): `/dfs/user/moritzs/quilt1m_control/plip_harmonization_control/plip_sample20000_seed0/`
- Output (GPU run): `/dfs/user/moritzs/quilt1m_control/plip_harmonization_control/plip_sample20000_seed0_gpu/`

Both runs produced identical results (differences only in 6th decimal place from CPU vs GPU numerics).

## How to reproduce

```bash
cd /sailhome/$USER/cellwhisperer_private
sbatch src/spatialwhisperer_eval/experiments/plip_harmonization_control/run_snap_controller.sh
```

## Notes

- `uv run` with `--extra-index-url https://download.pytorch.org/whl/cu124` is required on hyperturing2 (CUDA 12.9, driver 575.51) to get GPU-compatible torch. Without it, torch falls back to CPU.
- The CPU-only run took ~56 min; the GPU run took ~28 min. The bottleneck in both cases is the 20k-class torchmetrics multiclass AUROC, which runs on CPU regardless.
- Saved embeddings (`.npy`) allow resuming from the metric step without re-encoding.
