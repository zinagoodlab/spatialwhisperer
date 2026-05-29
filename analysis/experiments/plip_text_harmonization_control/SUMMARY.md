# PLIP Text Harmonization Specificity Control

## Goal

Test whether QUILT-1M caption curation improves retrieval for a pathology VLM with no gene-expression bridge. The main control is `vinid/plip`. If needed, the same pipeline can also be run with CONCH as a secondary control.

This directly addresses the Section 4.3 interpretation: if the bridge model benefits from transcriptomics-style rewrites but PLIP does not, then the gain is specific to bridge-mediated alignment rather than generic text polishing.

## Status

- [x] Experiment scaffold created
- [x] SNAP paths identified for raw QUILT-1M, curated captions, and the existing Section 4.3 subset
- [x] PLIP inference + retrieval metric scripts added
- [x] SNAP submission scripts added
- [ ] PLIP run on SNAP
- [ ] Optional CONCH run on SNAP
- [ ] Final rebuttal table filled with measured AUROCs

## SNAP data paths

- Raw QUILT-1M images: `~/oak/moritzs/cellwhisperer/resources/quilt1m/fullres`
- Original filtered captions: `~/oak/moritzs/cellwhisperer/results/quilt1m/quilt_1M_lookup.csv`
- Curated captions: `~/oak/moritzs/cellwhisperer/results/quilt1m_curated/quilt_1M_lookup_curated.csv`
- Original crop h5ads: `~/oak/moritzs/cellwhisperer/results/quilt1m/h5ads`
- Curated crop h5ads: `~/oak/moritzs/cellwhisperer/results/quilt1m_curated/h5ads`
- Section 4.3 subset source: `~/oak/moritzs/cellwhisperer/results/spatialwhisperer_eval/csv_logging/sweval___cellxgene_census__archs4_geo__hest1k___quilt1m_curated/test_individual_clip_scores.csv`
- Control output root: `/dfs/user/moritzs/quilt1m_control`

## What was added

- Rule file: `src/spatialwhisperer_eval/rules/quilt_text_harmonization_control.smk`
- Subset manifest builder: `src/spatialwhisperer_eval/experiments/plip_text_harmonization_control/scripts/build_subset_manifest.py`
- Control inference script: `src/spatialwhisperer_eval/experiments/plip_text_harmonization_control/scripts/run_quilt_retrieval_control.py`
- Result aggregator: `src/spatialwhisperer_eval/experiments/plip_text_harmonization_control/scripts/summarize_control_results.py`
- SNAP batch helpers:
  - `src/spatialwhisperer_eval/experiments/plip_text_harmonization_control/run_snap_control.sh`
  - `src/spatialwhisperer_eval/experiments/plip_text_harmonization_control/submit_snap_control.sh`

## Evaluation design

- Subset: exact 20k crop subset from the existing Section 4.3 curated retrieval run, extracted from `orig_ids`
- Images: unchanged across conditions; only captions switch between original and curated h5ads
- Conditions:
  - `original`
  - `curated`
- Metrics recorded per condition:
  - image-to-text macro AUROC
  - text-to-image macro AUROC
  - mean macro AUROC across both directions (used for the compact rebuttal table)

## Outputs

The SNAP run writes to `/dfs/user/moritzs/quilt1m_control/`:

- `subset/section43_patch_subset.csv`
- `results/<model>/<condition>/metrics.json`
- `results/<model>/<condition>/per_sample_scores.csv`
- `results/<model>/<condition>/similarity_matrix.npz`
- `results/control_summary.csv`
- `results/control_summary.md`

`per_sample_scores.csv` contains the matched-pair score, retrieval rank, and hit@k per sample. Exact AUROC recomputation uses the saved `similarity_matrix.npz`, which stores the full score matrix compactly because a full CSV would be unmanageably large for 20k x 20k pairs.

## How to run on SNAP

Testing:

```bash
cd ~/cellwhisperer_private/src/spatialwhisperer_eval/experiments/plip_text_harmonization_control
bash submit_snap_control.sh il-interactive plip
```

Production:

```bash
cd ~/cellwhisperer_private/src/spatialwhisperer_eval/experiments/plip_text_harmonization_control
bash submit_snap_control.sh il plip
```

Optional CONCH run (requires license access and `HUGGINGFACE_TOKEN`):

```bash
bash submit_snap_control.sh il-interactive conch
```

## Planned final table

| Model | Original captions (AUROC) | Curated captions (AUROC) | Delta |
|---|---:|---:|---:|
| PLIP | pending | pending | pending |
| Our bridge model | 0.645 | 0.695 | +0.050 |

If CONCH is run successfully, it will be appended to the same summary table.
