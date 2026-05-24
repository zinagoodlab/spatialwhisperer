# Curated Bridge HEST Benchmark: quilt1m vs quilt1m_curated (without hest1k)

## Motivation

During the revision phase (response to 9wBy Q4), we repeated Section 4.3 analysis
(similarity of QUILT-1M image-text pairs; uncurated vs curated). The expectation was
that in PLIP (no transcriptomics bridge), curation should not enhance similarity
because text alignment benefits from our transcriptomics formulation. However, PLIP
also showed improvement from curation (+0.079 vs our +0.050), suggesting the gain is
generic text quality, not bridge-specific.

This jeopardizes the conclusion that "cross-dataset modality alignment is the reason
for improved performance" (experimental support for O(sqrt(delta)) bound). The
strongest counter-evidence would be to train a model on G<->T + T<->I data (curated)
without any direct I<->G data and evaluate on G<->I tasks. If curated text improves
G<->I retrieval even in the bridge-only setting (no hest1k), that would show the text
bridge matters.

**This experiment**: evaluate the **bridge-only** model (no hest1k training data)
with curated vs uncurated QUILT-1M on HEST benchmark (G<->I retrieval). If the
curated bridge model improves, it demonstrates that text harmonization propagates
through the bridge to improve cross-modal retrieval.

## Models

| Label               | Dataset combo                               | Checkpoint                                                           | Description                         |
|---------------------|---------------------------------------------|----------------------------------------------------------------------|-------------------------------------|
| Bridge (uncurated)  | `cellxgene_census__archs4_geo__quilt1m`     | `spotwhisperer_cellxgene_census__archs4_geo__quilt1m.ckpt`           | G<->T + T<->I (uncurated QUILT-1M) |
| Bridge (curated)    | `cellxgene_census__archs4_geo__quilt1m_curated` | `spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated.ckpt` | G<->T + T<->I (curated QUILT-1M)   |

Checkpoints at: `results/models/jointemb/` on Sherlock.

Both models have **no direct I<->G training data** (no hest1k). They must bridge
image-to-transcriptome through the shared text modality. This is the cleanest test
of whether text harmonization improves cross-modal transfer.

## Evaluation

HEST benchmark (10 tissue types): IDC, PRAD, PAAD, SKCM, COAD, READ, CCRCC, HCC, LUNG, LYMPH_IDC

Metric: `test_retrieval/transcriptome_image/rocauc_macroAvg` (and f1)

## Pre-existing Results

- **Bridge (uncurated)**: All 10 HEST datasets already evaluated (Dec 2025)
- **Bridge (curated)**: Checkpoint exists, **no HEST eval yet** -- needs to be run

## Code

- **Snakemake rule**: `hest_spotwhisperer_test` in `src/spotwhisperer_eval/rules/hest_benchmark.smk`
- **Controller script**: `sherlock_controller.sh` (this directory)
- **Comparison script**: `compare_results.py` (this directory)

## How to Run

```bash
# From local machine:
ssh sherlock "sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/src/spotwhisperer_eval/experiments/curated_bridge_hest/sherlock_controller.sh"
```

## Execution History

### Attempt 1 -- job 19971889 (SUCCESS)
- Submitted 2026-03-30, all 10 sub-jobs completed
- Runtime per dataset: 12--39 min (PRAD largest at 27G, took 35 min)
- Logs: `/scratch/users/moritzs/curated_bridge_hest_controller_19971889.{out,err}`

## Results

### Mean AUROC across 10 HEST datasets

| Direction | Uncurated | Curated | Delta | Wins | Wilcoxon p (one-sided) | Sign test p |
|-----------|-----------|---------|-------|------|------------------------|-------------|
| I→G       | 0.587     | 0.608   | **+0.021** | 9/10 | 0.024 | 0.011 |
| G→I       | 0.590     | 0.606   | **+0.016** | 9/10 | 0.032 | 0.011 |

### Per-dataset AUROC (transcriptome→image direction)

| Dataset   | Uncurated | Curated | Delta  |
|-----------|-----------|---------|--------|
| CCRCC     | 0.656     | 0.679   | +0.023 |
| COAD      | 0.523     | 0.543   | +0.021 |
| HCC       | 0.521     | 0.483   | -0.038 |
| IDC       | 0.503     | 0.518   | +0.015 |
| LUNG      | 0.520     | 0.543   | +0.023 |
| LYMPH_IDC | 0.679     | 0.697   | +0.018 |
| PAAD      | 0.501     | 0.544   | +0.043 |
| PRAD      | 0.735     | 0.756   | +0.021 |
| READ      | 0.717     | 0.729   | +0.012 |
| SKCM      | 0.549     | 0.571   | +0.022 |

Curated wins on **9/10 datasets** (both retrieval directions). Only HCC shows a regression.

Statistical significance: Wilcoxon signed-rank test (one-sided, curated > uncurated)
p=0.024 (I→G) and p=0.032 (G→I). Sign test (binomial): p=0.011 for both.

### Interpretation

Text harmonization (curating QUILT-1M captions to match transcriptomics-style text)
improves G<->I retrieval in the **bridge-only** setting where there is no direct I<->G
training data. The model must transfer through text, and curated text provides a better
bridge. This supports the claim that cross-dataset same-modality alignment (reducing
delta in the O(sqrt(delta)) bound) is the mechanism, not just generic text quality.
