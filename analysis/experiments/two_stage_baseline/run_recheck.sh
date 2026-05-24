#!/bin/bash
# Recheck (control) controller. Recomputes:
#   - Bimodal bridge (T↔G+G↔I) on PathoCell + Lizard + PanNuke
#   - Trimodal (with Quilt-1M raw) on PathoCell only
#   - Bimodal I↔T (Quilt-1M only) on PathoCell only
#   - Two-stage UNI2→GF on PathoCell only
#   - OmiCLIP short + extended on PathoCell only
#
# Outputs go under <model>_recheck/... directories so existing CSVs are
# untouched (per user instruction).
#
# Submit with:
#   ssh sherlock 'sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/analysis/experiments/two_stage_baseline/run_recheck.sh'

#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=tsb_recheck
#SBATCH --output=/scratch/users/moritzs/two_stage_baseline_recheck_%j.out
#SBATCH --error=/scratch/users/moritzs/two_stage_baseline_recheck_%j.err

set -eo pipefail

PD=/home/groups/zinaida/moritzs/cellwhisperer_private
cd "$PD"

source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer
set -u

# Spotwhisperer rechecks rely on _recheck.ckpt symlinks under
# results/models/jointemb/. The lizard_/pannuke_/pathocell_metrics_from_scores
# rules use a {model} wildcard, so passing <model>_recheck as model
# automatically routes both inputs (per-dataset scores) and the aggregate
# JSON output under <model>_recheck/.

SW_TRI=spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_recheck
SW_BRI=spotwhisperer_cellxgene_census__archs4_geo__hest1k_recheck
SW_QLT=spotwhisperer_quilt1m_recheck

# 6 PathoCell aggregates + 2 bridge-on-Lizard/PanNuke aggregates = 8 targets.
# These naturally pull in all per-dataset score recomputations.
TARGETS=(
    # PathoCell CRC (all 6 methods)
    "$PD/results/pathocell_evaluation/${SW_TRI}/summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/${SW_BRI}/summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/${SW_QLT}/summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/two_stage_baseline_recheck/summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip_recheck/summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip_pseudobulk_recheck/summary/patch_metrics_from_scores_aggregated.json"
    # Bimodal bridge on Lizard + PanNuke (already on file pre-March; verify reproducibility)
    "$PD/results/pathocell_evaluation/${SW_BRI}/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/${SW_BRI}/pannuke_summary/patch_metrics_from_scores_aggregated.json"
)

snakemake \
    --snakefile analysis/Snakefile \
    --profile sm7_slurm \
    "${TARGETS[@]}"
