#!/bin/bash
# Retry the 2 lizard_cell_type_prediction jobs that hit transient
# "Kernel didn't respond in 60 seconds" errors during the main controller
# run (sbatch 23985697). Also re-trigger the lizard summaries that were
# blocked on those scores.

#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=tsb_retry
#SBATCH --output=/scratch/users/moritzs/two_stage_baseline_retry_%j.out
#SBATCH --error=/scratch/users/moritzs/two_stage_baseline_retry_%j.err

set -eo pipefail

PD=/home/groups/zinaida/moritzs/cellwhisperer_private
cd "$PD"

source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer
set -u

TARGETS=(
    "$PD/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m/lizard/glas_4_patch_scores_seed0.csv"
    "$PD/results/pathocell_evaluation/spotwhisperer_quilt1m/lizard/dpath_69_patch_scores_seed0.csv"
    "$PD/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/spotwhisperer_quilt1m/lizard_summary/patch_metrics_from_scores_aggregated.json"
)

snakemake \
    --snakefile analysis/Snakefile \
    --profile sm7_slurm \
    "${TARGETS[@]}"
