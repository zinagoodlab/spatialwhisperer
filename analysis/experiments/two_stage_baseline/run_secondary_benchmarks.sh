#!/bin/bash
# Snakemake controller for the "Two-stage pipeline baselines" analysis (Lizard + PanNuke).
# Computes the missing per-method × per-benchmark scores listed in the TODO item
# Apply revision items > Two-stage pipeline (parent file:
# 20260406090326-icml_spatialwhisperer_methods_paper.org).
#
# Submits child SLURM jobs via the sm7_slurm profile (max 20 parallel).
#
# Submit with:
#   ssh sherlock 'sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/src/spotwhisperer_eval/experiments/two_stage_baseline/run_secondary_benchmarks.sh'

#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=tsb_secondary
#SBATCH --output=/scratch/users/moritzs/two_stage_baseline_secondary_%j.out
#SBATCH --error=/scratch/users/moritzs/two_stage_baseline_secondary_%j.err

set -eo pipefail

PD=/home/groups/zinaida/moritzs/cellwhisperer_private
cd "$PD"

# Conda's activate.d hooks reference a few unbound vars on first activation;
# only enable nounset after the env is live.
source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer
set -u

# 5 methods × 2 benchmarks = 10 aggregate targets.
# Trimodal + Quilt1m use the existing lizard/pannuke_cell_type_prediction rules.
# Two-stage uses the new two_stage_lizard/pannuke_predict rules.
# OmiCLIP short + extended use the new omiclip_secondary_score rules.
TARGETS=(
    "$PD/results/pathocell_evaluation/two_stage_baseline/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/two_stage_baseline/pannuke_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m/pannuke_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/spotwhisperer_quilt1m/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/spotwhisperer_quilt1m/pannuke_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip/pannuke_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip_pseudobulk/lizard_summary/patch_metrics_from_scores_aggregated.json"
    "$PD/results/pathocell_evaluation/omiclip_pseudobulk/pannuke_summary/patch_metrics_from_scores_aggregated.json"
)

snakemake \
    --snakefile src/spotwhisperer_eval/Snakefile \
    --profile sm7_slurm \
    "${TARGETS[@]}"
