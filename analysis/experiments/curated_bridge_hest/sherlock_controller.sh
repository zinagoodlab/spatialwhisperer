#!/bin/bash
#SBATCH --job-name=curated_bridge_hest_ctrl
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/moritzs/curated_bridge_hest_controller_%j.out
#SBATCH --error=/scratch/users/moritzs/curated_bridge_hest_controller_%j.err

set -eo pipefail

PROJECT_DIR=/home/groups/zinaida/moritzs/cellwhisperer_private
export XDG_CACHE_HOME=/scratch/users/moritzs/.cache
mkdir -p "$XDG_CACHE_HOME"

source /home/users/moritzs/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

cd "$PROJECT_DIR"

# Run HEST benchmark for the curated bridge model (no hest1k training data)
# The uncurated version results already exist.
# Target: metrics.csv for all 10 HEST datasets
snakemake --snakefile analysis/Snakefile --profile sm7_slurm --unlock
snakemake --snakefile analysis/Snakefile --profile sm7_slurm \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___IDC/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___PRAD/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___PAAD/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___SKCM/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___COAD/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___READ/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___CCRCC/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___HCC/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___LUNG/metrics.csv \
    ${PROJECT_DIR}/results/spotwhisperer_eval/csv_logging/hest_eval___spotwhisperer_cellxgene_census__archs4_geo__quilt1m_curated___LYMPH_IDC/metrics.csv
