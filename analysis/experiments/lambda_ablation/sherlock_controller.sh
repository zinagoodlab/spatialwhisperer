#!/bin/bash
#SBATCH --job-name=lambda_ablation_ctrl
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/moritzs/lambda_ablation_controller_%j.out
#SBATCH --error=/scratch/users/moritzs/lambda_ablation_controller_%j.err

set -eo pipefail

PROJECT_DIR=/home/groups/zinaida/moritzs/cellwhisperer_private
export XDG_CACHE_HOME=/scratch/users/moritzs/.cache
mkdir -p "$XDG_CACHE_HOME"

source /home/users/moritzs/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

cd "$PROJECT_DIR"

snakemake --snakefile analysis/Snakefile --profile sm7_slurm --unlock
snakemake --snakefile analysis/Snakefile --profile sm7_slurm lambda_ablation_all
