#!/bin/bash
#SBATCH --job-name=fe_appendix_ctrl
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/moritzs/fe_appendix_controller_%j.out
#SBATCH --error=/scratch/users/moritzs/fe_appendix_controller_%j.err
#
# Submit with one of:
#   sbatch sherlock_controller.sh smoke  # train *_smoke configs only
#   sbatch sherlock_controller.sh full   # train + eval all production configs
#
# Default (no arg) is `smoke` to prevent accidental full-sweep launches.

set -eo pipefail

PROJECT_DIR=/home/groups/zinaida/moritzs/cellwhisperer_private

TARGET="${1:-smoke}"
case "$TARGET" in
    smoke)                       SNAKE_TARGET=freezing_encoder_appendix_smoke ;;
    full)                        SNAKE_TARGET=freezing_encoder_appendix_all ;;
    eval_trained)                SNAKE_TARGET=freezing_encoder_appendix_eval_trained ;;
    train_remaining)             SNAKE_TARGET=freezing_encoder_appendix_train_remaining ;;
    baseline|lll|llu|ull|uce)    SNAKE_TARGET="$PROJECT_DIR/results/models/jointemb/spotwhisperer_fe_${TARGET}.ckpt" ;;
    *)                           echo "Unknown target: $TARGET (expected smoke|full|train_remaining|baseline|lll|llu|ull|uce)" >&2; exit 2 ;;
esac

export XDG_CACHE_HOME=/scratch/users/moritzs/.cache
mkdir -p "$XDG_CACHE_HOME"

source /home/users/moritzs/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

cd "$PROJECT_DIR"

snakemake --snakefile analysis/Snakefile --profile sm7_slurm --unlock
snakemake --snakefile analysis/Snakefile --profile sm7_slurm "$SNAKE_TARGET"
