#!/bin/bash
# Re-evaluates the HEST-1K UNI2→MLP decoder (wandb run dl047zmb,
# results/two_stage_baseline/decoder/decoder.ckpt) under DeepSpot/Nonchev2025-
# comparable metrics (per-gene + per-organ + per-slide + train-mean baseline).
#
# Submit with:
#   ssh sherlock 'sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/src/spotwhisperer_eval/experiments/decoder_correlation_audit/run_eval.sh'

#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=decoder_corr_audit
#SBATCH --output=/scratch/users/moritzs/decoder_corr_audit_%j.out
#SBATCH --error=/scratch/users/moritzs/decoder_corr_audit_%j.err

set -eo pipefail

PD=/home/groups/zinaida/moritzs/cellwhisperer_private
cd "$PD"

source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

UNI2_DIR="${PD}/resources/uni2"
GENE_LIST="${PD}/resources/two_stage_baseline/hest_geneformer_genes.csv"
HEST_BENCH_ROOT="${PD}/resources/hest_bench_data"
OUT_DIR="${PD}/results/decoder_correlation_audit"

mkdir -p "${OUT_DIR}"

python -u src/spotwhisperer_eval/experiments/decoder_correlation_audit/scripts/eval_decoder_correlations.py \
    --decoder-ckpt "${PD}/results/two_stage_baseline/decoder/decoder.ckpt" \
    --gene-list "${GENE_LIST}" \
    --uni2-weights-dir "${UNI2_DIR}" \
    --hest-bench-root "${HEST_BENCH_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --batch-size 256 \
    --num-workers 4
