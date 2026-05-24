#!/bin/bash
# DeepSpot-comparable held-out eval on the 10 HEST-bench organs.
#
# Submit with:
#   ssh sherlock 'sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/analysis/experiments/decoder_correlation_audit/run_hesteval.sh'

#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=decoder_corr_hesteval
#SBATCH --output=/scratch/users/moritzs/decoder_corr_hesteval_%j.out
#SBATCH --error=/scratch/users/moritzs/decoder_corr_hesteval_%j.err

set -eo pipefail

PD=/home/groups/zinaida/moritzs/cellwhisperer_private
cd "$PD"

source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/group_home/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

UNI2_DIR="${PD}/resources/uni2"
GENE_LIST="${PD}/resources/two_stage_baseline/hest_geneformer_genes.csv"
HESTEVAL_ROOT="${PD}/results"
OUT_DIR="${PD}/results/decoder_correlation_audit"

mkdir -p "${OUT_DIR}"

python -u analysis/experiments/decoder_correlation_audit/scripts/eval_decoder_hesteval.py \
    --decoder-ckpt "${PD}/results/two_stage_baseline/decoder/decoder.ckpt" \
    --gene-list "${GENE_LIST}" \
    --uni2-weights-dir "${UNI2_DIR}" \
    --hesteval-root "${HESTEVAL_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --batch-size 256 \
    --num-workers 4
