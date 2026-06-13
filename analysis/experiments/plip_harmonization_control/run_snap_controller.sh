#!/bin/bash
#SBATCH --job-name=plip-q1m-control
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00

set -euo pipefail

export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/cellwhisperer
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share

cd /sailhome/$USER/cellwhisperer_private

uv run --no-progress \
  --index-url https://download.pytorch.org/whl/cu124 \
  --with "torch>=2.6.0" \
  --with torchmetrics \
  --with pandas \
  --with numpy \
  --with transformers \
  --with pillow \
  --with tabulate \
  python analysis/scripts/plip_quilt1m_harmonization_control.py \
    --original-metadata /dfs/user/$USER/quilt1m_control/inputs/quilt_1M_lookup.csv \
    --curated-metadata /dfs/user/$USER/quilt1m_control/inputs/quilt_1M_lookup_curated.csv \
    --image-zip-root /lfs/local/0/$USER/quilt1m_control/inputs/fullres \
    --output-dir /dfs/user/$USER/quilt1m_control/plip_harmonization_control/plip_sample20000_seed0 \
    --sample-size 20000 \
    --seed 0 \
    --model-name vinid/plip \
    --bridge-original 0.645 \
    --bridge-curated 0.695
