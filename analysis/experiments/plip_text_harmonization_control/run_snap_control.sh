#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-il-interactive}"
MODEL="${2:-plip}"

export UV_PROJECT_ENVIRONMENT="/lfs/local/0/$USER/uv-envs/cellwhisperer"
export XDG_CACHE_HOME="/lfs/local/0/$USER/.cache"
export XDG_BIN_HOME="/lfs/local/0/$USER/.local/bin"
export XDG_DATA_HOME="/lfs/local/0/$USER/.local/share"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cellwhisperer_private}"
CONTROL_ROOT="/dfs/user/$USER/quilt1m_control"
SOURCE_SUBSET_CSV="$CONTROL_ROOT/subset/test_individual_clip_scores.csv"
ORIGINAL_H5AD_DIR="/lfs/local/0/$USER/quilt1m_control/subset_h5ads/original"
CURATED_H5AD_DIR="/lfs/local/0/$USER/quilt1m_control/subset_h5ads/curated"
IMAGE_ZIP_ROOT="$CONTROL_ROOT/inputs/fullres"

MANIFEST="$CONTROL_ROOT/subset/section43_patch_subset.csv"

cd "$PROJECT_ROOT"

UV_PY=(uv run --no-progress \
  --with pandas \
  --with numpy \
  --with anndata \
  --with scipy \
  --with torch \
  --with transformers \
  --with pillow \
  --with torchmetrics \
  python)

"${UV_PY[@]}" src/spotwhisperer_eval/experiments/plip_text_harmonization_control/scripts/build_subset_manifest.py \
  --source-csv "$SOURCE_SUBSET_CSV" \
  --output-csv "$MANIFEST"

for CONDITION in original curated; do
  "${UV_PY[@]}" src/spotwhisperer_eval/experiments/plip_text_harmonization_control/scripts/run_quilt_retrieval_control.py \
    --model "$MODEL" \
    --caption-condition "$CONDITION" \
    --subset-manifest "$MANIFEST" \
    --original-h5ad-dir "$ORIGINAL_H5AD_DIR" \
    --curated-h5ad-dir "$CURATED_H5AD_DIR" \
    --image-zip-root "$IMAGE_ZIP_ROOT" \
    --output-dir "$CONTROL_ROOT/results/$MODEL/$CONDITION"
done

"${UV_PY[@]}" src/spotwhisperer_eval/experiments/plip_text_harmonization_control/scripts/summarize_control_results.py \
  --results-root "$CONTROL_ROOT/results" \
  --output-csv "$CONTROL_ROOT/results/control_summary.csv" \
  --output-md "$CONTROL_ROOT/results/control_summary.md"
