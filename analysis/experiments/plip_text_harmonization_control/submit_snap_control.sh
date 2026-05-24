#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-il-interactive}"
MODEL="${2:-plip}"

TIME="04:00:00"
if [[ "$PARTITION" == "il" ]]; then
  TIME="12:00:00"
fi

mkdir -p "/dfs/user/$USER/quilt1m_control/logs"

sbatch \
  --account=infolab \
  --partition="$PARTITION" \
  --nodelist=hyperturing2 \
  --gres=gpu:rtx8000:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time="$TIME" \
  --output="/dfs/user/$USER/quilt1m_control/logs/${MODEL}_%j.out" \
  --error="/dfs/user/$USER/quilt1m_control/logs/${MODEL}_%j.err" \
  --wrap="bash $HOME/cellwhisperer_private/src/spotwhisperer_eval/experiments/plip_text_harmonization_control/run_snap_control.sh $PARTITION $MODEL"
