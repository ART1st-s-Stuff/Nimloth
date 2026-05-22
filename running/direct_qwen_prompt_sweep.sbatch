#!/usr/bin/env bash
#SBATCH --job-name=direct_qwen_prompt
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-16
#SBATCH --requeue
#SBATCH --output=/project/peilab/atst/flower/running/slurm_logs/%x-%j.out
#SBATCH --error=/project/peilab/atst/flower/running/slurm_logs/%x-%j.err

set -euo pipefail

cd /project/peilab/atst/flower

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  source /project/peilab/atst/.env
  set +a
fi

. .venv/bin/activate
export PYTHONUNBUFFERED=1
export HOME=/project/peilab/atst
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/ai2thor-direct-qwen-prompt-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

source running/setup_ai2thor_env.sh

ROOT_OUT=${ROOT_OUT:-outputs/superpod/direct_qwen_prompt_sweep_$(date +%Y%m%d_%H%M%S)}
PLANNER_LORA=${PLANNER_LORA:-models/qwen_planner_lora}
mkdir -p "$ROOT_OUT"

for VARIANT in ${PROMPT_VARIANTS:-baseline recovery balanced_fewshot}; do
  python dev/tmp_direct_qwen_ebnav_eval.py \
    --planner-lora "$PLANNER_LORA" \
    --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
    --embodiedbench-root /project/peilab/atst/EmbodiedBench \
    --output-dir "$ROOT_OUT/$VARIANT" \
    --num-episodes "${NUM_EPISODES:-3}" \
    --max-steps "${MAX_STEPS:-20}" \
    --split "${SPLIT:-all}" \
    --seed "${SEED:-20260522}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --prompt-variant "$VARIANT"
done

python - "$ROOT_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    item = json.loads(path.read_text())
    rows.append({
        "variant": path.parent.name,
        "success": item.get("task_success_rate"),
        "planner_failure": item.get("planner_failure_rate"),
        "actions": item.get("action_distribution"),
        "blocked_forward": item.get("blocked_forward_count"),
        "blocked_forward_repeat": item.get("blocked_forward_repeat_count"),
        "longest_repeat": item.get("longest_repeat_streak"),
        "transitions": item.get("num_transitions"),
    })
summary = {"root": str(root), "variants": rows}
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY
