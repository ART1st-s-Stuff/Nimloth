#!/usr/bin/env bash
#SBATCH --job-name=sft1-parent-eval
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/sft1_parent_vagen_eval/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/sft1_parent_vagen_eval/slurm/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120
#SBATCH --gres=gpu:6
#SBATCH --mem=480G

set -euo pipefail

: "${EVAL_WORKTREE:?EVAL_WORKTREE is required}"
: "${EVAL_COMMIT:?EVAL_COMMIT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${SFT1_MODEL:?SFT1_MODEL is required}"
: "${VAGEN_MODEL:?VAGEN_MODEL is required}"
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${SFT1_WANDB_RUN_NAME:?SFT1_WANDB_RUN_NAME is required}"
: "${SFT1_WANDB_RUN_ID:?SFT1_WANDB_RUN_ID is required}"
: "${VAGEN_WANDB_RUN_NAME:?VAGEN_WANDB_RUN_NAME is required}"
: "${VAGEN_WANDB_RUN_ID:?VAGEN_WANDB_RUN_ID is required}"

SL=/cm/shared/apps/slurm/current/bin
ARM_SCRIPT=${EVAL_WORKTREE}/experiments/training/sft1/run_parent_checkpoint_eval_arm.sh
CONTROLLER_LOG=${OUTPUT_ROOT}/controller.log
STEP_PIDS=()

cleanup() {
  set +e
  for pid in "${STEP_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${STEP_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing non-empty or pre-existing output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/sft1" "${OUTPUT_ROOT}/vagen"
mapfile -t NODES < <("${SL}/scontrol" show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 2 )); then
  echo "expected exactly two allocated nodes, got ${NODES[*]}" >&2
  exit 2
fi

cat > "${OUTPUT_ROOT}/README.md" <<EOF
# SFT1 parent and VAGEN parent heldout evaluation

- Purpose: measure success before SFT2 for the exact SFT1 parent and its VAGEN parent.
- Nimloth commit: ${EVAL_COMMIT}
- VAGEN commit: 192c35a91f3941b72d5e1272af6603ef7a7d93e0
- SFT1 checkpoint: ${SFT1_MODEL}
- VAGEN checkpoint: ${VAGEN_MODEL}
- Data: test split, base/common_sense/complex_instruction/visual_appearance/long_horizon, seeds 1..60 each (300 trajectories per arm).
- Sampling: do_sample=false, temperature=0, top_p=1, top_k=-1, n=1, 512 response tokens per turn, 20 turns, one action per turn.
- Navigation source-eval dynamics: resolution=255, step_length=0.3, success_threshold=1.0, format_reward=0, per_turn_format_reward=0.01, success_reward=1.0.
- Action formats: SFT1 k16 injected Nimloth tokens; VAGEN source-compatible XML action format.
- Modules: every model parameter is frozen; validation only, no optimizer and no checkpoint output.
- Resume: SFT1 atomically resumes contiguous per-dataset episode prefixes. VAGEN emits one strict 300-row dump; a failed body must use a new attempt directory.
- W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}; ${SFT1_WANDB_RUN_NAME} (${SFT1_WANDB_RUN_ID}) and ${VAGEN_WANDB_RUN_NAME} (${VAGEN_WANDB_RUN_ID}).
- Slurm: ${SLURM_JOB_ID}, normal, nodes=${NODES[*]}, 6 GPUs and 120 CPUs per node, 3 hour limit.
EOF

{
  echo "time=$(date --iso-8601=seconds)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "nodes=${NODES[*]}"
  echo "commit=${EVAL_COMMIT}"
  echo "output=${OUTPUT_ROOT}"
  echo "sft1_checkpoint=${SFT1_MODEL}"
  echo "vagen_checkpoint=${VAGEN_MODEL}"
  echo "resources=normal_nodes2_gpu6_per_node_cpu120_per_node_mem480G_per_node"
} | tee "${CONTROLLER_LOG}"

COMMON_EXPORTS=(
  ALL
  "EVAL_WORKTREE=${EVAL_WORKTREE}"
  "EVAL_COMMIT=${EVAL_COMMIT}"
  "WANDB_ENTITY=${WANDB_ENTITY}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
)

"${SL}/srun" --exclusive --exact --kill-on-bad-exit=0 \
  --nodes=1 --nodelist="${NODES[0]}" --ntasks=1 --cpus-per-task=120 \
  --gres=gpu:6 \
  --export="$(IFS=,; echo "${COMMON_EXPORTS[*]}")",MODEL_PATH="${SFT1_MODEL}",ARM_OUTPUT="${OUTPUT_ROOT}/sft1",WANDB_RUN_NAME="${SFT1_WANDB_RUN_NAME}",WANDB_RUN_ID="${SFT1_WANDB_RUN_ID}" \
  bash "${ARM_SCRIPT}" sft1 \
  > "${OUTPUT_ROOT}/sft1_step.log" 2>&1 &
STEP_PIDS+=("$!")

"${SL}/srun" --exclusive --exact --kill-on-bad-exit=0 \
  --nodes=1 --nodelist="${NODES[1]}" --ntasks=1 --cpus-per-task=120 \
  --gres=gpu:6 \
  --export="$(IFS=,; echo "${COMMON_EXPORTS[*]}")",MODEL_PATH="${VAGEN_MODEL}",ARM_OUTPUT="${OUTPUT_ROOT}/vagen",WANDB_RUN_NAME="${VAGEN_WANDB_RUN_NAME}",WANDB_RUN_ID="${VAGEN_WANDB_RUN_ID}" \
  bash "${ARM_SCRIPT}" vagen \
  > "${OUTPUT_ROOT}/vagen_step.log" 2>&1 &
STEP_PIDS+=("$!")

failed=0
set +e
for index in "${!STEP_PIDS[@]}"; do
  if ! wait "${STEP_PIDS[$index]}"; then
    failed=1
    arm=sft1
    (( index == 1 )) && arm=vagen
    echo "arm_failed=${arm}" | tee -a "${CONTROLLER_LOG}"
    tail -160 "${OUTPUT_ROOT}/${arm}_step.log" >&2 || true
  fi
done
set -e
(( failed == 0 )) || exit 5
[[ -s "${OUTPUT_ROOT}/sft1/done.flag" && -s "${OUTPUT_ROOT}/vagen/done.flag" ]]

"/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sft1 = json.loads((root / "sft1" / "summary.json").read_text())
vagen = json.loads((root / "vagen" / "summary.json").read_text())
comparison = {
    "status": "ALL_OK",
    "sft1_success_rate": sft1["metrics"]["overall"]["success_rate"],
    "vagen_success_rate": vagen["metrics"]["overall"]["success_rate"],
    "sft1_minus_vagen": (
        sft1["metrics"]["overall"]["success_rate"]
        - vagen["metrics"]["overall"]["success_rate"]
    ),
    "by_eval_set": {
        name: {
            "sft1": sft1["metrics"]["by_eval_set"][name]["success_rate"],
            "vagen": vagen["metrics"]["by_eval_set"][name]["success_rate"],
        }
        for name in sft1["metrics"]["by_eval_set"]
    },
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(comparison, sort_keys=True))
PY
echo "ALL_OK" > "${OUTPUT_ROOT}/done.flag"
echo "status=ALL_OK time=$(date --iso-8601=seconds)" | tee -a "${CONTROLLER_LOG}"
