#!/bin/bash
#SBATCH --job-name=sft2-mcts-eval
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/sft2_mcts_pre_rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/sft2_mcts_pre_rl/slurm/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:6
#SBATCH --mem=512G

set -euo pipefail

: "${EVAL_WORKTREE:?EVAL_WORKTREE is required}"
: "${EVAL_COMMIT:?EVAL_COMMIT is required}"
: "${SFT2_CHECKPOINT:?SFT2_CHECKPOINT is required}"
: "${EVAL_OUTPUT:?EVAL_OUTPUT is required}"
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
SL=/cm/shared/apps/slurm/current/bin
ENV_PORT=$((18400 + SLURM_JOB_ID % 1000))
ENV_URL=http://127.0.0.1:${ENV_PORT}
EVAL_SETS=(base common_sense complex_instruction visual_appearance long_horizon)
EPISODES_PER_SET=60
ATTEMPT_ROOT=${EVAL_OUTPUT}/.attempts/job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}
ENV_LOG=${EVAL_OUTPUT}/env_server_${SLURM_JOB_ID}.log
CONTROLLER_LOG=${EVAL_OUTPUT}/controller_${SLURM_JOB_ID}.log
ENV_STEP_PID=
CHILD_PIDS=()
CHILD_SETS=()
CHILD_ATTEMPTS=()

mkdir -p "${EVAL_OUTPUT}" "${EVAL_OUTPUT}/eval_sets" "${ATTEMPT_ROOT}"

export PATH=${ROOT}/.venv-vagen-main/bin:${PATH}
export PYTHONPATH=${EVAL_WORKTREE}/src:${ROOT}/external/VAGEN:${ROOT}/external/VAGEN/verl:${ROOT}/external/le-wm:${PYTHONPATH:-}
export HOME=${ROOT}/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_DIR=${EVAL_OUTPUT}/wandb
mkdir -p "${HOME}" "${WANDB_DIR}" /project/peilab/atst/nimloth/outputs/experiments/sft2_mcts_pre_rl/slurm
if test -f /project/peilab/atst/flower/.env; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi

cleanup() {
  set +e
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if test -n "${ENV_STEP_PID}"; then
    kill "${ENV_STEP_PID}" >/dev/null 2>&1 || true
    wait "${ENV_STEP_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

{
  echo "time=$(date --iso-8601=seconds)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=${SLURM_JOB_NODELIST}"
  echo "commit=${EVAL_COMMIT}"
  echo "checkpoint=${SFT2_CHECKPOINT}"
  echo "output=${EVAL_OUTPUT}"
  echo "eval_sets=${EVAL_SETS[*]}"
  echo "episodes_per_set=${EPISODES_PER_SET}"
  echo "mcts=H1_K4_sim100_c1"
  echo "sampling=temp0.7_top_p0.95_max_tokens512"
  echo "resources=normal_1node_6gpu_128cpu_512G"
  echo "wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME}/${WANDB_RUN_ID}"
} | tee "${CONTROLLER_LOG}"

test "$(git -C "${EVAL_WORKTREE}" rev-parse HEAD)" = "${EVAL_COMMIT}"
git -C "${EVAL_WORKTREE}" diff --quiet
git -C "${EVAL_WORKTREE}" diff --cached --quiet
test -x "${PY}"
test -f "${SFT2_CHECKPOINT}/training_state.pt"
test -f "${SFT2_CHECKPOINT}/wm_predictor/predictor.pt"
test -f "${SFT2_CHECKPOINT}/state_proj.pt"
test -f "${SFT2_CHECKPOINT}/value_head/value_head.pt"

"${PY}" - "${SFT2_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path
from nimloth.training.sft2.mcts_evaluation import load_sft2_mcts_evaluation_contract

contract = load_sft2_mcts_evaluation_contract(Path(sys.argv[1]))
if (contract.history_size, contract.prediction_horizon, contract.action_count) != (1, 4, 8):
    raise RuntimeError(f"unexpected evaluation contract: {contract}")
print(json.dumps(contract.__dict__, default=str, sort_keys=True), flush=True)
PY

"${PY}" - "${ROOT}/external/VAGEN/vagen/env/navigation/datasets" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"):
    records = json.loads((root / f"{name}.json").read_text())
    if len(records) != 60:
        raise RuntimeError(f"{name} expected 60 eval tasks, got {len(records)}")
    print(json.dumps({"eval_set": name, "tasks": len(records)}), flush=True)
PY

# One dedicated GPU owns the real AI2-THOR/VAGEN environment service.
"${SL}/srun" --exclusive --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 \
  bash -lc "
    set -euo pipefail
    export PATH='${ROOT}/.venv-vagen-main/bin':\"\${PATH}\"
    export PYTHONPATH='${ROOT}/external/VAGEN:${ROOT}/external/VAGEN/verl'
    source '${EVAL_WORKTREE}/experiments/training/baseline/setup_ai2thor_env.sh'
    cd '${ROOT}/external/VAGEN'
    exec '${PY}' -m vagen.server.server \\
      server.host=0.0.0.0 server.port='${ENV_PORT}' \\
      use_state_reward=False navigation.devices='[0]' navigation.max_workers=16
  " > "${ENV_LOG}" 2>&1 &
ENV_STEP_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS --max-time 10 "${ENV_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${ENV_STEP_PID}" >/dev/null 2>&1; then
    tail -200 "${ENV_LOG}" >&2 || true
    exit 4
  fi
  sleep 3
done
curl -fsS --max-time 10 "${ENV_URL}/health" >/dev/null
echo "environment_ready=${ENV_URL}" | tee -a "${CONTROLLER_LOG}"

for eval_set in "${EVAL_SETS[@]}"; do
  final_dir=${EVAL_OUTPUT}/eval_sets/${eval_set}
  if test -s "${final_dir}/rollout_summary.json"; then
    echo "resume_skip=${eval_set}" | tee -a "${CONTROLLER_LOG}"
    continue
  fi
  attempt_dir=${ATTEMPT_ROOT}/${eval_set}
  log_path=${EVAL_OUTPUT}/${eval_set}_${SLURM_JOB_ID}.log
  mkdir -p "${attempt_dir}"
  "${SL}/srun" --exclusive --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 \
    bash -lc "
      set -euo pipefail
      export PATH='${ROOT}/.venv-vagen-main/bin':\"\${PATH}\"
      export PYTHONPATH='${PYTHONPATH}'
      export HOME='${HOME}' HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' TORCH_HOME='${TORCH_HOME}'
      export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export TRITON_CACHE_DIR='/tmp/triton_mcts_${SLURM_JOB_ID}_${eval_set}'
      export XDG_CACHE_HOME='/tmp/xdg_mcts_${SLURM_JOB_ID}_${eval_set}'
      mkdir -p \"\${TRITON_CACHE_DIR}\" \"\${XDG_CACHE_HOME}\"
      cd '${EVAL_WORKTREE}'
      exec '${PY}' experiments/training/sft2/eval_mcts_rollout.py \\
        --sft2-checkpoint '${SFT2_CHECKPOINT}' \\
        --env-url '${ENV_URL}' \\
        --output-dir '${attempt_dir}' \\
        --eval-sets '${eval_set}' --split test \\
        --episodes-per-eval-set '${EPISODES_PER_SET}' --seed-offset 1 --max-steps 20 \\
        --temperature 0.7 --top-p 0.95 --max-response-tokens 512 \\
        --num-simulations 100 --exploration-constant 1.0 \\
        --tensor-parallel-size 1 --planner-device cuda:0 \\
        --max-model-len 32768 --gpu-memory-utilization 0.60 \\
        --vllm-mm-processor-cache-gb 0
    " > "${log_path}" 2>&1 &
  CHILD_PIDS+=("$!")
  CHILD_SETS+=("${eval_set}")
  CHILD_ATTEMPTS+=("${attempt_dir}")
done

failed=0
set +e
for index in "${!CHILD_PIDS[@]}"; do
  wait "${CHILD_PIDS[$index]}"
  code=$?
  eval_set=${CHILD_SETS[$index]}
  attempt_dir=${CHILD_ATTEMPTS[$index]}
  final_dir=${EVAL_OUTPUT}/eval_sets/${eval_set}
  if test "${code}" -eq 0 && test -s "${attempt_dir}/rollout_summary.json"; then
    mv "${attempt_dir}" "${final_dir}"
    echo "eval_set_done=${eval_set}" | tee -a "${CONTROLLER_LOG}"
  else
    echo "eval_set_failed=${eval_set} exit=${code}" | tee -a "${CONTROLLER_LOG}"
    failed=1
  fi
done
set -e
if test "${failed}" -ne 0; then
  exit 5
fi

"${PY}" "${EVAL_WORKTREE}/experiments/training/sft2/aggregate_mcts_rollouts.py" \
  --output-dir "${EVAL_OUTPUT}" \
  --eval-sets "${EVAL_SETS[@]}" \
  --episodes-per-eval-set "${EPISODES_PER_SET}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"

echo "ALL_OK $(date --iso-8601=seconds)" | tee -a "${CONTROLLER_LOG}"
