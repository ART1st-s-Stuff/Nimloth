#!/usr/bin/env bash
# Collect a fresh two-TP4 rollout and run the PlannerPolicyHead PPO gate on a
# held homogeneous two-node, four-GPU-per-node allocation.
set -euo pipefail

: "${HOLD_JOB:?set HOLD_JOB to the running 4+4 allocation}"
: "${REPO:?set REPO to the committed server worktree}"
: "${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${MODEL:?set MODEL to the immutable behavior checkpoint}"
: "${PLANNER_POLICY_HEAD_CKPT:?set the fresh PlannerPolicyHead artifact}"
: "${RL_CONFIG:?set the 4+4 gate config}"
: "${RUN_OUT:?set a new exclusive output directory}"
: "${WANDB_RUN_NAME:?set the reserved run identity}"

PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth-rl}
SEED_OFFSET=${SEED_OFFSET:-185}
ENV_PORT_BASE=${ENV_PORT_BASE:-9860}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-32940}
GATE_MASTER_PORT=${GATE_MASTER_PORT:-32941}
MINIMUM_STATE_TOKENS=${MINIMUM_STATE_TOKENS:-14000}
PARALLEL_RUNNER=${REPO}/experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh
GATE=${REPO}/experiments/training/rl/gpu_gate_ppo_value_critic.py
ROLLOUT_OUT=${RUN_OUT}/rollouts/iter_0001
TRAJECTORY_JSONL=${ROLLOUT_OUT}/trajectories.jsonl
FRESH_ROLLOUT_MANIFEST=${ROLLOUT_OUT}/fresh_policy_manifest.json
GATE_OUT=${RUN_OUT}/gpu_gate
STAGE_LOG=${RUN_OUT}.controller.log
RENDER_PREFLIGHT_OUT=${RUN_OUT}.render_preflight
RENDER_PREFLIGHT_TIMEOUT=${RENDER_PREFLIGHT_TIMEOUT:-150}

export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export PATH=/cm/shared/apps/slurm/current/bin:${PATH}
source "${REPO}/experiments/training/rl/slurm_allocation.sh"

[[ "$(squeue -h -j "${HOLD_JOB}" -o %T)" == RUNNING ]] || {
  echo "hold ${HOLD_JOB} is not running" >&2
  exit 1
}
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o %N)")
[[ "${#NODES[@]}" == 2 ]] || {
  echo "PlannerPolicyHead gate requires exactly two nodes" >&2
  exit 1
}
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
for node in "${NODES[@]}"; do
  [[ "${GPU_COUNTS[${node}]:-}" == 4 ]] || {
    echo "node ${node} does not have exactly four allocated GPUs" >&2
    exit 1
  }
done

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "runtime commit differs from EXPECTED_COMMIT" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=untracked)" ]] || {
  echo "runtime worktree has tracked changes" >&2
  exit 1
}
[[ -x "${PYTHON}" && -f "${PARALLEL_RUNNER}" && -f "${GATE}" ]] || {
  echo "runtime entrypoint is missing" >&2
  exit 1
}
[[ -f "${MODEL}/config.json" ]] || { echo "model checkpoint is missing" >&2; exit 1; }
[[ -s "${MODEL}/wm_predictor/predictor.pt" ]] || { echo "WM checkpoint is missing" >&2; exit 1; }
[[ -s "${MODEL}/state_proj.pt" ]] || { echo "StateProjector checkpoint is missing" >&2; exit 1; }
[[ -s "${MODEL}/value_head/value_head.pt" ]] || { echo "ValueHead checkpoint is missing" >&2; exit 1; }
[[ -s "${PLANNER_POLICY_HEAD_CKPT}/planner_policy_head.pt" ]] || {
  echo "PlannerPolicyHead checkpoint is missing" >&2
  exit 1
}
[[ ! -e "${RUN_OUT}" && ! -e "${STAGE_LOG}" && ! -e "${RENDER_PREFLIGHT_OUT}" ]] || {
  echo "gate output identity already exists" >&2
  exit 1
}

read -r CONFIG_NODES CONFIG_WORLD CONFIG_GPUS_PER_RANK CONFIG_TOTAL CONFIG_TP CONFIG_EPISODES POLICY_ENABLED POLICY_EPOCHS < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(
    config.distributed.nodes,
    config.distributed.world_size,
    config.distributed.gpus_per_rank,
    config.distributed.total_gpus,
    config.distributed.rollout_tensor_parallel_size,
    config.rl.envs_per_iteration,
    str(config.planner_policy.enabled).lower(),
    config.planner_policy.ppo_epochs,
)
' "${RL_CONFIG}"
)
[[ "${CONFIG_NODES} ${CONFIG_WORLD} ${CONFIG_GPUS_PER_RANK} ${CONFIG_TOTAL} ${CONFIG_TP}" == "2 4 2 8 4" ]] || {
  echo "gate config is not a true 4+4 topology" >&2
  exit 1
}
[[ "${CONFIG_EPISODES} ${POLICY_ENABLED} ${POLICY_EPOCHS}" == "8 true 4" ]] || {
  echo "gate config does not declare eight episodes and four PlannerPolicyHead PPO epochs" >&2
  exit 1
}

mkdir -p "${RUN_OUT%/*}"
CURRENT_STAGE=preflight
record_exit() {
  status=$?
  if (( status != 0 )); then
    printf '%s stage=%s status=failed exit=%s\n' \
      "$(date -Iseconds)" "${CURRENT_STAGE}" "${status}" >> "${STAGE_LOG}"
  fi
}
trap record_exit EXIT
printf '%s stage=preflight status=passed hold=%s nodes=%s commit=%s\n' \
  "$(date -Iseconds)" "${HOLD_JOB}" "${NODES[*]}" "${EXPECTED_COMMIT}" > "${STAGE_LOG}"

CURRENT_STAGE=render_preflight
printf '%s stage=render_preflight status=starting\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
mkdir -p "${RENDER_PREFLIGHT_OUT}"
RENDER_STEP_PIDS=()
for node in "${NODES[@]}"; do
  node_out=${RENDER_PREFLIGHT_OUT}/${node}
  mkdir -p "${node_out}"
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
    -w "${node}" --gres=gpu:1 \
    env REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
      NODE_OUT="${node_out}" RENDER_PREFLIGHT_TIMEOUT="${RENDER_PREFLIGHT_TIMEOUT}" \
    bash -lc '
      set -euo pipefail
      export AI2THOR_HOME_ROOT="${NODE_OUT}/home"
      source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
      export PYTHONPATH="${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${ENV_REPO}/external/le-wm"
      timeout --signal=TERM --kill-after=10s "${RENDER_PREFLIGHT_TIMEOUT}s" \
        "${PYTHON}" -m nimloth.environment.navigation.direct_render_probe \
          --gpu-device 0
    ' >"${node_out}/gpu0.log" 2>&1 &
  RENDER_STEP_PIDS+=("$!")
done
render_status=0
for pid in "${RENDER_STEP_PIDS[@]}"; do
  wait "${pid}" || render_status=$?
done
for node in "${NODES[@]}"; do
  grep -Fq '"status": "AI2THOR_RENDER_OK"' \
    "${RENDER_PREFLIGHT_OUT}/${node}/gpu0.log" || render_status=1
done
if (( render_status != 0 )); then
  tail -n 100 "${RENDER_PREFLIGHT_OUT}"/*/gpu0.log >&2
  exit "${render_status}"
fi
printf '%s stage=render_preflight status=passed nodes=%s\n' \
  "$(date -Iseconds)" "${NODES[*]}" >> "${STAGE_LOG}"

CURRENT_STAGE=rollout
printf '%s stage=rollout status=starting\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
env \
  HOLD_JOB="${HOLD_JOB}" REPO="${REPO}" ENV_REPO="${ENV_REPO}" \
  PYTHON="${PYTHON}" MODEL="${MODEL}" WM_CKPT="${MODEL}" \
  PLANNER_POLICY_HEAD_CKPT="${PLANNER_POLICY_HEAD_CKPT}" \
  REFERENCE_MODEL="${MODEL}" RL_CONFIG="${RL_CONFIG}" RUN_OUT="${RUN_OUT}" \
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  WANDB_MODE_OVERRIDE=disabled ITERATION=1 TOTAL_ITERATIONS=1 \
  RUN_INITIAL_GLOBAL_STEP=0 SEED_OFFSET="${SEED_OFFSET}" \
  ENV_PORT_BASE="${ENV_PORT_BASE}" TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT}" \
  ROLLOUT_WORKERS=2 PIPELINE_MODE=train PIPELINE_PHASE=rollout \
  bash "${PARALLEL_RUNNER}"
[[ -s "${TRAJECTORY_JSONL}" && -s "${FRESH_ROLLOUT_MANIFEST}" ]] || {
  echo "rollout stage returned without merged fresh artifacts" >&2
  exit 1
}
printf '%s stage=rollout status=passed manifest=%s\n' \
  "$(date -Iseconds)" "${FRESH_ROLLOUT_MANIFEST}" >> "${STAGE_LOG}"

COMMON_ENV=(
  PYTHON="${PYTHON}"
  GATE="${GATE}"
  REPO="${REPO}"
  RL_CONFIG="${RL_CONFIG}"
  MODEL="${MODEL}"
  WM_CHECKPOINT="${MODEL}/wm_predictor"
  STATE_PROJ_CHECKPOINT="${MODEL}/state_proj.pt"
  VALUE_HEAD_CHECKPOINT="${MODEL}/value_head"
  PLANNER_POLICY_HEAD_CKPT="${PLANNER_POLICY_HEAD_CKPT}"
  TRAJECTORY_JSONL="${TRAJECTORY_JSONL}"
  FRESH_ROLLOUT_MANIFEST="${FRESH_ROLLOUT_MANIFEST}"
  GATE_OUT="${GATE_OUT}"
  MINIMUM_STATE_TOKENS="${MINIMUM_STATE_TOKENS}"
  HF_HOME=/project/peilab/atst/.cache/huggingface
  TORCH_HOME=/project/peilab/atst/flower/.cache/torch
  TOKENIZERS_PARALLELISM=true
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  WANDB_MODE=disabled
)

CURRENT_STAGE=single_grad
printf '%s stage=single_grad status=starting\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${NODES[0]}" --gres=gpu:1 env "${COMMON_ENV[@]}" bash -lc '
  set -euo pipefail
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
  "${PYTHON}" "${GATE}" \
    --mode single_grad --config "${RL_CONFIG}" --model "${MODEL}" \
    --wm-checkpoint "${WM_CHECKPOINT}" \
    --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
    --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
    --planner-policy-head-checkpoint "${PLANNER_POLICY_HEAD_CKPT}" \
    --trajectory-jsonl "${TRAJECTORY_JSONL}" \
    --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
    --output-dir "${GATE_OUT}" --gpus-per-rank 1 \
    --select-longest-final-transition \
    --minimum-state-tokens "${MINIMUM_STATE_TOKENS}"
' 2>&1 | tee "${RUN_OUT}/single_grad.log"
[[ -s "${GATE_OUT}/single_grad_rank_00.json" ]] || {
  echo "single-gradient gate result is missing" >&2
  exit 1
}
printf '%s stage=single_grad status=passed\n' "$(date -Iseconds)" >> "${STAGE_LOG}"

CURRENT_STAGE=ddp_step
printf '%s stage=ddp_step status=starting world=4 gpus_per_rank=2\n' \
  "$(date -Iseconds)" >> "${STAGE_LOG}"
RDZV_IP=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${NODES[0]}" --gpus=0 hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${RDZV_IP}" ]] || { echo "missing multi-node rendezvous IP" >&2; exit 1; }

DDP_STEP_PIDS=()
rank_offset=0
for node_index in "${!NODES[@]}"; do
  node=${NODES[${node_index}]}
  node_log=${RUN_OUT}/ddp_node_${node_index}_${node}.log
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
    -w "${node}" --gres=gpu:4 env "${COMMON_ENV[@]}" \
      NIMLOTH_RANK_OFFSET="${rank_offset}" MASTER_ADDR="${RDZV_IP}" \
      MASTER_PORT="${GATE_MASTER_PORT}" WORLD_SIZE=4 bash -lc '
      set -euo pipefail
      export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
      pids=()
      for local_rank in 0 1; do
        export RANK=$((NIMLOTH_RANK_OFFSET + local_rank))
        export LOCAL_RANK="${local_rank}"
        "${PYTHON}" "${GATE}" \
          --mode ddp_step --config "${RL_CONFIG}" --model "${MODEL}" \
          --wm-checkpoint "${WM_CHECKPOINT}" \
          --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
          --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
          --planner-policy-head-checkpoint "${PLANNER_POLICY_HEAD_CKPT}" \
          --trajectory-jsonl "${TRAJECTORY_JSONL}" \
          --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
          --output-dir "${GATE_OUT}" --gpus-per-rank 2 \
          --select-longest-final-transition \
          --minimum-state-tokens "${MINIMUM_STATE_TOKENS}" &
        pids+=("$!")
      done
      status=0
      for pid in "${pids[@]}"; do
        wait "${pid}" || status=$?
      done
      exit "${status}"
    ' >"${node_log}" 2>&1 &
  DDP_STEP_PIDS+=("$!")
  rank_offset=$((rank_offset + 2))
done
ddp_status=0
for pid in "${DDP_STEP_PIDS[@]}"; do
  wait "${pid}" || ddp_status=$?
done
if (( ddp_status != 0 )); then
  tail -n 200 "${RUN_OUT}"/ddp_node_*.log >&2
  exit "${ddp_status}"
fi

PYTHONPATH="${REPO}/src" "${PYTHON}" - "${GATE_OUT}" <<'PY' | tee "${RUN_OUT}/gate_summary.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [root / f"ddp_step_rank_{rank:02d}.json" for rank in range(4)]
results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
for rank, result in enumerate(results):
    assert result["status"] == "passed"
    assert result["rank"] == rank
    assert result["world_size"] == 4
    assert result["gpus_per_rank"] == 2
    assert result["ppo_epochs"] == 4
    assert result["qwen_grad_max"] > 0
    assert result["value_grad_max"] > 0
    assert result["planner_policy_grad_max"] > 0
    assert result["value_parameter_delta_max"] > 0
    assert result["planner_policy_parameter_delta_max"] > 0
    assert result["lm_head_grad_is_none"] is True
    assert result["state_projector_grads_absent"] is True
    assert result["vision_grads_absent"] is True
print(json.dumps({"status": "passed", "world_size": 4, "ranks": len(results)}))
PY
printf '%s stage=ddp_step status=passed results=%s\n' \
  "$(date -Iseconds)" "${GATE_OUT}" >> "${STAGE_LOG}"

CURRENT_STAGE=complete
trap - EXIT
printf '%s stage=complete status=all_passed\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
