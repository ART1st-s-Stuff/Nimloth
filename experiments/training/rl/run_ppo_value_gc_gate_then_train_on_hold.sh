#!/usr/bin/env bash
# Stage one fresh two-TP4 rollout, gate long-prefix critic memory, then train.
# This controller must run as one attached step inside an existing 1x8 hold.
set -euo pipefail

: "${HOLD_JOB:?set HOLD_JOB to the running 1x8 allocation}"
: "${REPO:?set REPO to the committed server worktree}"
: "${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to the committed policy input}"
: "${RL_CONFIG:?set RL_CONFIG to the one-iteration resume config}"
: "${RUN_OUT:?set RUN_OUT to a new exclusive output directory}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"

PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth-rl}
ITERATION=${ITERATION:-16}
TOTAL_ITERATIONS=${TOTAL_ITERATIONS:-16}
RUN_INITIAL_GLOBAL_STEP=${RUN_INITIAL_GLOBAL_STEP:-15}
SEED_OFFSET=${SEED_OFFSET:-121}
ENV_PORT=${ENV_PORT:-9730}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-32830}
MINIMUM_STATE_TOKENS=${MINIMUM_STATE_TOKENS:-14000}
PARALLEL_RUNNER=${REPO}/experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh
GPU_GATE=${REPO}/experiments/training/rl/gpu_gate_ppo_value_critic.slurm
ITERATION_TAG=$(printf 'iter_%04d' "${ITERATION}")
ROLLOUT_OUT=${RUN_OUT}/rollouts/${ITERATION_TAG}
TRAJECTORY_JSONL=${ROLLOUT_OUT}/trajectories.jsonl
FRESH_ROLLOUT_MANIFEST=${ROLLOUT_OUT}/fresh_policy_manifest.json
GATE_OUT=${RUN_OUT}/gpu_gate_longprefix
STAGE_LOG=${RUN_OUT}.staged_controller.log

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${PARALLEL_RUNNER}" ]] || { echo "missing parallel runner" >&2; exit 1; }
[[ -f "${GPU_GATE}" ]] || { echo "missing GPU gate" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config" >&2; exit 1; }
[[ -s "${RESUME_CHECKPOINT}/rl_state.pt" ]] || {
  echo "resume checkpoint has no rl_state.pt" >&2
  exit 1
}
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "runtime commit differs from EXPECTED_COMMIT" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=dirty)" ]] || {
  echo "runtime worktree has tracked changes" >&2
  exit 1
}
[[ ! -e "${RUN_OUT}" ]] || { echo "RUN_OUT already exists" >&2; exit 1; }
[[ ! -e "${STAGE_LOG}" ]] || { echo "stage log already exists" >&2; exit 1; }

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

printf '%s stage=preflight status=passed commit=%s checkpoint=%s\n' \
  "$(date -Iseconds)" "${EXPECTED_COMMIT}" "${RESUME_CHECKPOINT}" > "${STAGE_LOG}"

run_parallel_phase() {
  local phase=$1
  local wandb_mode=$2
  env \
    HOLD_JOB="${HOLD_JOB}" REPO="${REPO}" ENV_REPO="${ENV_REPO}" \
    PYTHON="${PYTHON}" MODEL="${RESUME_CHECKPOINT}" \
    WM_CKPT="${RESUME_CHECKPOINT}" REFERENCE_MODEL="${RESUME_CHECKPOINT}" \
    RL_CONFIG="${RL_CONFIG}" RUN_OUT="${RUN_OUT}" \
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${wandb_mode}" \
    ITERATION="${ITERATION}" TOTAL_ITERATIONS="${TOTAL_ITERATIONS}" \
    RUN_INITIAL_GLOBAL_STEP="${RUN_INITIAL_GLOBAL_STEP}" \
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" SEED_OFFSET="${SEED_OFFSET}" \
    ENV_PORT="${ENV_PORT}" TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT}" \
    ROLLOUT_WORKERS=2 PIPELINE_MODE=train PIPELINE_PHASE="${phase}" \
    bash "${PARALLEL_RUNNER}"
}

CURRENT_STAGE=rollout
printf '%s stage=rollout status=starting\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
run_parallel_phase rollout disabled
[[ -s "${TRAJECTORY_JSONL}" && -s "${FRESH_ROLLOUT_MANIFEST}" ]] || {
  echo "rollout stage returned without merged fresh artifacts" >&2
  exit 1
}
printf '%s stage=rollout status=passed manifest=%s\n' \
  "$(date -Iseconds)" "${FRESH_ROLLOUT_MANIFEST}" >> "${STAGE_LOG}"

CURRENT_STAGE=gpu_gate
printf '%s stage=gpu_gate status=starting min_state_tokens=%s\n' \
  "$(date -Iseconds)" "${MINIMUM_STATE_TOKENS}" >> "${STAGE_LOG}"
env \
  RUNTIME_REPO="${REPO}" EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  OUTPUT_DIR="${GATE_OUT}" SFT2_CHECKPOINT="${RESUME_CHECKPOINT}" \
  TRAJECTORY_JSONL="${TRAJECTORY_JSONL}" \
  FRESH_ROLLOUT_MANIFEST="${FRESH_ROLLOUT_MANIFEST}" \
  MINIMUM_STATE_TOKENS="${MINIMUM_STATE_TOKENS}" \
  bash "${GPU_GATE}"
for result in \
  "${GATE_OUT}/single_grad_rank_00.json" \
  "${GATE_OUT}/ddp_step_rank_00.json" \
  "${GATE_OUT}/ddp_step_rank_01.json"; do
  [[ -s "${result}" ]] || { echo "GPU gate result is missing: ${result}" >&2; exit 1; }
done
printf '%s stage=gpu_gate status=passed output=%s\n' \
  "$(date -Iseconds)" "${GATE_OUT}" >> "${STAGE_LOG}"

CURRENT_STAGE=train
printf '%s stage=train status=starting\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
run_parallel_phase train online
printf '%s stage=train status=passed checkpoint=%s\n' \
  "$(date -Iseconds)" "${RUN_OUT}/train/final" >> "${STAGE_LOG}"

CURRENT_STAGE=complete
trap - EXIT
printf '%s stage=complete status=all_passed\n' "$(date -Iseconds)" >> "${STAGE_LOG}"
