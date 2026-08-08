#!/usr/bin/env bash
# Resume-safe outer loop: one immutable-policy rollout and one update per iteration.
set -euo pipefail

SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/shared/apps/slurm/current/bin}
SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export SLURM_CONF
export PATH="${SLURM_BIN_DIR}:${PATH}"

HOLD_JOB=${HOLD_JOB:-${SLURM_JOB_ID:-}}
HOLD_JOB=${HOLD_JOB:?set HOLD_JOB or run as a Slurm batch job}
REPO=${REPO:?set REPO to the committed server worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/planner_greedy_h2_full.yaml}
RUN_OUT=${RUN_OUT:?set RUN_OUT to the exclusive formal-run output directory}
FORMAL_OUTPUT_ROOT=${FORMAL_OUTPUT_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/training/rl}
FORMAL_OUTPUT_ROOT=${FORMAL_OUTPUT_ROOT%/}
ITERATION_RUNNER=${ITERATION_RUNNER:-${REPO}/experiments/training/rl/run_vllm_online_ppo_slurm.sh}
EVALUATION_RUNNER=${EVALUATION_RUNNER:-${ITERATION_RUNNER}}
INITIAL_MODEL=${INITIAL_MODEL:?set INITIAL_MODEL to the complete SFT2 HF checkpoint}
INITIAL_WM_CKPT=${INITIAL_WM_CKPT:-${INITIAL_MODEL}}
INITIAL_PLANNER_POLICY_HEAD_CKPT=${INITIAL_PLANNER_POLICY_HEAD_CKPT:-}
INITIAL_RESUME_CHECKPOINT=${INITIAL_RESUME_CHECKPOINT:-}
REFERENCE_MODEL=${REFERENCE_MODEL:-${INITIAL_MODEL}}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth-rl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
RAY_PORT_BASE=${RAY_PORT_BASE:-6400}
ENV_PORT_BASE=${ENV_PORT_BASE:-8600}
TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-29800}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing config: ${RL_CONFIG}" >&2; exit 1; }
[[ -f "${ITERATION_RUNNER}" ]] || {
  echo "missing iteration runner: ${ITERATION_RUNNER}" >&2
  exit 1
}
[[ -f "${EVALUATION_RUNNER}" ]] || {
  echo "missing evaluation runner: ${EVALUATION_RUNNER}" >&2
  exit 1
}
case "${RUN_OUT}" in
  "${FORMAL_OUTPUT_ROOT}"/*) ;;
  *) echo "RUN_OUT is outside the formal RL output root: ${RUN_OUT}" >&2; exit 1 ;;
esac
[[ -f "${INITIAL_MODEL}/config.json" ]] || {
  echo "missing initial model: ${INITIAL_MODEL}" >&2
  exit 1
}

read -r CONFIG_ITERATIONS CONFIG_EPISODES CONFIG_LOG_INTERVAL TRAIN_DATASET_COUNT VALIDATION_ENABLED VALIDATION_EXTERNAL VALIDATION_INTERVAL VALIDATION_ENVS EVAL_DATASETS_CSV PLANNER_POLICY_ENABLED < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(
    config.rl.iterations,
    config.rl.envs_per_iteration,
    config.training.log_interval,
    len(config.rollout.train_datasets),
    str(config.validation.enabled).lower(),
    str(config.validation.external).lower(),
    config.validation.interval,
    config.validation.envs,
    ",".join(config.rollout.eval_datasets),
    str(config.planner_policy.enabled).lower(),
)
' "${RL_CONFIG}"
)
if [[ "${PLANNER_POLICY_ENABLED}" == true ]]; then
  [[ -s "${INITIAL_PLANNER_POLICY_HEAD_CKPT}/planner_policy_head.pt" ]] || {
    echo "planner policy run requires INITIAL_PLANNER_POLICY_HEAD_CKPT" >&2
    exit 1
  }
fi
TOTAL_ITERATIONS=${TOTAL_ITERATIONS:-${CONFIG_ITERATIONS}}
[[ "${TOTAL_ITERATIONS}" == "${CONFIG_ITERATIONS}" ]] || {
  echo "TOTAL_ITERATIONS disagrees with rl.iterations" >&2
  exit 1
}
[[ "${CONFIG_LOG_INTERVAL}" == 1 ]] || {
  echo "resume-safe full runner requires training.log_interval=1" >&2
  exit 1
}
(( TRAIN_DATASET_COUNT > 0 && CONFIG_EPISODES % TRAIN_DATASET_COUNT == 0 )) || {
  echo "training episodes must divide evenly across configured datasets" >&2
  exit 1
}
SEEDS_PER_DATASET_PER_ITERATION=$((CONFIG_EPISODES / TRAIN_DATASET_COUNT))
if [[ "${VALIDATION_EXTERNAL}" == true ]]; then
  [[ "${VALIDATION_ENABLED}" == false ]] || {
    echo "external validation requires built-in validation to be disabled" >&2
    exit 1
  }
  [[ "${VALIDATION_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "external validation interval must be positive" >&2
    exit 1
  }
  (( TOTAL_ITERATIONS % VALIDATION_INTERVAL == 0 )) || {
    echo "external validation interval must divide the configured training horizon" >&2
    exit 1
  }
  [[ "${VALIDATION_ENVS}" == 120 ]] || {
    echo "formal external validation requires 120 episodes" >&2
    exit 1
  }
  [[ "${EVAL_DATASETS_CSV}" == base,common_sense ]] || {
    echo "formal external validation requires held-out base,common_sense" >&2
    exit 1
  }
fi
if [[ -n "${INITIAL_RESUME_CHECKPOINT}" ]]; then
  [[ -s "${INITIAL_RESUME_CHECKPOINT}/rl_state.pt" ]] || {
    echo "initial resume checkpoint has no rl_state.pt: ${INITIAL_RESUME_CHECKPOINT}" >&2
    exit 1
  }
  CHECKPOINT_GLOBAL_STEP=$("${PYTHON}" -c '
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
print(int(state.get("global_step", -1)))
' "${INITIAL_RESUME_CHECKPOINT}/rl_state.pt")
  INITIAL_GLOBAL_STEP=${INITIAL_GLOBAL_STEP:-${CHECKPOINT_GLOBAL_STEP}}
  [[ "${INITIAL_GLOBAL_STEP}" == "${CHECKPOINT_GLOBAL_STEP}" ]] || {
    echo "INITIAL_GLOBAL_STEP disagrees with initial resume checkpoint" >&2
    exit 1
  }
else
  INITIAL_GLOBAL_STEP=${INITIAL_GLOBAL_STEP:-0}
  [[ "${INITIAL_GLOBAL_STEP}" == 0 ]] || {
    echo "nonzero INITIAL_GLOBAL_STEP requires INITIAL_RESUME_CHECKPOINT" >&2
    exit 1
  }
fi
[[ "${INITIAL_GLOBAL_STEP}" =~ ^[0-9]+$ ]] || {
  echo "INITIAL_GLOBAL_STEP must be a non-negative integer" >&2
  exit 1
}
(( INITIAL_GLOBAL_STEP < TOTAL_ITERATIONS )) || {
  echo "initial checkpoint already reached the configured training horizon" >&2
  exit 1
}
FIRST_ITERATION=$((INITIAL_GLOBAL_STEP + 1))
DEFAULT_FIRST_ITERATION_SEED_OFFSET=$((
  INITIAL_GLOBAL_STEP * SEEDS_PER_DATASET_PER_ITERATION + 1
))
FIRST_ITERATION_SEED_OFFSET=${FIRST_ITERATION_SEED_OFFSET:-${DEFAULT_FIRST_ITERATION_SEED_OFFSET}}
[[ "${FIRST_ITERATION_SEED_OFFSET}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FIRST_ITERATION_SEED_OFFSET must be a positive integer" >&2
  exit 1
}

TRAIN_OUT=${RUN_OUT}/train
POLICY_INPUT_ROOT=${TRAIN_OUT}/policy_inputs
PROGRESS_LOG=${RUN_OUT}.iteration_progress.log
RUN_PARENT=${RUN_OUT%/*}
# The progress log is adjacent to RUN_OUT and is written before the iteration
# runner creates RUN_OUT itself. Preserve the empty-output gate while ensuring a
# previously unused date parent can receive that first durable status record.
mkdir -p "${RUN_PARENT}"
CURRENT_ITERATION=0

record_exit() {
  status=$?
  if (( status != 0 )); then
    printf '%s iteration=%s status=controller_failed exit=%s\n' \
      "$(date -Iseconds)" "${CURRENT_ITERATION}" "${status}" >> "${PROGRESS_LOG}"
  fi
}
trap record_exit EXIT

read -r last_completed START_ITERATION discarded_log_rows recovery_archive < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" \
    -m nimloth.training.rl.continuation \
    prepare-run "${RUN_OUT}" "${TOTAL_ITERATIONS}" \
    --initial-global-step "${INITIAL_GLOBAL_STEP}"
)
if [[ "${recovery_archive}" != - ]]; then
  printf '%s iteration=%s status=recovered_interrupted_attempt discarded_log_rows=%s archive=%s\n' \
    "$(date -Iseconds)" "${START_ITERATION}" \
    "${discarded_log_rows}" "${recovery_archive}" >> "${PROGRESS_LOG}"
fi

run_due_evaluation() {
  local evaluated_iteration=$1
  local evaluated_checkpoint=$2
  if [[ "${VALIDATION_EXTERNAL}" != true ]] || (( evaluated_iteration % VALIDATION_INTERVAL != 0 )); then
    return 0
  fi
  local evaluation_tag
  evaluation_tag=$(printf 'iter_%04d' "${evaluated_iteration}")
  local done_flag=${RUN_OUT}/evaluation/${evaluation_tag}/eval_done.flag
  if [[ -s "${done_flag}" ]]; then
    return 0
  fi
  [[ -s "${evaluated_checkpoint}/rl_state.pt" ]] || {
    echo "due evaluation checkpoint is missing: ${evaluated_checkpoint}" >&2
    return 1
  }
  EVALUATION_ENV=(
    HOLD_JOB="${HOLD_JOB}"
    REPO="${REPO}"
    ENV_REPO="${ENV_REPO}"
    PYTHON="${PYTHON}"
    RL_CONFIG="${RL_CONFIG}"
    RUN_OUT="${RUN_OUT}"
    MODEL="${evaluated_checkpoint}"
    WM_CKPT="${evaluated_checkpoint}"
    PLANNER_POLICY_HEAD_CKPT="${evaluated_checkpoint}/planner_policy_head"
    REFERENCE_MODEL="${REFERENCE_MODEL}"
    RESUME_CHECKPOINT="${evaluated_checkpoint}"
    WANDB_PROJECT="${WANDB_PROJECT}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME}"
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}"
    PIPELINE_MODE=eval
    ITERATION="${evaluated_iteration}"
    TOTAL_ITERATIONS="${TOTAL_ITERATIONS}"
    SEED_OFFSET=1
    ENV_PORT="$((ENV_PORT_BASE + evaluated_iteration))"
    TRAIN_MASTER_PORT="$((TRAIN_MASTER_PORT_BASE + evaluated_iteration))"
    RUN_INITIAL_GLOBAL_STEP="${INITIAL_GLOBAL_STEP}"
  )
  if [[ -x "${EVALUATION_RUNNER}" ]]; then
    env "${EVALUATION_ENV[@]}" "${EVALUATION_RUNNER}"
  else
    env "${EVALUATION_ENV[@]}" bash "${EVALUATION_RUNNER}"
  fi
  [[ -s "${done_flag}" ]] || {
    echo "evaluation runner returned without a durable done flag: ${done_flag}" >&2
    return 1
  }
  printf '%s iteration=%s status=evaluated checkpoint=%s\n' \
    "$(date -Iseconds)" "${evaluated_iteration}" "${evaluated_checkpoint}" >> "${PROGRESS_LOG}"
}

# If a controller stopped after committing a due training step but before its
# external eval completed, finish that eval before collecting the next policy
# batch.  The initialization checkpoint predates this run and is not backfilled.
if (( last_completed > INITIAL_GLOBAL_STEP )); then
  run_due_evaluation "${last_completed}" "${TRAIN_OUT}/latest"
fi
if (( last_completed >= TOTAL_ITERATIONS )); then
  echo "formal RL run already completed at global_step=${last_completed}"
  exit 0
fi

for ((iteration=START_ITERATION; iteration<=TOTAL_ITERATIONS; iteration++)); do
  CURRENT_ITERATION=${iteration}
  iteration_tag=$(printf 'iter_%04d' "${iteration}")
  seed_offset=$((
    FIRST_ITERATION_SEED_OFFSET
    + (iteration - FIRST_ITERATION) * SEEDS_PER_DATASET_PER_ITERATION
  ))
  resume_checkpoint=""
  if (( iteration == FIRST_ITERATION )); then
    model=${INITIAL_MODEL}
    wm_checkpoint=${INITIAL_WM_CKPT}
    planner_policy_head_checkpoint=${INITIAL_PLANNER_POLICY_HEAD_CKPT}
    resume_checkpoint=${INITIAL_RESUME_CHECKPOINT}
  else
    snapshot=$(PYTHONPATH="${REPO}/src" "${PYTHON}" \
      -m nimloth.training.rl.continuation \
      prepare-policy "${RUN_OUT}" "${iteration}")
    model=${snapshot}
    wm_checkpoint=${snapshot}
    planner_policy_head_checkpoint=${snapshot}/planner_policy_head
    resume_checkpoint=${snapshot}
  fi

  printf '%s iteration=%s status=starting model=%s seed_offset=%s\n' \
    "$(date -Iseconds)" "${iteration}" "${model}" "${seed_offset}" >> "${PROGRESS_LOG}"

  ITERATION_ENV=(
    HOLD_JOB="${HOLD_JOB}"
    REPO="${REPO}"
    ENV_REPO="${ENV_REPO}"
    PYTHON="${PYTHON}"
    RL_CONFIG="${RL_CONFIG}"
    RUN_OUT="${RUN_OUT}"
    MODEL="${model}"
    WM_CKPT="${wm_checkpoint}"
    PLANNER_POLICY_HEAD_CKPT="${planner_policy_head_checkpoint}"
    REFERENCE_MODEL="${REFERENCE_MODEL}"
    RESUME_CHECKPOINT="${resume_checkpoint}"
    WANDB_PROJECT="${WANDB_PROJECT}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME}"
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}"
    RUN_MODE=full
    ITERATION="${iteration}"
    TOTAL_ITERATIONS="${TOTAL_ITERATIONS}"
    SEED_OFFSET="${seed_offset}"
    RAY_PORT="$((RAY_PORT_BASE + iteration))"
    ENV_PORT="$((ENV_PORT_BASE + iteration))"
    TRAIN_MASTER_PORT="$((TRAIN_MASTER_PORT_BASE + iteration))"
    RAY_HEAD_NODE="${RAY_HEAD_NODE:-}"
    RUN_INITIAL_GLOBAL_STEP="${INITIAL_GLOBAL_STEP}"
  )
  if [[ -x "${ITERATION_RUNNER}" ]]; then
    env "${ITERATION_ENV[@]}" "${ITERATION_RUNNER}"
  else
    env "${ITERATION_ENV[@]}" bash "${ITERATION_RUNNER}"
  fi

  [[ -s "${TRAIN_OUT}/latest/rl_state.pt" ]] || {
    echo "iteration ${iteration} completed without latest checkpoint" >&2
    exit 1
  }
  PYTHONPATH="${REPO}/src" "${PYTHON}" \
    -m nimloth.training.rl.continuation \
    validate-iteration "${RUN_OUT}" "${iteration}" "${TRAIN_OUT}/latest"
  run_due_evaluation "${iteration}" "${TRAIN_OUT}/latest"
  printf '%s iteration=%s status=completed checkpoint=%s\n' \
    "$(date -Iseconds)" "${iteration}" "${TRAIN_OUT}/latest" >> "${PROGRESS_LOG}"

  # Retain the latest pre-update policy for rollback. Older snapshots belong
  # only to this run and are removed after their successor is durably complete.
  if (( iteration > 2 )); then
    prune_tag=$(printf 'iter_%04d' "$((iteration - 1))")
    prune_path=${POLICY_INPUT_ROOT}/${prune_tag}
    if [[ -d "${prune_path}" ]]; then
      case "${prune_path}" in
        "${POLICY_INPUT_ROOT}"/iter_*) ;;
        *) echo "unsafe policy snapshot path: ${prune_path}" >&2; exit 1 ;;
      esac
      rm -rf -- "${prune_path}"
      printf '%s iteration=%s status=pruned_policy_snapshot path=%s\n' \
        "$(date -Iseconds)" "${iteration}" "${prune_path}" >> "${PROGRESS_LOG}"
    fi
  fi
done

trap - EXIT
printf '%s iteration=%s status=all_completed\n' \
  "$(date -Iseconds)" "${TOTAL_ITERATIONS}" >> "${PROGRESS_LOG}"
