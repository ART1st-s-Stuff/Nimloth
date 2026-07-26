#!/usr/bin/env bash
# Resume-safe outer loop: one immutable-policy rollout and one update per iteration.
set -euo pipefail

SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/shared/apps/slurm/current/bin}
SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export SLURM_CONF
export PATH="${SLURM_BIN_DIR}:${PATH}"

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB to one running allocation}
REPO=${REPO:?set REPO to the committed server worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/planner_greedy_h2_full.yaml}
RUN_OUT=${RUN_OUT:?set RUN_OUT to the exclusive formal-run output directory}
FORMAL_OUTPUT_ROOT=${FORMAL_OUTPUT_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/training/rl}
FORMAL_OUTPUT_ROOT=${FORMAL_OUTPUT_ROOT%/}
ITERATION_RUNNER=${ITERATION_RUNNER:-${REPO}/experiments/training/rl/run_vllm_online_ppo_slurm.sh}
INITIAL_MODEL=${INITIAL_MODEL:?set INITIAL_MODEL to the complete SFT2 HF checkpoint}
INITIAL_WM_CKPT=${INITIAL_WM_CKPT:-${INITIAL_MODEL}}
REFERENCE_MODEL=${REFERENCE_MODEL:-${INITIAL_MODEL}}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth-rl}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
RAY_PORT_BASE=${RAY_PORT_BASE:-6400}
ENV_PORT_BASE=${ENV_PORT_BASE:-8600}
TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-29800}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing config: ${RL_CONFIG}" >&2; exit 1; }
[[ -x "${ITERATION_RUNNER}" ]] || {
  echo "missing iteration runner: ${ITERATION_RUNNER}" >&2
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

read -r CONFIG_ITERATIONS CONFIG_EPISODES < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(config.rl.iterations, config.rl.envs_per_iteration)
' "${RL_CONFIG}"
)
TOTAL_ITERATIONS=${TOTAL_ITERATIONS:-${CONFIG_ITERATIONS}}
[[ "${TOTAL_ITERATIONS}" == "${CONFIG_ITERATIONS}" ]] || {
  echo "TOTAL_ITERATIONS disagrees with rl.iterations" >&2
  exit 1
}

TRAIN_OUT=${RUN_OUT}/train
POLICY_INPUT_ROOT=${TRAIN_OUT}/policy_inputs
PROGRESS_LOG=${RUN_OUT}.iteration_progress.log
mkdir -p "${FORMAL_OUTPUT_ROOT}"
CURRENT_ITERATION=0

record_exit() {
  status=$?
  if (( status != 0 )); then
    printf '%s iteration=%s status=controller_failed exit=%s\n' \
      "$(date -Iseconds)" "${CURRENT_ITERATION}" "${status}" >> "${PROGRESS_LOG}"
  fi
}
trap record_exit EXIT

last_completed=0
if [[ -s "${TRAIN_OUT}/train_step_log.csv" ]]; then
  last_completed=$("${PYTHON}" -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print(int(rows[-1]["global_step"]) if rows else 0)
' "${TRAIN_OUT}/train_step_log.csv")
fi

if (( last_completed >= TOTAL_ITERATIONS )); then
  [[ -s "${TRAIN_OUT}/final/rl_state.pt" ]] || {
    echo "step log reached target but final checkpoint is missing" >&2
    exit 1
  }
  echo "formal RL run already completed at global_step=${last_completed}"
  exit 0
fi

START_ITERATION=$((last_completed + 1))
if (( START_ITERATION > 1 )) && [[ ! -s "${TRAIN_OUT}/latest/rl_state.pt" ]]; then
  echo "latest checkpoint is missing after global_step=${last_completed}; inspect policy_inputs before recovery" >&2
  exit 1
fi

for ((iteration=START_ITERATION; iteration<=TOTAL_ITERATIONS; iteration++)); do
  CURRENT_ITERATION=${iteration}
  iteration_tag=$(printf 'iter_%04d' "${iteration}")
  seed_offset=$(( (iteration - 1) * CONFIG_EPISODES + 1 ))
  resume_checkpoint=""
  if (( iteration == 1 )); then
    model=${INITIAL_MODEL}
    wm_checkpoint=${INITIAL_WM_CKPT}
  else
    mkdir -p "${POLICY_INPUT_ROOT}"
    snapshot=${POLICY_INPUT_ROOT}/${iteration_tag}
    [[ ! -e "${snapshot}" ]] || {
      echo "policy input already exists for ${iteration_tag}: ${snapshot}" >&2
      exit 1
    }
    mv "${TRAIN_OUT}/latest" "${snapshot}"
    model=${snapshot}
    wm_checkpoint=${snapshot}
    resume_checkpoint=${snapshot}

    previous_manifest=${RUN_OUT}/rollouts/$(printf 'iter_%04d' "$((iteration - 1))")/fresh_policy_manifest.json
    previous_consumption=${previous_manifest}.consumption.json
    "${PYTHON}" -c '
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, old_path, new_path = map(Path, sys.argv[1:])
payload = json.loads(path.read_text(encoding="utf-8"))
if Path(payload.get("checkpoint_path", "")) != old_path:
    raise SystemExit(f"consumption checkpoint relocation mismatch: {payload}")
payload["checkpoint_path"] = str(new_path)
payload["checkpoint_relocated_at"] = datetime.now(timezone.utc).isoformat()
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
' "${previous_consumption}" "${TRAIN_OUT}/latest" "${snapshot}"
  fi

  printf '%s iteration=%s status=starting model=%s seed_offset=%s\n' \
    "$(date -Iseconds)" "${iteration}" "${model}" "${seed_offset}" >> "${PROGRESS_LOG}"

  env \
    HOLD_JOB="${HOLD_JOB}" \
    REPO="${REPO}" \
    ENV_REPO="${ENV_REPO}" \
    PYTHON="${PYTHON}" \
    RL_CONFIG="${RL_CONFIG}" \
    RUN_OUT="${RUN_OUT}" \
    MODEL="${model}" \
    WM_CKPT="${wm_checkpoint}" \
    REFERENCE_MODEL="${REFERENCE_MODEL}" \
    RESUME_CHECKPOINT="${resume_checkpoint}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}" \
    RUN_MODE=full \
    ITERATION="${iteration}" \
    TOTAL_ITERATIONS="${TOTAL_ITERATIONS}" \
    SEED_OFFSET="${seed_offset}" \
    RAY_PORT="$((RAY_PORT_BASE + iteration))" \
    ENV_PORT="$((ENV_PORT_BASE + iteration))" \
    TRAIN_MASTER_PORT="$((TRAIN_MASTER_PORT_BASE + iteration))" \
    RAY_HEAD_NODE="${RAY_HEAD_NODE:-}" \
    "${ITERATION_RUNNER}"

  [[ -s "${TRAIN_OUT}/latest/rl_state.pt" ]] || {
    echo "iteration ${iteration} completed without latest checkpoint" >&2
    exit 1
  }
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
