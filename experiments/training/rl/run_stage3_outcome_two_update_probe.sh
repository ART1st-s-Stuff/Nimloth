#!/usr/bin/env bash
set -euo pipefail

# Concrete a100/non-Slurm controller: each iteration creates a fresh rollout from
# the exact checkpoint it then updates. The whole probe has one 15-minute budget.
REPO=${REPO:?set REPO to the committed remote worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:?set PYTHON to the verified runtime interpreter}
SOURCE_CKPT=${SOURCE_CKPT:?set SOURCE_CKPT to the selected Stage3 epoch checkpoint}
RUN_OUT=${RUN_OUT:?set RUN_OUT to a new exclusive output directory}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/stage3_outcome_best_two_update_probe.yaml}
PIPELINE=${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh
MIN_FREE_GIB=${MIN_FREE_GIB:-110}
TOTAL_TIMEOUT_SECONDS=${TOTAL_TIMEOUT_SECONDS:-900}
export ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

[[ "${TOTAL_TIMEOUT_SECONDS}" == 900 ]] || {
  echo "the approved short probe has a fixed 900-second total deadline" >&2
  exit 2
}
[[ -x "${PYTHON}" && -x "${PIPELINE}" && -f "${RL_CONFIG}" ]]
read -r ACTOR_ENABLED CREDIT_ASSIGNMENT < <(
  "${PYTHON}" - "${RL_CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path

config = yaml.safe_load(Path(sys.argv[1]).read_text())
actor = config.get("actor", {})
print(str(actor.get("enabled") is True).lower(), actor.get("credit_assignment", ""))
PY
)
if [[ "${ACTOR_ENABLED}" == true && "${CREDIT_ASSIGNMENT}" != turn ]]; then
  echo "direct Qwen PPO probe requires actor.credit_assignment=turn" >&2
  exit 2
fi
[[ ! -e "${RUN_OUT}" ]] || {
  echo "refusing to reuse probe output: ${RUN_OUT}" >&2
  exit 2
}
for relative in config.json state_proj.pt wm_predictor/predictor.pt \
  value_head/value_head.pt outcome_head.pt training_state.pt; do
  [[ -f "${SOURCE_CKPT}/${relative}" ]] || {
    echo "incomplete Stage3 source checkpoint: ${relative}" >&2
    exit 2
  }
done

# latest is a full recovery checkpoint. During the second atomic save, the first
# latest and its replacement coexist. 110 GiB covers two observed ~44 GiB trees
# plus a 22 GiB safety margin; do not weaken recovery by disabling the save.
AVAILABLE_BYTES=$(df -PB1 "$(dirname "${RUN_OUT}")" | awk 'NR==2 {print $4}')
REQUIRED_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
  echo "insufficient checkpoint peak space: available=${AVAILABLE_BYTES}, required=${REQUIRED_BYTES}" >&2
  exit 3
fi

STARTED=$(date +%s)
run_iteration() {
  local iteration=$1 model=$2 initial_step=$3 resume=${4:-}
  local elapsed remaining
  elapsed=$(( $(date +%s) - STARTED ))
  remaining=$(( TOTAL_TIMEOUT_SECONDS - elapsed ))
  (( remaining > 0 )) || { echo "probe deadline expired before iteration ${iteration}" >&2; return 124; }
  local resume_env=()
  [[ -z "${resume}" ]] || resume_env+=(RESUME_CHECKPOINT="${resume}")
  timeout --signal=TERM --kill-after=30s "${remaining}s" \
    env REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
      MODEL="${model}" WM_CKPT="${model}" OUTCOME_HEAD_CKPT="${model}/outcome_head.pt" \
      RL_CONFIG="${RL_CONFIG}" RUN_OUT="${RUN_OUT}" RUN_MODE=smoke \
      ITERATION="${iteration}" RUN_INITIAL_GLOBAL_STEP="${initial_step}" \
      TOTAL_ITERATIONS=2 WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
      WANDB_MODE_OVERRIDE=offline "${resume_env[@]}" \
      "${PIPELINE}"
}

run_iteration 1 "${SOURCE_CKPT}" 0
LATEST=${RUN_OUT}/train/latest
[[ -f "${LATEST}/rl_state.pt" && -f "${LATEST}/outcome_head.pt" ]]
run_iteration 2 "${LATEST}" 0 "${LATEST}"

[[ -f "${RUN_OUT}/train/final/rl_state.pt" ]]
"${PYTHON}" - "${RUN_OUT}" "${ACTOR_ENABLED}" "${CREDIT_ASSIGNMENT}" <<'PY'
import csv, json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
actor_enabled = sys.argv[2] == "true"
if actor_enabled and sys.argv[3] != "turn":
    raise SystemExit("direct Qwen PPO probe requires actor.credit_assignment=turn")
rows = list(csv.DictReader((root / "train/train_step_log.csv").open()))
if len(rows) != 2 or [int(row["global_step"]) for row in rows] != [1, 2]:
    raise SystemExit(f"expected exactly two updates: {rows}")
required = ("wm_mse", "dino_grid_mse", "value_loss", "outcome_bce", "total_loss",
            "outcome_count", "loss_finite", "gradient_finite", "optimizer_updates")
for row in rows:
    actor_required = (
        "actor_loss", "policy_tokens", "mean_ratio", "clip_fraction",
        "mean_advantage", "mean_abs_advantage", "entropy",
        "gradient_l2", "gradient_parameter_count",
        "gradient_qwen_l2", "gradient_qwen_parameter_count",
    ) if actor_enabled else ()
    for key in (*required, *actor_required):
        if key not in row or not math.isfinite(float(row[key])):
            raise SystemExit(f"invalid {key}: {row}")
    if float(row["outcome_count"]) <= 0:
        raise SystemExit(f"update has no Outcome labels: {row}")
    if float(row["loss_finite"]) != 1 or float(row["gradient_finite"]) != 1:
        raise SystemExit(f"non-finite update evidence: {row}")
    if float(row["optimizer_updates"]) != 1:
        raise SystemExit(f"missing optimizer update evidence: {row}")
    if actor_enabled:
        if float(row["policy_tokens"]) <= 0:
            raise SystemExit(f"direct PPO update has no policy tokens: {row}")
        if float(row["mean_abs_advantage"]) <= 0:
            raise SystemExit(f"direct PPO update has no nonzero advantages: {row}")
        if float(row["gradient_parameter_count"]) <= 0:
            raise SystemExit(f"direct PPO update has no gradient parameters: {row}")
        if float(row["gradient_l2"]) <= 0:
            raise SystemExit(f"direct PPO update has zero gradient norm: {row}")
        if float(row["gradient_qwen_parameter_count"]) <= 0:
            raise SystemExit(f"direct PPO update has no Qwen gradient parameters: {row}")
        if float(row["gradient_qwen_l2"]) <= 0:
            raise SystemExit(f"direct PPO update has zero Qwen gradient norm: {row}")
(root / "two_update_probe_complete.json").write_text(
    json.dumps({
        "status": "ALL_OK",
        "updates": 2,
        "actor_enabled": actor_enabled,
        "final": str(root / "train/final"),
    }, indent=2) + "\n"
)
PY
