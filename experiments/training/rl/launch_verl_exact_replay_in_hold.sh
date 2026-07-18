#!/usr/bin/env bash
set -euo pipefail

: "${HOLD_JOB:?set HOLD_JOB}"
: "${REPO:?set REPO}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${MODEL:?set MODEL}"
: "${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${WANDB_PROJECT:?set WANDB_PROJECT}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"
: "${WANDB_RUN_ID:?set WANDB_RUN_ID}"

SLURM=/cm/shared/apps/slurm/current/bin
JOB_LINE=$(${SLURM}/scontrol show job "${HOLD_JOB}" -o)
[[ "${JOB_LINE}" == *"JobState=RUNNING"* ]] || {
  echo "hold job is not RUNNING: ${JOB_LINE}" >&2
  exit 2
}
[[ "${JOB_LINE}" == *"Partition=normal"* ]] || {
  echo "hold job is not on normal partition" >&2
  exit 2
}
mapfile -t NODES < <(${SLURM}/scontrol show hostnames "$(${SLURM}/squeue -j "${HOLD_JOB}" -h -o '%N')")
[[ ${#NODES[@]} -eq 1 ]] || {
  echo "exact replay normal8 gate requires one allocated node, got ${NODES[*]}" >&2
  exit 2
}
NODE=${NODES[0]}
MASTER_ADDR=$(${SLURM}/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${NODE}" \
  bash -lc "hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}'")
[[ -n "${MASTER_ADDR}" ]] || {
  echo "failed to resolve 10.23 master address" >&2
  exit 2
}
MASTER_PORT=$(${SLURM}/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${NODE}" \
  /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
mkdir -p "${OUTPUT_DIR}"
LOG=${OUTPUT_DIR}/worker_gate.log
{
  echo "launch_time=$(date --iso-8601=seconds)"
  echo "hold_job=${HOLD_JOB} node=${NODE} world_size=8"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "commit=${EXPECTED_COMMIT}"
} | tee -a "${LOG}"

set +e
${SLURM}/srun \
  --jobid="${HOLD_JOB}" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus=8 \
  --cpus-per-task=96 \
  --kill-on-bad-exit=1 \
  --export="ALL,REPO=${REPO},EXPECTED_COMMIT=${EXPECTED_COMMIT},MODEL=${MODEL},TRAJECTORY_JSONL=${TRAJECTORY_JSONL},TRAJECTORY_INDEX=${TRAJECTORY_INDEX:-0},OUTPUT_DIR=${OUTPUT_DIR},MAX_TOKEN_LENGTH=${MAX_TOKEN_LENGTH:-8192},RESUME_CHECKPOINT_ROOT=${RESUME_CHECKPOINT_ROOT:-},RESUME_RESULT=${RESUME_RESULT:-},SAVE_GLOBAL_STEP=${SAVE_GLOBAL_STEP:-1},WM_AUX_MECHANICS=${WM_AUX_MECHANICS:-0},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_NAME=${WANDB_RUN_NAME},WANDB_RUN_ID=${WANDB_RUN_ID},MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},WORLD_SIZE=8" \
  bash "${REPO}/experiments/training/rl/run_verl_exact_replay_torchrun.sh" \
  2>&1 | tee -a "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e
if [[ ${STATUS} -ne 0 ]]; then
  echo "VERL_EXACT_REPLAY_FAILED status=${STATUS}" | tee -a "${LOG}"
  exit "${STATUS}"
fi

/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 - "${OUTPUT_DIR}" "${SAVE_GLOBAL_STEP:-1}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
global_step = int(sys.argv[2])
result = json.loads((root / "result.json").read_text())
assert result["status"] == "VERL_EXACT_REPLAY_ALL_OK", result
assert result["save_global_step"] == global_step, result
for role in ("actor", "critic"):
    checkpoint = root / "checkpoints" / f"global_step_{global_step}" / role
    for prefix in ("model", "optim", "extra_state"):
        files = list(checkpoint.glob(f"{prefix}_world_size_8_rank_*.pt"))
        assert len(files) == 8, (role, prefix, len(files))
if result.get("wm_aux_after") is not None:
    assert (root / "checkpoints" / f"global_step_{global_step}" / "actor" / "nimloth_wm_aux.pt").is_file()
print("VERL_EXACT_REPLAY_ARTIFACTS_OK")
PY
