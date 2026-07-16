#!/usr/bin/env bash
#SBATCH --job-name=rl-dyn-smoke
#SBATCH --time=00:45:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO to the clean launch worktree}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${MODEL:?must set MODEL to the k=1 inject SFT2 checkpoint}"
: "${WANDB_RUN_ID:?must reserve and persist WANDB_RUN_ID before submit}"
: "${WANDB_RUN_NAME:?must set WANDB_RUN_NAME}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
TRAIN_OUT="${RUN_ROOT}/train"
mkdir -p "${TRAIN_OUT}"
ENV_JOB_ID=""
SUCCESS=0
cleanup() {
  touch "${RUN_ROOT}/trainer_done.flag"
  if [[ -n "${ENV_JOB_ID}" && "${SUCCESS}" != 1 ]]; then
    "${SCANCEL}" "${ENV_JOB_ID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

ACTUAL_COMMIT=$(git -C "${REPO}" rev-parse HEAD)
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit mismatch expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] || {
  echo "launch worktree has tracked changes" >&2
  git -C "${REPO}" status --short >&2
  exit 1
}

if [[ "${AUTO_SUBMIT_ENV:-0}" == 1 ]]; then
  : "${ENV_REPO:?AUTO_SUBMIT_ENV requires ENV_REPO}"
  export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
  ENV_JOB_ID=$("${SBATCH}" --parsable --nodelist="${ENV_NODE:-dgx-51}" \
    --cpus-per-task="${ENV_CPUS:-8}" \
    --output="${RUN_ROOT}/env_slurm_%j.out" \
    --error="${RUN_ROOT}/env_slurm_%j.err" \
    --export="ALL,RUN_ROOT=${RUN_ROOT},ENV_REPO=${ENV_REPO}" \
    "${REPO}/experiments/training/rl/dynamic_env_server.slurm")
  printf '%s\n' "${ENV_JOB_ID}" >"${RUN_ROOT}/env_job_id.txt"
  echo "submitted env job ${ENV_JOB_ID} on ${ENV_NODE:-dgx-51}"
fi

for _ in $(seq 1 300); do
  [[ -s "${RUN_ROOT}/env_url.txt" ]] && break
  sleep 2
done
[[ -s "${RUN_ROOT}/env_url.txt" ]] || { echo "missing env URL" >&2; exit 1; }
ENV_URL=$(<"${RUN_ROOT}/env_url.txt")
curl --fail --silent "${ENV_URL}/health" >/dev/null

for path in \
  "${MODEL}/config.json" \
  "${MODEL}/state_proj.pt" \
  "${MODEL}/wm_predictor/predictor.pt" \
  "${MODEL}/value_head/value_head.pt"; do
  [[ -f "${path}" ]] || { echo "missing checkpoint artifact: ${path}" >&2; exit 1; }
done

source /project/peilab/atst/flower/.env
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:${PATH}
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/le-wm:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export HOME=/project/peilab/atst/nimloth/.home
export WANDB_DIR="${REPO}/.cache/wandb"
export WANDB_PROJECT=nimloth-rl
export WANDB_RUN_ID WANDB_RUN_NAME
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

cd "${REPO}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 -- \
  -m nimloth.training.rl.cli \
  --config configs/training/rl/dynamic_fsdp_smoke.yaml \
  --model "${MODEL}" \
  --llm-tune full \
  --vision-tune freeze \
  --gradient-checkpointing \
  --wm-checkpoint "${MODEL}/wm_predictor" \
  --state-proj-checkpoint "${MODEL}/state_proj.pt" \
  --value-head-checkpoint "${MODEL}/value_head" \
  --env-url "${ENV_URL}" \
  --attn-implementation sdpa \
  --max-pixels 3136 \
  --experiment-name "${WANDB_RUN_NAME}" \
  --output-dir "${TRAIN_OUT}" \
  2>&1 | tee "${RUN_ROOT}/trainer.log"

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import csv, json, math, sys, torch
from pathlib import Path
from safetensors import safe_open
root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "train/train_step_log.csv").open()))
assert len(rows) == 1, len(rows)
for key in ("wm_mse", "value_loss", "total_loss", "actor_loss", "entropy"):
    assert math.isfinite(float(rows[0][key])), (key, rows[0][key])
records = [json.loads(line) for line in (root / "train/rollouts/iter_0001/trajectories.jsonl").open()]
assert len(records) == 2, len(records)
for record in records:
    assert record["split"] == "train"
    assert len(record["action_indices"]) == 1
    assert len(record["image_paths"]) == 2
    assert len(record["action_log_probs"]) == 1
    assert len(record["action_log_probs"][0]) == 8
final = root / "train/final"
index = json.loads((final / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 2, shards
for shard in shards:
    with safe_open(final / shard, framework="pt", device="cpu") as handle:
        assert all(0 not in handle.get_slice(key).get_shape() for key in handle.keys())
assert (final / "optimizer_rank_00000.pt").stat().st_size > 0
assert (final / "optimizer_rank_00001.pt").stat().st_size > 0
state = torch.load(final / "rl_state.pt", map_location="cpu", weights_only=False)
assert state["iteration"] == 1 and state["global_step"] == 1
assert state["optimizer_world_size"] == 2
assert state["rollout_protocol"]["mode"] == "dynamic_env"
assert state["rollout_protocol"]["env_timeout"] == 600
assert state["rollout_protocol"]["control_backend"] == "gloo"
print(json.dumps({"status": "ALL_OK", "records": len(records), "shards": len(shards)}))
PY

echo "=== dynamic FSDP smoke ALL_OK $(date --iso-8601=seconds) ==="
SUCCESS=1
touch "${RUN_ROOT}/trainer_done.flag"
if [[ -n "${ENV_JOB_ID}" ]]; then
  for _ in $(seq 1 12); do
    [[ -z "$("${SQUEUE}" -h -j "${ENV_JOB_ID}" 2>/dev/null)" ]] && break
    sleep 5
  done
fi
