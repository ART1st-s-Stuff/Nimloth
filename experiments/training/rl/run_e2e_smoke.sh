#!/usr/bin/env bash
# Run inside a held single-node allocation with at least two GPUs.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/feat-rl-validation}
ENV_REPO=${ENV_REPO:-/project/peilab/atst/nimloth/.worktree/exp-vagen-1action}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
SFT2_ROOT=${SFT2_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2}
RUN_OUT=${RUN_OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-07-11/post_fsdp_fix_e2e_smoke_retry1}
ENV_PORT=${ENV_PORT:-8500}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/e2e_smoke.yaml}
NUM_EPISODES=${NUM_EPISODES:-4}
MAX_STEPS=${MAX_STEPS:-2}

MODEL=${MODEL:-${SFT2_ROOT}/export_best_hf}
WM_CKPT=${WM_CKPT:-${SFT2_ROOT}/best}
ROLLOUT_OUT=${RUN_OUT}/rollout
TRAIN_OUT=${RUN_OUT}/train
LOG=${RUN_OUT}/pipeline.log
WANDB_PROJECT_REQUESTED=${WANDB_PROJECT:-nimloth-rl}
WANDB_MODE_REQUESTED=${WANDB_MODE_OVERRIDE:-disabled}
WANDB_RUN_NAME_REQUESTED=${WANDB_RUN_NAME:-1_smoke_k1ep2_rl_e2e4x2_fsdp2_iter2}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config: ${RL_CONFIG}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 1; }
for path in "${WM_CKPT}/state_proj.pt" "${WM_CKPT}/wm_predictor/predictor.pt" "${WM_CKPT}/value_head/value_head.pt"; do
  [[ -f "${path}" ]] || { echo "missing checkpoint file: ${path}" >&2; exit 1; }
done
[[ -f "${ENV_REPO}/external/VAGEN/vagen/env/navigation/datasets/base_train.json" ]] || {
  echo "ENV_REPO does not contain the verified base_train dataset" >&2
  exit 1
}
if [[ -e "${RUN_OUT}" ]] && find "${RUN_OUT}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to reuse non-empty output directory: ${RUN_OUT}" >&2
  exit 1
fi
mkdir -p "${RUN_OUT}" "${ROLLOUT_OUT}" "${TRAIN_OUT}"

COMMIT=$(git -C "${REPO}" rev-parse HEAD)
ENV_COMMIT=$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)
cat > "${RUN_OUT}/README.md" <<EOF
# Post-FSDP-fix RL end-to-end smoke

- status: running
- Nimloth commit: ${COMMIT}
- env VAGEN commit: ${ENV_COMMIT}
- data: navigation base_train, seeds 1..${NUM_EPISODES} (training scenes/tasks)
- rollout: ${NUM_EPISODES} episodes, at most ${MAX_STEPS} actions each, Nimloth action-token policy
- RL config: ${RL_CONFIG}
- model initialization: ${MODEL}
- WM/value initialization: ${WM_CKPT}
- trainable: Qwen language model full parameters, WM predictor, value head
- frozen: vision tower, state projector
- training: two-rank FSDP, JSONL collector, one step followed by one resume step
- output: ${RUN_OUT}
- W&B: project ${WANDB_PROJECT_REQUESTED}, run ${WANDB_RUN_NAME_REQUESTED}, mode ${WANDB_MODE_REQUESTED}
- monitor: rollout schema/count, finite losses, actor update, checkpoint and resume global_step
EOF

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
fi
export WANDB_PROJECT=${WANDB_PROJECT_REQUESTED}
export WANDB_MODE=${WANDB_MODE_REQUESTED}
export WANDB_RUN_NAME=${WANDB_RUN_NAME_REQUESTED}
export WANDB_DIR=${WANDB_DIR:-${REPO}/.cache/wandb}

VISIBLE=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a GPUS <<< "${VISIBLE}"
if (( ${#GPUS[@]} < 2 )); then
  echo "need at least two allocated GPUs; CUDA_VISIBLE_DEVICES=${VISIBLE}" >&2
  exit 1
fi
ENV_GPU=${GPUS[0]}
ROLLOUT_GPU=${GPUS[1]}
TRAIN_GPUS=${GPUS[0]},${GPUS[1]}
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || HEAD_IP=$(hostname -I | awk '{print $1}')
ENV_URL=http://${HEAD_IP}:${ENV_PORT}
ENV_LOG=${RUN_OUT}/env_server.log
ENV_PID=""

cleanup() {
  if [[ -n "${ENV_PID}" ]]; then
    kill "${ENV_PID}" 2>/dev/null || true
    wait "${ENV_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

{
  echo "=== RL e2e smoke start $(date -Iseconds) ==="
  echo "node=$(hostname) allocation_gpus=${VISIBLE} env_url=${ENV_URL}"
  echo "nimloth=${COMMIT} env_vagen=${ENV_COMMIT}"
} | tee "${LOG}"

(
  export CUDA_VISIBLE_DEVICES=${ENV_GPU}
  export PYTHONPATH=${ENV_REPO}/external/VAGEN
  source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
  cd "${ENV_REPO}/external/VAGEN"
  exec "${PYTHON}" -m vagen.server.server \
    server.host=0.0.0.0 \
    server.port=${ENV_PORT} \
    use_state_reward=False \
    navigation.devices=[0] \
    navigation.max_workers=1
) >"${ENV_LOG}" 2>&1 &
ENV_PID=$!

for i in $(seq 1 300); do
  if curl -fsS "${ENV_URL}/health" >/dev/null 2>&1; then
    echo "env ready after ${i}s" | tee -a "${LOG}"
    break
  fi
  if ! kill -0 "${ENV_PID}" 2>/dev/null; then
    echo "env server exited before health check" | tee -a "${LOG}"
    tail -100 "${ENV_LOG}" | tee -a "${LOG}"
    exit 1
  fi
  sleep 1
done
curl -fsS "${ENV_URL}/health" | tee -a "${LOG}"

export CUDA_VISIBLE_DEVICES=${ROLLOUT_GPU}
export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
"${PYTHON}" "${REPO}/experiments/training/rl/rollout_env.py" \
  --model "${MODEL}" \
  --env-url "${ENV_URL}" \
  --output-dir "${ROLLOUT_OUT}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --eval-set base_train \
  --split train \
  --seed-offset 1 \
  --temperature 0.7 \
  --top-p 0.95 \
  --attn-implementation sdpa \
  --max-pixels 3136 \
  2>&1 | tee -a "${LOG}"

cleanup
ENV_PID=""

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
TRAIN_ARGS=(
  -m nimloth.training.rl.cli
  --config "${RL_CONFIG}"
  --model "${MODEL}"
  --llm-tune full
  --vision-tune freeze
  --gradient-checkpointing
  --wm-checkpoint "${WM_CKPT}/wm_predictor"
  --state-proj-checkpoint "${WM_CKPT}/state_proj.pt"
  --value-head-checkpoint "${WM_CKPT}/value_head"
  --use-jsonl-rollout
  --jsonl-sources "${ROLLOUT_OUT}/trajectories.jsonl"
  --attn-implementation sdpa
  --max-pixels 3136
  --experiment-name post-fsdp-fix-e2e-smoke
  --output-dir "${TRAIN_OUT}"
)

"${PYTHON}" -m torch.distributed.run --nproc_per_node=2 -- "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"

# A second process must load best/ and perform iteration 2, proving resume works.
"${PYTHON}" -m torch.distributed.run --nproc_per_node=2 -- \
  "${TRAIN_ARGS[@]}" --resume --rl-iterations 2 \
  2>&1 | tee -a "${LOG}"

"${PYTHON}" - <<PY | tee -a "${LOG}"
import csv
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
root = Path("${TRAIN_OUT}")
final = root / "final"
state = torch.load(final / "rl_state.pt", map_location="cpu", weights_only=False)
required = [
    root / "train_step_log.csv",
    root / "best" / "rl_state.pt",
    final / "rl_state.pt",
    final / "wm_predictor" / "predictor.pt",
    final / "value_head" / "value_head.pt",
    final / "model.safetensors.index.json",
    final / "optimizer_rank_00000.pt",
    final / "optimizer_rank_00001.pt",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
if state.get("iteration") != 2 or state.get("global_step") != 2:
    raise SystemExit(f"resume did not reach iteration/global_step 2: {state}")
rows = list(csv.DictReader((root / "train_step_log.csv").open()))
if [int(row["global_step"]) for row in rows] != [1, 2]:
    raise SystemExit(f"unexpected training steps: {rows}")
finite_keys = ("wm_mse", "value_loss", "total_loss", "actor_loss", "entropy")
for row in rows:
    bad = {key: row[key] for key in finite_keys if not math.isfinite(float(row[key]))}
    if bad:
        raise SystemExit(f"non-finite RL metrics at step {row['global_step']}: {bad}")
index = json.loads((final / "model.safetensors.index.json").read_text())
for shard_name in set(index["weight_map"].values()):
    shard = final / shard_name
    if not shard.is_file() or shard.stat().st_size == 0:
        raise SystemExit(f"missing or empty model shard: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        empty_shapes = [key for key in handle.keys() if 0 in handle.get_slice(key).get_shape()]
    if empty_shapes:
        raise SystemExit(f"empty FSDP tensors in {shard}: {empty_shapes[:3]}")
print(json.dumps({
    "status": "ALL_OK",
    "iteration": 2,
    "global_step": 2,
    "finite_metric_rows": len(rows),
    "model_shards": len(set(index["weight_map"].values())),
}))
PY

sed -i 's/- status: running/- status: completed/' "${RUN_OUT}/README.md"
echo "=== RL e2e smoke ALL_OK $(date -Iseconds) ===" | tee -a "${LOG}"
