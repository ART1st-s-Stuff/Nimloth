#!/usr/bin/env bash
# Run inside a held single-node allocation with at least two GPUs.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/feat-rl-validation}
ENV_REPO=${ENV_REPO:-/project/peilab/atst/nimloth/.worktree/exp-vagen-1action}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
SFT2_ROOT=${SFT2_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2}
RUN_OUT=${RUN_OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-07-11/post_fsdp_fix_e2e_smoke_retry1}
ENV_PORT=${ENV_PORT:-8500}

MODEL=${SFT2_ROOT}/export_best_hf
WM_CKPT=${SFT2_ROOT}/best
ROLLOUT_OUT=${RUN_OUT}/rollout
TRAIN_OUT=${RUN_OUT}/train
LOG=${RUN_OUT}/pipeline.log

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
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
- data: navigation base_train, seeds 1..4 (training scenes/tasks)
- rollout: 4 episodes, at most 2 actions each, Nimloth action-token policy
- initialization: ${SFT2_ROOT}
- trainable: Qwen language model full parameters, WM predictor, value head
- frozen: vision tower, state projector
- training: two-rank FSDP, JSONL collector, one step followed by one resume step
- output: ${RUN_OUT}
- monitor: rollout schema/count, wm_mse, value_loss, actor_loss, entropy, checkpoint and resume global_step
EOF

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled

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
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
"${PYTHON}" "${REPO}/experiments/training/rl/rollout_env.py" \
  --model "${MODEL}" \
  --env-url "${ENV_URL}" \
  --output-dir "${ROLLOUT_OUT}" \
  --num-episodes 4 \
  --max-steps 2 \
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
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
TRAIN_ARGS=(
  -m nimloth.training.rl.cli
  --config "${REPO}/configs/training/rl/e2e_smoke.yaml"
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
import json
from pathlib import Path
import torch
root = Path("${TRAIN_OUT}")
state = torch.load(root / "final" / "rl_state.pt", map_location="cpu", weights_only=False)
required = [
    root / "train_step_log.csv",
    root / "best" / "rl_state.pt",
    root / "final" / "rl_state.pt",
    root / "final" / "wm_predictor" / "predictor.pt",
    root / "final" / "value_head" / "value_head.pt",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
if state.get("iteration") != 2 or state.get("global_step") != 2:
    raise SystemExit(f"resume did not reach iteration/global_step 2: {state}")
print(json.dumps({"status": "ALL_OK", "iteration": 2, "global_step": 2}))
PY

sed -i 's/- status: running/- status: completed/' "${RUN_OUT}/README.md"
echo "=== RL e2e smoke ALL_OK $(date -Iseconds) ===" | tee -a "${LOG}"
