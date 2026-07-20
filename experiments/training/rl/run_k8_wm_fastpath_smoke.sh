#!/usr/bin/env bash
# Run inside a held single-node allocation with at least two GPUs.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/rl-kgt1-wm-multiaction}
ENV_REPO=${ENV_REPO:-/project/peilab/atst/nimloth/.worktree/exp-vagen-1action}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
K8_CKPT=${K8_CKPT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/2_ddpsyncfix_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/train/epoch_002}
: "${RUN_OUT:?RUN_OUT must be a fresh output directory}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME must follow the nimloth-rl numeric smoke convention}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID must be reserved before launch}"
ENV_PORT=${ENV_PORT:-8500}

MODEL=${MODEL:-${K8_CKPT}}
WM_CKPT=${WM_CKPT:-${K8_CKPT}}
ROLLOUT_OUT=${RUN_OUT}/rollout
TRAIN_OUT=${RUN_OUT}/train
LOG=${RUN_OUT}/pipeline.log
WANDB_PROJECT_REQUESTED=${WANDB_PROJECT:-nimloth-rl}
WANDB_ENTITY_REQUESTED=${WANDB_ENTITY:-art2nd-hong-kong-university-of-science-and-technology}
WANDB_MODE_REQUESTED=${WANDB_MODE_OVERRIDE:-online}
WANDB_RUN_NAME_REQUESTED=${WANDB_RUN_NAME}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 1; }
for path in \
  "${MODEL}/model.safetensors.index.json" \
  "${MODEL}/tokenizer_config.json" \
  "${WM_CKPT}/training_state.pt" \
  "${WM_CKPT}/state_proj.pt" \
  "${WM_CKPT}/wm_predictor/predictor.pt" \
  "${WM_CKPT}/value_head/value_head.pt"; do
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
# k=8 inject Qwen-first/WM-second fast-path RL end-to-end smoke

- status: running
- Nimloth commit: ${COMMIT}
- env VAGEN commit: ${ENV_COMMIT}
- data: navigation base_train, seeds 1..4 (1200-task training dataset)
- rollout: 4 episodes x at most 2 actions, policy=qwen_wm, fast_path_horizon=2
- behavior semantics: step0 Qwen samples action from GT k=8 state and records exact behavior log-prob; step1 greedy ValueHead acts on recursive WM predicted state
- model/WM/value initialization: ${K8_CKPT} (complete SFT2 epoch2/step2912)
- trainable: Qwen language full parameters, WM predictor, value head
- frozen: Qwen vision tower, state projector
- dynamics training: contiguous recursive rollout_steps=2, decay=1.0
- training: two-rank FSDP, one update followed by one new-process resume update
- output: ${RUN_OUT}
- W&B: entity ${WANDB_ENTITY_REQUESTED}, project ${WANDB_PROJECT_REQUESTED}, run ${WANDB_RUN_NAME_REQUESTED}, id ${WANDB_RUN_ID}, mode ${WANDB_MODE_REQUESTED}
- monitor: k/query/token/projector metadata; behavior provenance; finite WM/value losses; checkpoint/resume; frozen/trainable parameter deltas
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
export WANDB_ENTITY=${WANDB_ENTITY_REQUESTED}
export WANDB_PROJECT=${WANDB_PROJECT_REQUESTED}
export WANDB_MODE=${WANDB_MODE_REQUESTED}
export WANDB_RUN_NAME=${WANDB_RUN_NAME_REQUESTED}
export WANDB_RUN_ID=${WANDB_RUN_ID}
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
  --wm-checkpoint "${WM_CKPT}" \
  --policy qwen_wm \
  --fast-path-horizon 2 \
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
  --max-pixels 100352 \
  2>&1 | tee -a "${LOG}"

cleanup
ENV_PID=""

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
TRAIN_ARGS=(
  -m nimloth.training.rl.cli
  --config "${REPO}/configs/training/rl/k8_wm_fastpath_smoke.yaml"
  --model "${MODEL}"
  --llm-tune full
  --vision-tune freeze
  --wm-checkpoint "${WM_CKPT}/wm_predictor"
  --state-proj-checkpoint "${WM_CKPT}/state_proj.pt"
  --value-head-checkpoint "${WM_CKPT}/value_head"
  --use-jsonl-rollout
  --jsonl-sources "${ROLLOUT_OUT}/trajectories.jsonl"
  --attn-implementation sdpa
  --max-pixels 100352
  --experiment-name k8-wm-fastpath-e2e-smoke
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

run_root = Path("${RUN_OUT}")
rollout_root = Path("${ROLLOUT_OUT}")
train_root = Path("${TRAIN_OUT}")
source = Path("${K8_CKPT}")
final = train_root / "final"
state = torch.load(final / "rl_state.pt", map_location="cpu", weights_only=False)
required = [
    rollout_root / "trajectories.jsonl",
    train_root / "train_step_log.csv",
    train_root / "best" / "rl_state.pt",
    final / "rl_state.pt",
    final / "wm_predictor" / "predictor.pt",
    final / "value_head" / "value_head.pt",
    final / "state_proj.pt",
    final / "config.json",
    final / "model.safetensors.index.json",
    final / "optimizer_rank_00000.pt",
    final / "optimizer_rank_00001.pt",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")

records = [json.loads(line) for line in (rollout_root / "trajectories.jsonl").read_text().splitlines() if line]
if len(records) != 4 or sum(len(record["action_indices"]) for record in records) != 8:
    raise SystemExit(f"unexpected rollout cardinality: records={len(records)}")
for record in records:
    steps = len(record["action_indices"])
    expected = {
        "latent_token_count": 8,
        "latent_query_mode": "inject",
        "rollout_policy": "qwen_wm",
        "fast_path_horizon": 2,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise SystemExit(f"rollout {record.get('id')} {key}={record.get(key)!r}, expected {value!r}")
    if steps != 2 or len(record["image_paths"]) != 3:
        raise SystemExit(f"incomplete rollout {record.get('id')}: actions={steps}, images={len(record['image_paths'])}")
    if record["policy_sources"] != ["qwen", "wm_value"]:
        raise SystemExit(f"wrong policy provenance: {record['policy_sources']}")
    if record["state_sources"] != ["qwen_gt", "wm_predicted"]:
        raise SystemExit(f"wrong state provenance: {record['state_sources']}")
    if record["fast_path_steps"] != [0, 1]:
        raise SystemExit(f"wrong fast-path steps: {record['fast_path_steps']}")
    log_probs = record["action_log_probs"]
    if (
        not isinstance(log_probs[0], list)
        or len(log_probs[0]) != 8
        or log_probs[0][record["action_indices"][0]] is None
        or log_probs[1] is not None
        or record.get("action_log_prob_semantics") != "sampling_distribution_v1"
    ):
        raise SystemExit("hybrid behavior log-prob ownership is invalid")

expected_state = {
    "iteration": 2,
    "global_step": 2,
    "latent_token_count": 8,
    "latent_query_mode": "inject",
    "qwen_hidden_dim": 2048,
    "state_proj_input_dim": 16384,
    "rollout_policy": "qwen_wm",
    "fast_path_horizon": 2,
    "predictor_rollout_steps": 2,
    "predictor_rollout_loss_decay": 1.0,
}
for key, value in expected_state.items():
    if state.get(key) != value:
        raise SystemExit(f"final state {key}={state.get(key)!r}, expected {value!r}")
query_ids = state.get("latent_query_token_ids")
if not isinstance(query_ids, list) or len(query_ids) != 8 or len(set(query_ids)) != 8:
    raise SystemExit(f"invalid latent query token IDs: {query_ids}")

rows = list(csv.DictReader((train_root / "train_step_log.csv").open()))
if [int(row["global_step"]) for row in rows] != [1, 2]:
    raise SystemExit(f"unexpected training steps: {rows}")
finite_keys = (
    "wm_mse", "wm_mse_h1", "wm_mse_hlast", "value_loss", "total_loss",
    "actor_loss", "entropy",
)
for row in rows:
    bad = {key: row[key] for key in finite_keys if not row[key] or not math.isfinite(float(row[key]))}
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

def load_torch_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)

def assert_same_state(name, before, after):
    if before.keys() != after.keys():
        raise SystemExit(f"{name} keys changed")
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    if changed:
        raise SystemExit(f"frozen {name} changed: {changed[:3]}")

def assert_changed_state(name, before, after):
    if before.keys() != after.keys():
        raise SystemExit(f"{name} keys changed")
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    if not changed:
        raise SystemExit(f"trainable {name} did not change")
    for key, tensor in after.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise SystemExit(f"non-finite {name} tensor: {key}")
    return changed

assert_same_state(
    "state_projector",
    load_torch_state(source / "state_proj.pt"),
    load_torch_state(final / "state_proj.pt"),
)
predictor_changed = assert_changed_state(
    "wm_predictor",
    load_torch_state(source / "wm_predictor" / "predictor.pt"),
    load_torch_state(final / "wm_predictor" / "predictor.pt"),
)
value_changed = assert_changed_state(
    "value_head",
    load_torch_state(source / "value_head" / "value_head.pt"),
    load_torch_state(final / "value_head" / "value_head.pt"),
)

def hf_tensor(root, key):
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    with safe_open(root / idx["weight_map"][key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)

source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
final_keys = set(index["weight_map"])
common = [key for key in source_index if key in final_keys]
language_key = next((
    key for key in common
    if "model.layers.0" in key and key.endswith("q_proj.weight")
), None)
vision_key = next((
    key for key in common
    if key.startswith("visual.") and key.endswith("weight")
), None)
if language_key is None or vision_key is None:
    raise SystemExit("could not find Qwen language/vision probe tensors")
if torch.equal(hf_tensor(source, language_key), hf_tensor(final, language_key)):
    raise SystemExit(f"trainable Qwen language tensor did not change: {language_key}")
if not torch.equal(hf_tensor(source, vision_key), hf_tensor(final, vision_key)):
    raise SystemExit(f"frozen Qwen vision tensor changed: {vision_key}")

print(json.dumps({
    "status": "ALL_OK",
    "iteration": 2,
    "global_step": 2,
    "rollout_records": len(records),
    "rollout_transitions": 8,
    "finite_metric_rows": len(rows),
    "model_shards": len(set(index["weight_map"].values())),
    "predictor_changed_tensors": len(predictor_changed),
    "value_changed_tensors": len(value_changed),
    "trainable_qwen_language_probe": language_key,
    "frozen_qwen_vision_probe": vision_key,
}))
PY

sed -i 's/- status: running/- status: completed/' "${RUN_OUT}/README.md"
echo "=== RL e2e smoke ALL_OK $(date -Iseconds) ===" | tee -a "${LOG}"
