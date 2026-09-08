#!/usr/bin/env bash
set -euo pipefail

if [[ "${NIMLOTH_ALLOCATION_STEP:-0}" != 1 ]]; then
  : "${ALLOCATION_JOB_ID:?set ALLOCATION_JOB_ID to the existing dgx-56 allocation}"
  exec srun --jobid="${ALLOCATION_JOB_ID}" --nodes=1 --ntasks=1 \
    --gres=gpu:8 --cpus-per-task=96 --mem=480G \
    --export=ALL,NIMLOTH_ALLOCATION_STEP=1 bash "$0"
fi

: "${REPO:?set REPO to the clean committed unified-SFT worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact Nimloth commit}"
: "${SOURCE_EPOCH:?set SOURCE_EPOCH to the completed SFT1 epoch checkpoint}"
: "${RUN_ROOT:?set a new persistent RUN_ROOT for this continuation segment}"
: "${SLURM_JOB_ID:?the controller must run inside the allocation step}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
ROOT=/project/peilab/atst/nimloth
SOURCE_CHECKPOINT=${ROOT}/experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface
DATA_ROOT=${ROOT}/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c
TRAIN_JSONL=${DATA_ROOT}/train_success.jsonl
VAL_JSONL=${DATA_ROOT}/val_all.jsonl
OUT=${RUN_ROOT}/stage1_continuation
MERGED=${RUN_ROOT}/stage1_for_sft2
RUNTIME_ROOT=/tmp/nimloth-sft1-cont-${SLURM_JOB_ID}

[[ -x "${PYTHON}" ]] || { echo "missing interpreter: ${PYTHON}" >&2; exit 2; }
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)" ]] || { echo "source is dirty" >&2; exit 2; }
[[ -f "${SOURCE_EPOCH}/COMMITTED" && -f "${SOURCE_EPOCH}/training_state.pt" ]] || {
  echo "SOURCE_EPOCH is not a committed epoch checkpoint" >&2; exit 2;
}
[[ "${SLURM_JOB_NODELIST:-}" == "dgx-56" ]] || { echo "allocation is not fixed to dgx-56" >&2; exit 2; }
IFS=',' read -r -a GPU_TOKENS <<<"${CUDA_VISIBLE_DEVICES:-}"
(( ${#GPU_TOKENS[@]} == 8 )) || { echo "launcher requires exactly eight GPUs" >&2; exit 2; }

export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:${PATH}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NIMLOTH_PIPELINE_RUNTIME_ROOT=${RUNTIME_ROOT}
if [[ -e "${RUN_ROOT}" && ! -f "${RUN_ROOT}/run_identity.json" ]]; then
  echo "existing RUN_ROOT has no durable identity" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}" "${RUNTIME_ROOT}"
exec > >(env -u NIMLOTH_PIPELINE_RUNTIME_ROOT tee -a "${RUN_ROOT}/controller.log") 2>&1
record_early_exit() {
  status=$?
  trap - EXIT
  "${PYTHON}" - "${RUN_ROOT}/controller_exit.json" "${status}" <<'PY' || true
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2]), "interrupted": False, "automatic_requeue": False}, indent=2) + "\n")
PY
  exit "${status}"
}
trap record_early_exit EXIT

"${PYTHON}" - "${SOURCE_EPOCH}" "${RUN_ROOT}/source_preflight.json" <<'PY'
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open

source, output = map(Path, sys.argv[1:])
state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
identity = state.get("identity")
if not isinstance(identity, dict):
    raise TypeError("source epoch has no strict training identity")
required = {"optimizer", "step", "epoch", "best_val", "world_size"}
if required - state.keys() or state.get("training_stage") != "format" or not state.get("lora"):
    raise ValueError("source epoch is not a complete SFT1 LoRA training checkpoint")
source_world = int(state["world_size"])
source_batch = int(identity.get("batch_size", -1))
source_accum = int(identity.get("grad_accum", -1))
if (source_world, source_batch, source_accum) != (4, 1, 8):
    raise ValueError("source epoch is not the reviewed world4/batch1/GA8 segment")
if source_world * source_batch * source_accum != 8 * 1 * 4:
    raise ValueError("source and target effective batch sizes differ")
adapter = source / "adapter_model.safetensors"
if not adapter.is_file() or not (source / "adapter_config.json").is_file():
    raise FileNotFoundError("source epoch has no complete LoRA adapter")
with safe_open(adapter, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
for module in ("embed_tokens", "lm_head"):
    matches = [key for key in keys if key.endswith(f"{module}.modules_to_save.weight")]
    if len(matches) != 1:
        raise ValueError(f"source adapter requires one trained {module} tensor, got {matches}")
payload = {
    "source": str(source.resolve()), "epoch": int(state["epoch"]),
    "step": int(state["step"]), "source_world_size": source_world,
    "source_grad_accum": source_accum, "target_world_size": 8,
    "target_grad_accum": 4, "effective_batch_size": 32,
    "adapter_tensor_count": len(keys),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

"${PYTHON}" - "${RUN_ROOT}/run_identity.json" "${EXPECTED_COMMIT}" "${SOURCE_EPOCH}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "schema": "sft1_scheduler_continuation_v1",
    "commit": sys.argv[2],
    "source_epoch": str(Path(sys.argv[3]).resolve()),
    "world_size": 8,
    "additional_epochs": 19,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
}
if path.exists():
    if json.loads(path.read_text()) != expected:
        raise ValueError("existing continuation run identity mismatch")
else:
    fd, temporary = tempfile.mkstemp(prefix=".run_identity.", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
PY

TRAIN_PID=
forward_preemption() {
  touch "${RUNTIME_ROOT}/preemption_notice"
  # This allocation is no-requeue. Stop the whole owned torchrun process group;
  # a later explicitly launched segment resume uses the latest periodic point.
  echo '{"status":"interrupted","requeue":false,"recovery":"latest_periodic_optimizer_checkpoint"}' >&2
  if [[ -n "${TRAIN_PID}" ]]; then kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true; fi
}
handle_term() {
  if [[ -f "${RUNTIME_ROOT}/preemption_notice" ]]; then
    return
  fi
  if [[ -n "${TRAIN_PID}" ]]; then kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true; fi
  exit 143
}
finish() {
  status=$?
  trap - EXIT
  "${PYTHON}" - "${RUN_ROOT}/controller_exit.json" "${status}" "$([[ -f "${RUNTIME_ROOT}/preemption_notice" ]] && echo true || echo false)" <<'PY' || true
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2]), "interrupted": sys.argv[3] == "true", "automatic_requeue": False}, indent=2) + "\n")
PY
  rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap forward_preemption USR1
trap handle_term TERM INT
trap finish EXIT

PORT=$((22000 + SLURM_JOB_ID % 8000))
if ss -ltnH "sport = :${PORT}" | grep -q .; then
  echo "training port is already occupied: ${PORT}" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 --master_port="${PORT}" \
  -m experiments.training.sft.evaluation.pipeline_contract rank-map \
  --world-size 8 --output "${RUN_ROOT}/rank_map.json"
set +e
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 --master_port="${PORT}" \
  -m nimloth.training.sft.stage1 \
  --model "${SOURCE_CHECKPOINT}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUT}" --epochs 19 --batch-size 1 --grad-accum 4 \
  --lr 2e-4 --embedding-lr 5e-4 --max-length 12000 --max-pixels 100352 \
  --latent-token-count 1 --latent-query-mode generate --lora --lora-r 64 --lora-alpha 128 \
  --no-cache --no-wandb --resume --resume-save-steps 5 --keep-resume-checkpoints 2 \
  --new-scheduler-segment-from "${SOURCE_EPOCH}" \
  --early-stopping-patience 3 --early-stopping-min-delta 0.001 \
  >>"${RUN_ROOT}/stage1_train.log" 2>&1 &
TRAIN_PID=$!
wait "${TRAIN_PID}"
status=$?
set -e
if (( status != 0 )); then exit "${status}"; fi

"${PYTHON}" - "${SOURCE_EPOCH}" "${OUT}" "${RUN_ROOT}/selected_adapter_path.txt" <<'PY'
import sys
from pathlib import Path

source, output, selection = map(Path, sys.argv[1:])
completed = [path for path in output.glob("epoch_*") if (path / "COMMITTED").is_file()]
best = output / "best"
if best.is_symlink():
    selected = best.resolve(strict=True)
    if selected not in [path.resolve() for path in completed]:
        raise ValueError("best pointer does not reference a committed continuation epoch")
else:
    # No continuation epoch beat the source baseline by min_delta. This is a
    # valid early-stop outcome, so retain the reviewed source epoch for SFT2.
    selected = source
selection.write_text(str(selected.resolve()) + "\n")
PY
read -r SELECTED <"${RUN_ROOT}/selected_adapter_path.txt"
if [[ ! -e "${MERGED}" ]]; then
  temporary=${RUN_ROOT}/.stage1_for_sft2.${SLURM_RESTART_COUNT:-0}
  [[ ! -e "${temporary}" ]] || { echo "stale merge temporary: ${temporary}" >&2; exit 2; }
  "${PYTHON}" -m nimloth.training.sft.stage1.checkpoint_export \
    --base-model "${SOURCE_CHECKPOINT}" --adapter-dir "${SELECTED}" --out-dir "${temporary}"
  mv "${temporary}" "${MERGED}"
fi
"${PYTHON}" - "${SELECTED}" "${MERGED}" "${RUN_ROOT}/selected_for_sft2.json" <<'PY'
import json, sys
from pathlib import Path
import torch

source, merged, output = map(Path, sys.argv[1:])
state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
if state.get("training_stage") != "format" or not state.get("lora"):
    raise ValueError("selected SFT2 initializer is not an SFT1 LoRA checkpoint")
if not (merged / "config.json").is_file():
    raise FileNotFoundError("merged SFT2 initializer has no config.json")
output.write_text(json.dumps({"adapter": str(source), "merged": str(merged), "epoch": state["epoch"], "val_loss": state["best_val"]}, indent=2) + "\n")
PY
