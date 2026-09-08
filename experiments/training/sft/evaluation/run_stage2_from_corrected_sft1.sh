#!/usr/bin/env bash
# Train and export only query/DINO SFT2 from the selected corrected SFT1 model.
set -euo pipefail

WORLD_SIZE=${WORLD_SIZE:-4}
EXPECTED_NODE=${EXPECTED_NODE:-dgx-22}
[[ "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( WORLD_SIZE <= 32 && 32 % WORLD_SIZE == 0 )) || {
  echo "WORLD_SIZE must divide effective batch 32" >&2; exit 2;
}
GRAD_ACCUM=$((32 / WORLD_SIZE))

if [[ "${NIMLOTH_ALLOCATION_STEP:-0}" != 1 ]]; then
  : "${ALLOCATION_JOB_ID:?set ALLOCATION_JOB_ID to the existing allocation}"
  exec srun --jobid="${ALLOCATION_JOB_ID}" --nodes=1 --ntasks=1 \
    --gres="gpu:${WORLD_SIZE}" --cpus-per-task="$((WORLD_SIZE * 12))" --mem="$((WORLD_SIZE * 60))G" \
    --export=ALL,NIMLOTH_ALLOCATION_STEP=1 bash "$0"
fi

: "${REPO:?set REPO to the clean committed unified-SFT worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact Nimloth commit}"
: "${SFT1_RUN_ROOT:?set SFT1_RUN_ROOT to the corrected SFT1 continuation output}"
: "${RUN_ROOT:?set RUN_ROOT to a new persistent Stage2 output directory}"
: "${SLURM_JOB_ID:?the controller must run inside the allocation step}"
: "${ALLOCATION_JOB_ID:?set ALLOCATION_JOB_ID to the existing allocation}"
[[ "${SLURM_JOB_ID}" == "${ALLOCATION_JOB_ID}" ]] || { echo "allocation mismatch" >&2; exit 2; }

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
ROOT=/project/peilab/atst/nimloth
DATA_ROOT=${ROOT}/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c
TRAIN_JSONL=${DATA_ROOT}/train_success.jsonl
VAL_JSONL=${DATA_ROOT}/val_all.jsonl
DINO_CACHE=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe
SELECTED=${SFT1_RUN_ROOT}/selected_for_sft2.json
STAGE1_MERGED=${SFT1_RUN_ROOT}/stage1_for_sft2
OUT=${RUN_ROOT}/stage2
MERGED=${RUN_ROOT}/stage2_for_evaluation

[[ -x "${PYTHON}" ]] || { echo "missing interpreter: ${PYTHON}" >&2; exit 2; }
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)" ]] || { echo "source is dirty" >&2; exit 2; }
[[ "${SLURM_JOB_NODELIST:-}" == "${EXPECTED_NODE}" ]] || { echo "allocation node mismatch" >&2; exit 2; }
IFS=',' read -r -a GPU_TOKENS <<<"${CUDA_VISIBLE_DEVICES:-}"
(( ${#GPU_TOKENS[@]} == WORLD_SIZE )) || { echo "allocated GPU count differs from WORLD_SIZE" >&2; exit 2; }
[[ -f "${SELECTED}" && -f "${STAGE1_MERGED}/config.json" ]] || {
  echo "selected corrected SFT1 initializer is incomplete" >&2; exit 2;
}
[[ -f "${DINO_CACHE}/manifest.json" ]] || { echo "DINO cache manifest is missing" >&2; exit 2; }

export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:${PATH}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -e "${RUN_ROOT}" && ! -f "${RUN_ROOT}/run_identity.json" ]]; then
  echo "existing RUN_ROOT has no durable identity" >&2; exit 2
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/controller.log") 2>&1

"${PYTHON}" - "${SELECTED}" "${STAGE1_MERGED}" "${RUN_ROOT}/stage2_input.json" "${WORLD_SIZE}" "${EXPECTED_NODE}" <<'PY'
import json, sys
from pathlib import Path

selected, merged, output = map(Path, sys.argv[1:4])
world_size, node = int(sys.argv[4]), sys.argv[5]
payload = json.loads(selected.read_text())
if Path(payload.get("merged", "")).resolve() != merged.resolve():
    raise ValueError("selected_for_sft2 does not name the supplied merged model")
if int(payload.get("epoch", -1)) != 3:
    raise ValueError("Stage2 requires the selected corrected SFT1 epoch 3 initializer")
config = json.loads((merged / "config.json").read_text())
if config.get("nimloth_training_stage") != "format":
    raise ValueError("Stage2 initializer must be a merged format-stage model")
output.write_text(json.dumps({
    "schema": "sft2_from_corrected_sft1_v1",
    "selected": str(selected.resolve()), "model": str(merged.resolve()),
    "stage1_epoch": 3, "stage1_val_loss": payload.get("val_loss"),
    "world_size": world_size, "grad_accum": 32 // world_size, "node": node,
    "lr": 1e-6, "embedding_lr": 5e-6,
}, indent=2, sort_keys=True) + "\n")
PY

"${PYTHON}" - "${RUN_ROOT}/run_identity.json" "${EXPECTED_COMMIT}" "${STAGE1_MERGED}" "${WORLD_SIZE}" "${EXPECTED_NODE}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "schema": "sft2_from_corrected_sft1_v1", "commit": sys.argv[2],
    "stage1_merged": str(Path(sys.argv[3]).resolve()), "world_size": int(sys.argv[4]),
    "node": sys.argv[5], "grad_accum": 32 // int(sys.argv[4]),
    "epochs": 1, "lr": 1e-6, "embedding_lr": 5e-6,
    "latent_token_count": 16, "latent_query_mode": "inject", "grid_size": 4,
}
if path.exists():
    if json.loads(path.read_text()) != expected:
        raise ValueError("existing Stage2 run identity mismatch")
else:
    fd, temporary = tempfile.mkstemp(prefix=".run_identity.", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
PY

finish() {
  status=$?
  trap - EXIT
  "${PYTHON}" - "${RUN_ROOT}/controller_exit.json" "${status}" <<'PY' || true
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2])}, indent=2) + "\n")
PY
  exit "${status}"
}
trap finish EXIT

PORT=$((22000 + SLURM_JOB_ID % 8000))
if ss -ltnH "sport = :${PORT}" | grep -q .; then
  echo "training port is already occupied: ${PORT}" >&2; exit 2
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" --master_port="${PORT}" \
  -m experiments.training.sft.evaluation.pipeline_contract rank-map \
  --world-size "${WORLD_SIZE}" --output "${RUN_ROOT}/rank_map.json"

"${PYTHON}" -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" --master_port="${PORT}" \
  -m nimloth.training.sft.stage2 \
  --model "${STAGE1_MERGED}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUT}" --dino-cache-root "${DINO_CACHE}" \
  --epochs 1 --batch-size 1 --grad-accum "${GRAD_ACCUM}" --max-length 12000 --max-pixels 100352 \
  --lr 1e-6 --embedding-lr 5e-6 --latent-token-count 16 --latent-query-mode inject \
  --grid-size 4 --lora --lora-r 64 --lora-alpha 128 --no-wandb --resume \
  --resume-save-steps 5 --keep-resume-checkpoints 2 >>"${RUN_ROOT}/stage2_train.log" 2>&1

STAGE2_EPOCH=${OUT}/epoch_001
"${PYTHON}" -m experiments.training.sft.evaluation.pipeline_contract validate-stage-checkpoint \
  --stage query --checkpoint "${STAGE2_EPOCH}"
if [[ ! -e "${MERGED}" ]]; then
  temporary=${RUN_ROOT}/.stage2_for_evaluation.${SLURM_RESTART_COUNT:-0}
  [[ ! -e "${temporary}" ]] || { echo "stale Stage2 merge temporary" >&2; exit 2; }
  "${PYTHON}" -m nimloth.training.sft.stage1.checkpoint_export \
    --base-model "${STAGE1_MERGED}" --adapter-dir "${STAGE2_EPOCH}" --out-dir "${temporary}"
  mv "${temporary}" "${MERGED}"
fi
"${PYTHON}" -m experiments.training.sft.evaluation.pipeline_contract validate-merged \
  --stage query --adapter-dir "${STAGE2_EPOCH}" --merged-dir "${MERGED}" \
  --output "${RUN_ROOT}/stage2_merge.json"
