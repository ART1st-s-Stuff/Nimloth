#!/usr/bin/env bash
# Run under: timeout --signal=TERM --kill-after=60s 6h bash launch.sh COMMIT RUN_ID
set -Eeuo pipefail
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_]+$ ]]
EXPECTED_COMMIT=$1
RUN_ID=$2
ROOT=/mnt/nimloth/.worktree/sft1-rollout2000
PY=/mnt/nimloth/venv/bin/python3
DATA=/mnt/nimloth/outputs/datasets/sft1-vagen-step60/20260910T093222Z_batch1_original_validation_k16
RUN=/mnt/nimloth/outputs/experiments/sft1-rollout2000/$RUN_ID
cd "$ROOT"
[[ $(git rev-parse HEAD) == "$EXPECTED_COMMIT" && -z $(git status --porcelain) ]]
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
exec > >(tee -a "$RUN/controller.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/controller_exit_code"' EXIT
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
printf '%s  %s\n' af1f8d11a52279051d0deea96db81f3522de7bf556b089f921943d832e1224e6 "$DATA/sft1_train_all.jsonl" 1632d9aebbe499076fe1d9477fa83c2dbfbadea593189ece640603bdad0115dd "$DATA/sft1_heldout_all.jsonl" | sha256sum --check
ARGS=(--model /mnt/nimloth/checkpoint/hf_actor --train-jsonl "$DATA/sft1_train_all.jsonl" --val-jsonl "$DATA/sft1_heldout_all.jsonl" --output-dir "$RUN" --epochs 1 --batch-size 1 --grad-accum 8 --lr 1e-6 --embedding-lr 5e-6 --lora --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 --weight-decay 0.01 --warmup-ratio 0.05 --latent-token-count 1 --latent-query-mode generate --max-length 20000 --max-pixels 100352 --min-pixels 3136 --attn-implementation flash_attention_2 --gradient-checkpointing --seed 42 --resume-save-steps 10 --no-wandb --cache-dir "$RUN/preprocess_cache" --cache-pixel-dtype bfloat16 --preprocess-workers 8 --num-workers 4 --format-eval-samples 32 --max-val-records -1 --max-val-batches -1)
"$PY" - "$RUN" "$EXPECTED_COMMIT" "${ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
run = Path(sys.argv[1])
(run/'launch.json').write_text(json.dumps(dict(run_output=str(run), commit=sys.argv[2], args=sys.argv[3:], controller_pid=os.getppid(), controller_pgid=os.getpgid(os.getppid()), budget='6h external timeout', wandb=False), indent=2)+'\n')
PY
printf '%q ' "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --require-prebuilt-cache > "$RUN/train_command.sh"
printf '\n' >> "$RUN/train_command.sh"
CUDA_VISIBLE_DEVICES='' "$PY" -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --cache-only --rebuild-cache
"$PY" - "$RUN" <<'PY_CACHE'
import json, sys
from pathlib import Path
import torch
run = Path(sys.argv[1])
result = {}
for split, count in [('train', 1709), ('val', 193)]:
    dirs = list((run/'preprocess_cache').glob(split+'_sft1_*'))
    assert len(dirs) == 1, dirs
    manifest = json.loads((dirs[0]/'manifest.json').read_text())
    expected = dict(count=count, max_length=20000, latent_token_count=1,
                    mask_latent_query_labels=False, latent_query_mode='generate',
                    cache_pixel_dtype='bfloat16', dir=str(dirs[0]))
    assert all(manifest.get(key) == value for key, value in expected.items()), manifest
    files = list(dirs[0].glob('*.pt'))
    assert len(files) == count, (split, len(files), count)
    maximum = 0
    for path in files:
        item = torch.load(path, map_location='cpu', weights_only=True)
        assert item['input_ids'].ndim == 1, path
        assert item['labels'].shape == item['input_ids'].shape, path
        assert item['attention_mask'].shape == item['input_ids'].shape, path
        length = item['input_ids'].numel()
        assert 0 < length < 20000, (path, length)
        assert (item['labels'] != -100).any().item(), path
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert torch.isfinite(value).all().item(), (path, key)
        maximum = max(maximum, length)
    result[split] = dict(count=count, maximum_length=maximum, no_truncation=True)
(run/'cache_validation.json').write_text(json.dumps(result, indent=2)+'\n')
PY_CACHE
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$RUN/gpu_before.csv"
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$RUN/gpu_processes_before.txt"
"$PY" - "$RUN" <<'PY'
import csv, sys
from pathlib import Path
run = Path(sys.argv[1])
rows = list(csv.reader((run/'gpu_before.csv').read_text().splitlines()))
assert len(rows) == 8 and {int(r[0]) for r in rows} == set(range(8)), rows
assert all(int(r[1]) < 100 and int(r[2]) == 0 for r in rows), rows
assert not (run/'gpu_processes_before.txt').read_text().strip(), 'GPU compute processes exist'
PY
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --require-prebuilt-cache
printf '0\n' > "$RUN/TRAIN_SUCCEEDED"
"$PY" .trellis/tasks/09-10-sft1-rollout2000/research/cleanup_checkpoints.py "$RUN"
