#!/usr/bin/env bash
# Preprocess: bash launch.sh COMMIT RUN_ID preprocess
# Train under the approved external timeout: bash launch.sh COMMIT RUN_ID train POLICY.json
set -Eeuo pipefail
[[ $# -ge 3 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_]+$ ]]
EXPECTED_COMMIT=$1
RUN_ID=$2
PHASE=$3
[[ $PHASE == preprocess || $PHASE == train || $PHASE == resume ]]
if [[ $PHASE == preprocess ]]; then [[ $# == 3 ]]; else [[ $# == 4 ]]; fi
POLICY=${4:-}
ROOT=/mnt/nimloth/.worktree/sft1-rollout2000
PY=/mnt/nimloth/venv/bin/python3
DATA=/mnt/nimloth/outputs/datasets/sft1-vagen-step60/20260910T093222Z_batch1_original_validation_k16
RUN=/mnt/nimloth/outputs/experiments/sft1-rollout2000/$RUN_ID
cd "$ROOT"
[[ $(git rev-parse HEAD) == "$EXPECTED_COMMIT" && -z $(git status --porcelain) ]]
mkdir -p "$(dirname "$RUN")"
if [[ $PHASE == preprocess ]]; then
    mkdir "$RUN"
else
    [[ -d $RUN && -f $RUN/PREPROCESS_SUCCEEDED ]]
fi
exec 9> "$RUN/.controller.lock"
flock -n 9
SEGMENT=$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee -a "$RUN/${PHASE}_${SEGMENT}_controller.log") 2>&1
WATCH_PID=''
WATCH_STOP="$RUN/${PHASE}_${SEGMENT}_watch_stop"
stop_watcher() {
    if [[ -n $WATCH_PID ]]; then
        touch "$WATCH_STOP"
        if wait "$WATCH_PID"; then
            printf '0\n' > "$RUN/${PHASE}_${SEGMENT}_watch_exit_code"
        else
            printf '%s\n' "$?" > "$RUN/${PHASE}_${SEGMENT}_watch_exit_code"
        fi
        WATCH_PID=''
    fi
}
on_exit() {
    rc=$?
    stop_watcher
    printf '%s\n' "$rc" > "$RUN/${PHASE}_${SEGMENT}_controller_exit_code"
}
trap on_exit EXIT
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CPATH=/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:/mnt/nimloth/dependencies/python310-dev/root/usr/include${CPATH:+:$CPATH}
CUDA_VISIBLE_DEVICES='' "$PY" - <<'PY_TRITON'
from pathlib import Path
from triton.backends.nvidia import driver
# Compile without initializing CUDA, so missing development headers fail before DDP.
driver.compile_module_from_src(Path(driver.__file__).with_name('driver.c').read_text(), 'cuda_utils')
print('Triton driver compilation preflight passed')
PY_TRITON
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
printf '%s  %s\n' af1f8d11a52279051d0deea96db81f3522de7bf556b089f921943d832e1224e6 "$DATA/sft1_train_all.jsonl" 1632d9aebbe499076fe1d9477fa83c2dbfbadea593189ece640603bdad0115dd "$DATA/sft1_heldout_all.jsonl" | sha256sum --check
ARGS=(--model /mnt/nimloth/checkpoint/hf_actor --train-jsonl "$DATA/sft1_train_all.jsonl" --val-jsonl "$DATA/sft1_heldout_all.jsonl" --output-dir "$RUN" --batch-size 1 --grad-accum 8 --lr 1e-6 --embedding-lr 5e-6 --lora --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 --weight-decay 0.01 --warmup-ratio 0.05 --max-length 20000 --max-pixels 100352 --min-pixels 3136 --attn-implementation flash_attention_2 --gradient-checkpointing --seed 42 --resume-save-steps 10 --no-wandb --cache-dir "$RUN/preprocess_cache" --cache-pixel-dtype bfloat16 --preprocess-workers 8 --num-workers 4 --format-eval-samples 32 --max-val-records -1 --max-val-batches -1)
if [[ $PHASE == preprocess ]]; then
"$PY" - "$RUN" "$EXPECTED_COMMIT" "${ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
run = Path(sys.argv[1])
(run/'launch.json').write_text(json.dumps(dict(run_output=str(run), commit=sys.argv[2], args=sys.argv[3:], controller_pid=os.getppid(), controller_pgid=os.getpgid(os.getppid()), phase='preprocess', wandb=False), indent=2)+'\n')
PY
if [[ -n ${SFT1_CACHE_SOURCE:-} ]]; then
    [[ $SFT1_CACHE_SOURCE == /mnt/nimloth/outputs/experiments/sft1-rollout2000/* && -f $SFT1_CACHE_SOURCE/PREPROCESS_SUCCEEDED && -f $SFT1_CACHE_SOURCE/cache_validation.json ]]
    [[ $(cat "$SFT1_CACHE_SOURCE/PREPROCESS_SUCCEEDED") == 0 ]]
    cp -a --reflink=auto "$SFT1_CACHE_SOURCE/preprocess_cache" "$RUN/preprocess_cache"
    printf '%s\n' "$SFT1_CACHE_SOURCE" > "$RUN/cache_source.txt"
    CUDA_VISIBLE_DEVICES='' "$PY" -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --cache-only
else
    CUDA_VISIBLE_DEVICES='' "$PY" -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --cache-only --rebuild-cache
fi
fi
"$PY" - "$RUN" <<'PY_CACHE'
import hashlib, json, re, sys
from pathlib import Path
import torch
from transformers import AutoProcessor
from nimloth.latent import add_special_tokens
run = Path(sys.argv[1])
processor = AutoProcessor.from_pretrained('/mnt/nimloth/checkpoint/hf_actor', trust_remote_code=True,
                                         min_pixels=3136, max_pixels=100352)
add_special_tokens(processor.tokenizer, latent_token_count=None)
marker = re.compile(r'<\|latent_state(?:_\d+)?\|>')
latent_ids = {token_id for token, token_id in processor.tokenizer.get_vocab().items()
              if marker.fullmatch(token)}
result = {'latent_token_ids_checked': sorted(latent_ids)}
for split, count in [('train', 1709), ('val', 193)]:
    dirs = list((run/'preprocess_cache').glob(split+'_sft1_*'))
    assert len(dirs) == 1, dirs
    manifest = json.loads((dirs[0]/'manifest.json').read_text())
    expected = dict(count=count, max_length=20000, latent_token_count=None,
                    mask_latent_query_labels=None, latent_query_mode=None,
                    cache_schema='nimloth_early_stage_cache_v7',
                    format_objective='format_answer_ce_v2',
                    text_projection='remove_latent_markers_all_roles',
                    cache_pixel_dtype='bfloat16', dir=str(dirs[0]))
    assert all(manifest.get(key) == value for key, value in expected.items()), manifest
    files = list(dirs[0].glob('*.pt'))
    assert len(files) == count, (split, len(files), count)
    maximum = 0
    hashes = {}
    for path in sorted(files):
        item = torch.load(path, map_location='cpu', weights_only=True)
        assert item['input_ids'].ndim == 1, path
        assert item['labels'].shape == item['input_ids'].shape, path
        assert item['attention_mask'].shape == item['input_ids'].shape, path
        length = item['input_ids'].numel()
        assert 0 < length < 20000, (path, length)
        assert (item['labels'] != -100).any().item(), path
        for key in ('input_ids', 'labels'):
            ids = [int(x) for x in item[key].tolist() if int(x) != -100]
            assert not latent_ids.intersection(ids), (path, key, 'latent token IDs')
            assert not marker.search(processor.tokenizer.decode(ids, skip_special_tokens=False)), (path, key)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert torch.isfinite(value).all().item(), (path, key)
        maximum = max(maximum, length)
    result[split] = dict(count=count, maximum_length=maximum, no_truncation=True, query_free=True, sha256=hashes, manifest=manifest)
validation = json.dumps(result, indent=2)+'\n'
if (run/'PREPROCESS_SUCCEEDED').exists():
    assert (run/'cache_validation.json').read_text() == validation, 'cache validation changed'
else:
    (run/'cache_validation.json').write_text(validation)
PY_CACHE
if [[ $PHASE == preprocess ]]; then
    printf '0\n' > "$RUN/PREPROCESS_SUCCEEDED"
    exit 0
fi
"$PY" - "$RUN" "$EXPECTED_COMMIT" "$POLICY" "$PHASE" "${ARGS[@]}" <<'PY_POLICY'
import json, sys
from pathlib import Path
run, commit, policy_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
metadata = json.loads((run/'launch.json').read_text())
assert (run/'PREPROCESS_SUCCEEDED').read_text().strip() == '0'
assert metadata['run_output'] == str(run) and metadata['commit'] == commit
assert metadata['args'] == sys.argv[5:], 'preprocess arguments changed'
policy = json.loads(policy_path.read_text())
assert policy.get('until_converged') is True
assert policy.get('min_epochs') == 2
assert policy.get('patience_epochs') == 2
assert policy.get('min_relative_improvement') == 0.01
assert policy.get('budget') == '6h per segment; pause and resume until convergence'
args = ['--until-converged', '--convergence-min-epochs', '2',
        '--convergence-patience-epochs', '2', '--convergence-min-relative-improvement', '0.01']
policy_output = run/'training_policy.json'
if sys.argv[4] == 'resume':
    assert json.loads(policy_output.read_text()) == policy, 'resume policy mismatch'
    assert (run/'TRAIN_STARTED').exists() and not (run/'TRAIN_SUCCEEDED').exists()
    args.append('--resume')
else:
    assert not (run/'TRAIN_STARTED').exists()
    with policy_output.open('x') as output:
        json.dump(policy, output, indent=2)
(run/'policy_args.nul').write_bytes(b''.join(x.encode()+b'\0' for x in args))
PY_POLICY
mapfile -d '' -t POLICY_ARGS < "$RUN/policy_args.nul"
ARGS+=("${POLICY_ARGS[@]}")
printf '%q ' "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --require-prebuilt-cache > "$RUN/${PHASE}_${SEGMENT}_command.sh"
printf '\n' >> "$RUN/${PHASE}_${SEGMENT}_command.sh"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$RUN/${PHASE}_${SEGMENT}_gpu_before.csv"
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$RUN/${PHASE}_${SEGMENT}_gpu_processes_before.txt"
"$PY" - "$RUN" "${PHASE}_${SEGMENT}" <<'PY'
import csv, sys
from pathlib import Path
run = Path(sys.argv[1])
rows = list(csv.reader((run/f'{sys.argv[2]}_gpu_before.csv').read_text().splitlines()))
assert len(rows) == 8 and {int(r[0]) for r in rows} == set(range(8)), rows
assert all(int(r[1]) < 100 and int(r[2]) == 0 for r in rows), rows
assert not (run/f'{sys.argv[2]}_gpu_processes_before.txt').read_text().strip(), 'GPU compute processes exist'
PY
printf 'started\n' > "$RUN/TRAIN_STARTED"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES='' "$PY" .trellis/tasks/09-10-sft1-rollout2000/research/watch_checkpoints.py "$RUN" "$WATCH_STOP" > "$RUN/${PHASE}_${SEGMENT}_cleanup_watch.log" 2>&1 &
WATCH_PID=$!
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m nimloth.training.sft.stage1.trainer "${ARGS[@]}" --require-prebuilt-cache
stop_watcher
"$PY" - "$RUN" <<'PY_CONVERGED'
import json, sys
from pathlib import Path
import torch
from nimloth.training.sft.stage1.convergence import ConvergencePolicy, ConvergenceState
run = Path(sys.argv[1])
marker = json.loads((run/'CONVERGED.json').read_text())
assert marker['monitor'] == 'validation_lm_loss'
assert marker['policy'] == ConvergencePolicy(2, 2, 0.01).state_dict()
state = ConvergenceState.from_state_dict(marker['state'])
assert state.converged and state.last_epoch >= 2 and state.bad_epochs >= 2
final = torch.load(run/'final'/'training_state.pt', map_location='cpu', weights_only=False)
assert final['epoch'] == state.last_epoch and final['step'] == marker['global_step']
assert final['convergence_state'] == marker['state']
PY_CONVERGED
printf '0\n' > "$RUN/TRAIN_SUCCEEDED"
"$PY" .trellis/tasks/09-10-sft1-rollout2000/research/cleanup_checkpoints.py "$RUN"
