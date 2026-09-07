#!/bin/bash
#SBATCH --job-name=step60-prepared-lane
#SBATCH --time=06:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --no-requeue

set -euo pipefail

for name in REPO VAGEN_DIR VERL_DIR PYTHON_ENV INIT_HF ROLLOUT_PREPARED_MANIFEST ROLLOUT_RUN_DIR EXPECTED_REPO_COMMIT SLURM_JOB_ID ROLLOUT_LANE_ID; do
  test -n "${!name:?required environment variable}"
done
case "$ROLLOUT_LANE_ID" in 0|1) ;; *) echo 'lane must be 0 or 1' >&2; exit 2;; esac
for name in REPO VAGEN_DIR VERL_DIR PYTHON_ENV INIT_HF ROLLOUT_PREPARED_MANIFEST ROLLOUT_RUN_DIR; do
  [[ "${!name}" = /* ]] || { echo "$name must be absolute" >&2; exit 2; }
done
test "${SLURM_JOB_NUM_NODES:?}" -eq 1
test "${SLURM_CPUS_PER_TASK:?}" -eq 28
test -x "$PYTHON_ENV/bin/python3"
test -d "$INIT_HF"
job_id=$SLURM_JOB_ID
expected_vagen=c44d7ba122090b1afc86ee8dc1164378e1abb9cf
expected_verl=bc26648bb632fe153cd4e7ba27d63c53869d86ca
check_sources() {
  local repo_path
  test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_REPO_COMMIT"
  test "$(git -C "$VAGEN_DIR" rev-parse HEAD)" = "$expected_vagen"
  test "$(git -C "$VERL_DIR" rev-parse HEAD)" = "$expected_verl"
  for repo_path in "$REPO" "$VAGEN_DIR" "$VERL_DIR"; do
    test "$(git -C "$repo_path" rev-parse --show-toplevel)" = "$repo_path"
    test -z "$(git -C "$repo_path" status --porcelain --untracked-files=all)"
  done
}
check_sources
export REPO VAGEN_DIR VERL_DIR PYTHON_ENV INIT_HF ROLLOUT_PREPARED_MANIFEST
export PYTHONDONTWRITEBYTECODE=1 INIT_HF_STEP=60 ROLLOUT_DUMP_STEP=0
export ROLLOUT_SMOKE=0 ROLLOUT_SOURCE_PARITY=0 ROLLOUT_TRAIN120=0
export TOTAL_GPU_COUNT=2 ENV_GPU_COUNT=1 RAY_USAGE_STATS_ENABLED=0
export ROLLOUT_PREPARED_OUTPUT_PREFIX=reconstruction1800
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
# 每个任务原子预留独占输出目录；失败目录不可复用。
run_root="$ROLLOUT_RUN_DIR"
mkdir "$run_root"
srun_pid=''
on_exit() {
  rc=$?
  trap - EXIT
  if [ -n "$srun_pid" ] && kill -0 "$srun_pid" 2>/dev/null; then
    kill -TERM "$srun_pid" || true
    wait "$srun_pid" || true
  fi
  printf '%s\n' "$rc" > "$run_root/controller_exit_code.txt"
  date -u +%FT%TZ > "$run_root/controller_end_utc.txt"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 143' TERM INT
PYTHONPATH="$REPO:$REPO/src" "$PYTHON_ENV/bin/python3" - "$run_root" <<'PYPLAN'
import json
import os
import sys
from collections import Counter
from pathlib import Path
from experiments.training.sft1.prepare_source200 import verify, sha256, runtime_overrides

root = Path(sys.argv[1])
path = Path(os.environ['ROLLOUT_PREPARED_MANIFEST'])
manifest = json.loads(path.read_text())
paths = verify(path)
if (manifest['format'] != 'source_batch1_remainder_prepared_v1'
        or manifest['count'] != 1800 or len(paths) != 90
        or manifest['runtime_overrides'] != runtime_overrides('step60_source_reconstruction')):
    raise ValueError('expected reconstruction remainder: 90 shards / 1800 rows')
rows = [row for shard in manifest['shards'] for row in shard['rows']]
if Counter(row['dataset_split'] for row in rows) != {'train': 1600, 'heldout': 200}:
    raise ValueError('remainder split mismatch')
lane = int(os.environ['ROLLOUT_LANE_ID'])
indices = list(range(lane, 90, 2))
if len(set(indices)) != 45 or sum(manifest['shards'][i]['count'] for i in indices) != 900:
    raise ValueError('lane must contain 45 unique shards / 900 rows')
chunks = [indices[start:start + 4] for start in range(0, 45, 4)]
contract = {name: os.environ[name] for name in (
    'REPO', 'VAGEN_DIR', 'VERL_DIR', 'PYTHON_ENV', 'INIT_HF',
    'EXPECTED_REPO_COMMIT', 'SLURM_JOB_ID', 'ROLLOUT_LANE_ID',
    'ROLLOUT_PREPARED_MANIFEST')}
contract.update(prepared_sha256=sha256(path), indices=indices, chunks=chunks,
                expected_rows=900, resources='1 node / 2 GPU / 28 CPU / 90G / 6h',
                vagen_commit='c44d7ba122090b1afc86ee8dc1164378e1abb9cf',
                verl_commit='bc26648bb632fe153cd4e7ba27d63c53869d86ca')
(root / 'launch_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
(root / 'chunks.txt').write_text(''.join(','.join(map(str, chunk)) + '\n' for chunk in chunks))
PYPLAN
chunk=0
while IFS= read -r indices; do
  check_sources
  printf -v chunk_name 'chunk_%02d' "$chunk"
  export ROLLOUT_PREPARED_SHARD_INDICES=$indices
  export ROLLOUT_RUN_DIR="$run_root/$chunk_name"
  export ROLLOUT_RUN_NAME="step60_remainder_${job_id}_lane${ROLLOUT_LANE_ID}_${chunk_name}"
  export RUNTIME_NAMESPACE="$ROLLOUT_RUN_NAME"
  export PORT_BASE=$((30000 + ROLLOUT_LANE_ID * 2000 + chunk * 10))
  export RAY_PORT=$((PORT_BASE + 2)) RAY_DASHBOARD_PORT=$((PORT_BASE + 3))
  test ! -e "$ROLLOUT_RUN_DIR"
  printf 'START %s indices=%s utc=%s\n' "$chunk_name" "$indices" "$(date -u +%FT%TZ)"
  /cm/shared/apps/slurm/current/bin/srun --jobid="$job_id" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=28 --gres=gpu:2 --mem=90G --export=ALL /bin/bash "$REPO/experiments/training/sft1/rollout_full_6gpu_preempt.slurm" < /dev/null > "$run_root/$chunk_name.controller.log" 2>&1 &
  srun_pid=$!
  printf '%s\n' "$srun_pid" > "$run_root/$chunk_name.srun_pid.txt"
  if wait "$srun_pid"; then rc=0; else rc=$?; fi
  srun_pid=''
  printf '%s\n' "$rc" > "$run_root/$chunk_name.exit_code.txt"
  test "$rc" -eq 0
  test -f "$ROLLOUT_RUN_DIR/preempt_2gpu_rollout_done.flag"
  "$PYTHON_ENV/bin/python3" - "$ROLLOUT_RUN_DIR" "$indices" "$run_root/launch_contract.json" <<'PYCHECK'
import hashlib
import json
import sys
from pathlib import Path

root, indices, contract_path = Path(sys.argv[1]), sys.argv[2].split(','), Path(sys.argv[3])
contract = json.loads(contract_path.read_text())
prepared = Path(contract['ROLLOUT_PREPARED_MANIFEST'])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(prepared) != contract['prepared_sha256']:
    raise ValueError('prepared manifest changed during collection')
manifest = json.loads(prepared.read_text())
expected = {root / 'validation/train' / f'reconstruction1800_{int(i):02d}' / '0.validation.json' for i in indices}
if set(root.rglob('*.validation.json')) != expected:
    raise ValueError('chunk completion marker membership mismatch')
for i in indices:
    marker = root / 'validation/train' / f'reconstruction1800_{int(i):02d}' / '0.validation.json'
    result = json.loads(marker.read_text())
    shard = manifest['shards'][int(i)]
    if (result['count'] != 20 or result['unique_identities'] != 20
            or result['identity_coverage'] != 'exact'
            or result['jsonl_sha256'] != sha(marker.with_name('0.jsonl'))
            or result['prepared_parquet_sha256'] != shard['sha256']):
        raise ValueError(f'invalid chunk completion evidence: {marker}')
PYCHECK
  printf 'COMPLETE %s utc=%s\n' "$chunk_name" "$(date -u +%FT%TZ)"
  chunk=$((chunk + 1))
done < "$run_root/chunks.txt"
test "$chunk" -eq 12
printf 'ALL_900_LANE_COLLECTION_COMPLETE\n' > "$run_root/collection_done.flag"
