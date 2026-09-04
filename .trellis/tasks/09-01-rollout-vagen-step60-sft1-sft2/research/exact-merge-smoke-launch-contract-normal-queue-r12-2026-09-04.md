# Exact normal-queue R12 launch contract — step60 actor merge + one-row GPU smoke

Date: 2026-09-04
Status: **candidate exact contract; not launch authorization**

This contract replaces terminal pre-root R11 after the human explicitly selected submission to `normal` with a maximum pending wait of 24 hours. R12 records one live normal snapshot, submits an unpinned 4-GPU/112-CPU/96G request even when no GPU is immediately free, monitors PENDING/RUNNING plus accounting for at most 24 hours, and cancels fail-closed on timeout or terminal scheduler state. Once allocated, it requires at least 80GiB observed free memory before partition/hash/merge. Detached execution, stdin-closed `srun`, merge/smoke gates and source-index-0 scope remain unchanged. It authorizes nothing until exact review/commit/push and separate launch approval; later stages remain excluded.

Remote CPU evidence before this draft: affected suite `125 passed`; all three readiness-marker variants passed NFSv3 success, existing-target preservation, concurrent single-winner, final-marker ordering and both interruption rejection gates at `/project/peilab/atst/nimloth/.local/tmp/step60-nfs-publication-probe-20260902T121624Z-32bcc045` (summary SHA256 `aa63a9a8851a6e2df0960b896fad0d08c98d167155d82d3119eedcb051db1d5f`). A real pinned-source partition was published and all ten sibling parquets rehashed at `/project/peilab/atst/nimloth/.local/tmp/step60-cpu-preflight-20260902T122133Z-32bcc045/partition`; manifest SHA256 `be7db7ea975927bc176186bcb51a202b3be191196ced26e043a57add5f99b87c`. Regenerated continuation evidence under `/project/peilab/atst/nimloth/.local/tmp/step60-cpu-preflight-cont-20260902T122241Z-32bcc045` is runtime-contract file SHA256 `7b9184b8e33d76c0d410b141d4cff9ea993bef43708f5f9d16e7b2972718e9e8` (payload `cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf`), checkpoint inspection SHA256 `a819cbef1fafd5b9b9ef391b546ec092fa7a2193cd656d461ec258afe17ab500`, and inert merge-plan SHA256 `5e9472705ac54bbe75bbe6c1688c26fc8ef530291ae268565f78db0495732c05`.

## Purpose, boundary and failure condition

Falsifiable question: can the hash-bound reconstruction runtime and frozen step60 actor execute one complete source-protocol trajectory under the human-selected vLLM 0.8.2 runtime while preserving prompt/parser/reward/image/EOS/terminal-non-step contracts? Any identity, merge/load, service, response-boundary, parser, reward, image, terminal, scheduler or resource failure rejects the smoke. Passing proves only this one path.

No module is trainable; no optimizer or objective exists. Actor, vision encoder, environment and Nimloth modules are frozen. The terminal draft action is generated but never executed or supervised.

## Immutable identities

- launch code worktree/commit: `/project/peilab/atst/nimloth/.worktree/rollout-vagen-step60-sft1-sft2` at `7dac687b733cccffaf0a211ef0a602ec001749dd`
- reconstruction worktree/head/tree/diff: `/project/peilab/atst/nimloth/.worktree/vagen-step60-runtime-reconstruction-vagen`, `170a673d1bf5855fc0ea6fbed0744b3d7168f8f0`, `58ef0eb66ad0bef7587c253c5c643af572c1d3a7`, `7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9`
- reconstruction parent/base: `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`
- merger dependency: Nimloth VAGEN gitlink `9f1e89eb8c9839a406b6e62aa75703494a79e5b5`; nested VERL `494f264494b2525f2c13595f63ac4912963e6d2f`; merger script SHA256 `3e2794e1e9e566a4aeb0d709dad7d2b8864c8b91e4f72cf0d265ecb62c311044`
- checkpoint actor: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60/actor`
- source parquet/SHA256: `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/train.parquet`, `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`
- executable Python/packages: `/project/peilab/atst/nimloth/.venv/bin/python3`; vLLM `0.8.2`, Transformers `4.49.0`, Torch `2.6.0`
- source package evidence: vLLM `0.8.5.post1`, Transformers `4.49.0`, Torch `2.6.0`
- executable vLLM source hashes: `outputs.py=047d469792ba4b332fd6bc6837af03340135cb49798e1ddfd2ffa730ead436f8`, `stop_checker.py=5ed39ad2df9912b7a4b9ff52168c50bfe9d937675d3f1122148c0824450afa28`
- v3 runtime-contract payload SHA256: `cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf`

## Output and resource contract

Unique run root; it and all launcher paths must be absent at exact launch, with executable fail-closed checks:

`/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260904T220000Z_step60_batch1_v3_normal_queue_r12_7dac687b`

Allowed children: `partition/`, `runtime_contract.json`, `merge/hf_actor/`, `smoke/source-index-00000/`, failed `smoke/source-index-00000.partial-<12 hex>/`, `logs/`, `control/`, `metadata.json`, `LAUNCH_CONTRACT.md`, `RESOLVED_LAUNCH_CONTRACT.md`, `README.md`, and `END.json`. Existing paths are never reused or removed. W&B is disabled.

Slurm: account `peilab`, partition `normal`, any healthy single node, one task, four GPUs, 112 CPUs, 96 GiB scheduler memory, walltime `03:00:00`. The one smoke step receives all four GPUs and fails unless its initial `CUDA_VISIBLE_DEVICES` resolves to four distinct entries. Policy is restricted to logical `0,1` with TP2; service is restricted to logical `2,3`, exposing its two devices internally as `[0,1]`, `max_workers=2`. R8 proved this lower envelope allocatable on `dgx-28`; that is historical evidence only. `dgx-51` remains excluded because it has not been requalified after prior AI2-THOR prewarm failures. Availability and requested CPU/memory are rechecked immediately before submission. Expected duration is under two hours.

## Exact execution script

Immediately before the remote script, run and enforce the repository resource query:

```bash
bash /workspace/remote2/nimloth/.local/scripts/query-resources.sh --partition normal
```

After exact experiment-launch approval, replace `<EXACT_APPROVED_DOCS_COMMIT>` below with its literal approved hash and execute this bootstrap exactly. It extracts the largest bash block byte-for-byte, verifies SHA256 `afaa97add79c8827f7388db5cf4e052c850ea553ad31a7fdfdbe9286e56ab99e`, reserves the remote temporary script with shell noclobber, atomically links it to the final script without overwrite, and starts detached execution with stdin `/dev/null`. All local/remote launcher paths and the run root must be absent; no retry occurs.

```bash
set -euo pipefail
DOCS_COMMIT=<EXACT_APPROVED_DOCS_COMMIT>
WT=/workspace/remote2/nimloth/.worktree/rollout-vagen-step60-docs-clean
DOC=.trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2/research/exact-merge-smoke-launch-contract-normal-queue-r12-2026-09-04.md
LOCAL=/tmp/nimloth-approved-step60-r12-main.sh
SHA=afaa97add79c8827f7388db5cf4e052c850ea553ad31a7fdfdbe9286e56ab99e
test ! -e "$LOCAL" && test ! -L "$LOCAL"
git -C "$WT" show "$DOCS_COMMIT:$DOC" | python3 -c 'import re,sys,pathlib; blocks=re.findall(r"```bash\n(.*?)\n```",sys.stdin.read(),re.S); main=max(blocks,key=len)+"\n"; pathlib.Path(sys.argv[1]).open("x").write(main)' "$LOCAL"
test "$(sha256sum "$LOCAL" | awk '{print $1}')" = "$SHA"
ssh superpod-csejzhang "set -euo pipefail; TMP=/project/peilab/atst/nimloth/.local/tmp/step60-r12-launch-20260904T220000Z.sh.tmp; SCRIPT=/project/peilab/atst/nimloth/.local/tmp/step60-r12-launch-20260904T220000Z.sh; LOG=/project/peilab/atst/nimloth/.local/tmp/step60-r12-launch-20260904T220000Z.log; PIDFILE=/project/peilab/atst/nimloth/.local/tmp/step60-r12-launch-20260904T220000Z.pid; RUN=/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260904T220000Z_step60_batch1_v3_normal_queue_r12_7dac687b; for p in \"\$TMP\" \"\$SCRIPT\" \"\$LOG\" \"\$PIDFILE\" \"\$RUN\"; do test ! -e \"\$p\" && test ! -L \"\$p\"; done; umask 077; set -o noclobber; cat > \"\$TMP\"; chmod 0500 \"\$TMP\"; test \"\$(sha256sum \"\$TMP\" | awk '{print \$1}')\" = $SHA; ln \"\$TMP\" \"\$SCRIPT\"; rm \"\$TMP\"; nohup bash \"\$SCRIPT\" $DOCS_COMMIT </dev/null >\"\$LOG\" 2>&1 & pid=\$!; printf '%s\\n' \"\$pid\" > \"\$PIDFILE\"; kill -0 \"\$pid\"; printf 'R12_LAUNCHER_PID=%s\\n' \"\$pid\"" < "$LOCAL"
```

Inside the detached main script `$1` is mandatory and hash-checked. Any error after `HOLD` is assigned triggers exact cancellation and a terminal Slurm state.

```bash
set -euo pipefail
ROOT=/project/peilab/atst/nimloth
NWT=$ROOT/.worktree/rollout-vagen-step60-sft1-sft2
VWT=$ROOT/.worktree/vagen-step60-runtime-reconstruction-vagen
PY=$ROOT/.venv/bin/python3
RUN=$ROOT/outputs/experiments/training/sft1-vagen-step60/20260904T220000Z_step60_batch1_v3_normal_queue_r12_7dac687b
ACTOR=/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60/actor
SOURCE=/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/train.parquet
TASK_REF=refs/remotes/origin/task/rollout-vagen-step60-sft1-sft2
TASK_DOC=.trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2/research/exact-merge-smoke-launch-contract-normal-queue-r12-2026-09-04.md
APPROVED_DOCS_COMMIT=${1:?missing-approved-docs-commit}
HOLD=
NODE=
cleanup_hold() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$HOLD"; then
    terminal=0
    state=
    while test "$terminal" -ne 1; do
      scancel "$HOLD" 2>/dev/null || true
      state=$({ sacct -n -X -j "$HOLD" --format=State -P 2>/dev/null || true; } | awk -F'|' 'NR==1 {value=$1} END {print value}')
      case "$state" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) terminal=1;; *) sleep 5;; esac
    done
    sacct -j "$HOLD" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList,AllocTRES,MaxRSS -P || true
  fi
  exit "$rc"
}
trap cleanup_hold EXIT
trap 'exit 130' INT TERM

# Pre-GPU identity/package/source/output gates.
test "$(git -C "$NWT" rev-parse --show-toplevel)" = "$NWT"
test "$(git -C "$NWT" rev-parse HEAD)" = 7dac687b733cccffaf0a211ef0a602ec001749dd
test "$(git -C "$NWT" rev-parse --git-common-dir)" = "$ROOT/.git"
test -z "$(git -C "$NWT" status --porcelain=v1 --untracked-files=all)"
NIMLOTH_WORKTREES=$(git -C "$ROOT" worktree list --porcelain)
grep -Fqx "worktree $NWT" <<< "$NIMLOTH_WORKTREES"
test "$(git -C "$NWT/external/VAGEN" rev-parse HEAD)" = 9f1e89eb8c9839a406b6e62aa75703494a79e5b5
test "$(git -C "$NWT/external/VAGEN/verl" rev-parse HEAD)" = 494f264494b2525f2c13595f63ac4912963e6d2f
test "$(git -C "$NWT/external/le-wm" rev-parse HEAD)" = 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
for d in "$NWT/external/VAGEN" "$NWT/external/VAGEN/verl" "$NWT/external/le-wm"; do test -z "$(git -C "$d" status --porcelain=v1 --untracked-files=all)"; done
test "$(git -C "$VWT" rev-parse --show-toplevel)" = "$VWT"
test "$(git -C "$VWT" rev-parse --git-common-dir)" = "$ROOT/.git/modules/external/VAGEN"
VAGEN_WORKTREES=$(git -C "$ROOT/external/VAGEN" worktree list --porcelain)
grep -Fqx "worktree $VWT" <<< "$VAGEN_WORKTREES"
test "$(git -C "$VWT" rev-parse HEAD)" = 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0
test "$(git -C "$VWT" rev-parse HEAD^)" = 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a
test "$(git -C "$VWT" rev-list --count 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD)" = 1
test "$(git -C "$VWT" rev-parse HEAD^{tree})" = 58ef0eb66ad0bef7587c253c5c643af572c1d3a7
test "$(git -C "$VWT" --no-pager diff --binary --full-index --no-ext-diff 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD -- | sha256sum | awk '{print $1}')" = 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9
test -z "$(git -C "$VWT" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$NWT/external/VAGEN/verl/scripts/legacy_model_merger.py" | awk '{print $1}')" = 3e2794e1e9e566a4aeb0d709dad7d2b8864c8b91e4f72cf0d265ecb62c311044
test "$(sha256sum "$ROOT/.venv/lib/python3.10/site-packages/vllm/outputs.py" | awk '{print $1}')" = 047d469792ba4b332fd6bc6837af03340135cb49798e1ddfd2ffa730ead436f8
test "$(sha256sum "$ROOT/.venv/lib/python3.10/site-packages/vllm/engine/output_processor/stop_checker.py" | awk '{print $1}')" = 5ed39ad2df9912b7a4b9ff52168c50bfe9d937675d3f1122148c0824450afa28
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/external/VAGEN/verl:$NWT/external/VAGEN:$NWT/src:$NWT" "$PY" - "$NWT" <<'PY'
from pathlib import Path
import sys, vagen, verl
import verl.utils.fsdp_utils as fsdp_utils
root=Path(sys.argv[1]).resolve()
for module, expected in ((vagen,root/'external/VAGEN'),(verl,root/'external/VAGEN/verl'),(fsdp_utils,root/'external/VAGEN/verl')):
    actual=Path(module.__file__).resolve()
    assert actual.is_relative_to(expected), (module.__name__,actual,expected)
PY
PREFLIGHT_TARGET=$ROOT/.local/tmp/step60-r12-inert-target-20260904T220000Z-7dac687b
test -d "$ROOT/.local/tmp"
test ! -e "$PREFLIGHT_TARGET" && test ! -L "$PREFLIGHT_TARGET"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/external/VAGEN/verl:$NWT/external/VAGEN:$NWT/src:$NWT" "$PY" - "$ACTOR" "$PREFLIGHT_TARGET" "$PY" "$NWT/external/VAGEN/verl/scripts/legacy_model_merger.py" <<'PY'
from pathlib import Path
import accelerate, sys, torch, torch.utils, torch.utils.hooks
from experiments.training.sft1.vagen_step60_checkpoint import prepare_merge_plan
assert sys.executable == '/project/peilab/atst/nimloth/.venv/bin/python3', sys.executable
assert sys.prefix == '/project/peilab/atst/nimloth/.venv', sys.prefix
plan=prepare_merge_plan(Path(sys.argv[1]),Path(sys.argv[2]),python_executable=Path(sys.argv[3]),merger_script=Path(sys.argv[4]))
assert plan['python_executable'] == sys.argv[3], plan['python_executable']
assert plan['command'][0] == sys.argv[3], plan['command'][0]
assert not Path(sys.argv[2]).exists()
print({'torch':torch.__version__,'accelerate':accelerate.__version__,'python_executable':plan['python_executable']})
PY
for code in 'import torch' 'import transformers' 'import vllm' 'import vagen.server.server' 'from experiments.training.sft1 import vagen_step60_collect'; do
  import_ok=0
  for attempt in 1 2 3; do
    echo "IMPORT_ATTEMPT=$attempt CODE=$code"
    if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$VWT:$NWT/src:$NWT" timeout 60s "$PY" -c "$code; print('IMPORT_OK')"; then
      import_ok=1
      break
    else
      rc=$?
      echo "IMPORT_RETRY_RC=$rc CODE=$code" >&2
    fi
  done
  test "$import_ok" -eq 1
done
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import importlib.metadata as m
assert {k: m.version(k) for k in ("vllm", "transformers", "torch")} == {"vllm":"0.8.2", "transformers":"4.49.0", "torch":"2.6.0"}
PY
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = 3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6

git -C "$ROOT" fetch origin refs/heads/task/rollout-vagen-step60-sft1-sft2:refs/remotes/origin/task/rollout-vagen-step60-sft1-sft2
test "$(git -C "$ROOT" rev-parse "$TASK_REF")" = "$APPROVED_DOCS_COMMIT"

# Complete live resource gate occurs immediately before run-root creation and submission.
module load slurm 2>/dev/null
RESOURCE_JSON=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT" python3 - <<'PY'
import json
from experiments.training.baseline.slurm_gpu_resources import parse_nodes
snapshot=[]
for row in parse_nodes():
    if row.partition!="normal": continue
    snapshot.append({"node":row.node,"state":row.state,"free_gpu":row.free_gpu,"free_cpu":row.free_cpu,"scheduler_free_mem_mb":(row.real_mem_mb or 0)-(row.alloc_mem_mb or 0),"observed_free_mem_mb":row.free_mem_mb})
snapshot.sort(key=lambda item:item["node"])
assert snapshot, "normal partition snapshot is empty"
print(json.dumps({"snapshot":snapshot,"selection":"scheduler queue; no pinned node","max_pending_seconds":86400},sort_keys=True))
PY
)
test ! -e "$RUN" && test ! -L "$RUN"

mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir "$RUN/logs" "$RUN/control" "$RUN/merge" "$RUN/smoke"
git -C "$ROOT" show "$APPROVED_DOCS_COMMIT:$TASK_DOC" > "$RUN/LAUNCH_CONTRACT.md"
cp "$RUN/LAUNCH_CONTRACT.md" "$RUN/RESOLVED_LAUNCH_CONTRACT.md"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" - "$RUN/metadata.json" "$APPROVED_DOCS_COMMIT" <<'PY'
import json, sys
payload={"format":"vagen_step60_merge_smoke_run_v1","purpose":"frozen step60 actor merge and one-row source-protocol smoke","nimloth_code_commit":"7dac687b733cccffaf0a211ef0a602ec001749dd","launch_contract_docs_commit":sys.argv[2],"vagen_reconstruction_commit":"170a673d1bf5855fc0ea6fbed0744b3d7168f8f0","checkpoint_component":"global_step_60/actor","trainable_modules":[],"objectives":[],"partition":"normal","gpus":4,"cpus":112,"memory":"96G","observed_free_memory_floor":"80GiB","max_pending":"24h","scheduler_completion_deadline_offset":"27h","walltime":"03:00:00","wandb":None,"resume":"validated partition/runtime/merged actor only; smoke partials are not resumable","validity":"one-row path smoke only"}
open(sys.argv[1],"x",encoding="utf-8").write(json.dumps(payload,indent=2)+"\n")
PY


# Persist the already-completed pre-root live resource decision without reselecting topology.
printf '%s\n' "$RESOURCE_JSON" > "$RUN/control/resources-immediately-before-sbatch.json"
SLURM_DEADLINE=$(date -u -d '+27 hours' +%Y-%m-%dT%H:%M:%S)
printf '%s\n' "$SLURM_DEADLINE" > "$RUN/control/slurm-completion-deadline"
# Three-hour TimeLimit plus this completion deadline prevents scheduler start after the 24-hour pending bound.
HOLD=$(sbatch --parsable --account=peilab --partition=normal --exclude=dgx-51 --nodes=1 --ntasks=1 --cpus-per-task=112 --gres=gpu:4 --mem=96G --time=03:00:00 --deadline="$SLURM_DEADLINE" --job-name=step60-b1-v3normalq12-smoke --output="$RUN/logs/hold_%j.out" --error="$RUN/logs/hold_%j.err" --wrap='sleep infinity')
printf '%s\n' "$HOLD" > "$RUN/control/hold_job_id"
state=
pending_deadline=$(( $(date +%s) + 86400 ))
while test "$(date +%s)" -lt "$pending_deadline"; do
  state=$(squeue -h -j "$HOLD" -o '%T' 2>/dev/null || true)
  if test -z "$state"; then state=$({ sacct -n -X -j "$HOLD" --format=State -P 2>/dev/null || true; } | awk -F'|' 'NR==1 {print $1}'); fi
  case "$state" in RUNNING) break;; COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) exit 1;; esac
  sleep 20
done
test "$state" = RUNNING
NODE=$(squeue -h -j "$HOLD" -o '%N')
test -n "$NODE" && test "$NODE" != '(null)'
printf '%s
' "$NODE" > "$RUN/control/node"
scontrol show job -dd "$HOLD" > "$RUN/control/scontrol-job.txt"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT" python3 - "$NODE" <<'PY' > "$RUN/control/allocated-node-memory.json"
import json,sys
from experiments.training.baseline.slurm_gpu_resources import parse_nodes
matches=[r for r in parse_nodes() if r.partition=="normal" and r.node==sys.argv[1]]
assert len(matches)==1
row=matches[0]
assert (row.free_mem_mb or 0)>=80*1024, f"allocated node observed free memory below 80GiB: {row.free_mem_mb} MiB"
print(json.dumps({"node":row.node,"observed_free_mem_mb":row.free_mem_mb,"required_min_mb":80*1024},sort_keys=True))
PY

# Revalidate mutable code/runtime identities after the queue wait and before expensive work.
test "$(git -C "$NWT" rev-parse HEAD)" = 7dac687b733cccffaf0a211ef0a602ec001749dd
test -z "$(git -C "$NWT" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$NWT/external/VAGEN" rev-parse HEAD)" = 9f1e89eb8c9839a406b6e62aa75703494a79e5b5
test "$(git -C "$NWT/external/VAGEN/verl" rev-parse HEAD)" = 494f264494b2525f2c13595f63ac4912963e6d2f
test "$(git -C "$NWT/external/le-wm" rev-parse HEAD)" = 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
for d in "$NWT/external/VAGEN" "$NWT/external/VAGEN/verl" "$NWT/external/le-wm"; do test -z "$(git -C "$d" status --porcelain=v1 --untracked-files=all)"; done
test "$(git -C "$VWT" rev-parse HEAD)" = 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0
test "$(git -C "$VWT" rev-parse HEAD^{tree})" = 58ef0eb66ad0bef7587c253c5c643af572c1d3a7
test "$(git -C "$VWT" --no-pager diff --binary --full-index --no-ext-diff 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD -- | sha256sum | awk '{print $1}')" = 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9
test -z "$(git -C "$VWT" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$ROOT" rev-parse "$TASK_REF")" = "$APPROVED_DOCS_COMMIT"
test "$(sha256sum "$NWT/external/VAGEN/verl/scripts/legacy_model_merger.py" | awk '{print $1}')" = 3e2794e1e9e566a4aeb0d709dad7d2b8864c8b91e4f72cf0d265ecb62c311044
test "$(sha256sum "$ROOT/.venv/lib/python3.10/site-packages/vllm/outputs.py" | awk '{print $1}')" = 047d469792ba4b332fd6bc6837af03340135cb49798e1ddfd2ffa730ead436f8
test "$(sha256sum "$ROOT/.venv/lib/python3.10/site-packages/vllm/engine/output_processor/stop_checker.py" | awk '{print $1}')" = 5ed39ad2df9912b7a4b9ff52168c50bfe9d937675d3f1122148c0824450afa28
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = 3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import importlib.metadata as m
assert {k:m.version(k) for k in ("vllm","transformers","torch")}=={"vllm":"0.8.2","transformers":"4.49.0","torch":"2.6.0"}
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/external/VAGEN/verl:$NWT/src:$NWT" "$PY" - "$NWT" <<'PY'
from pathlib import Path
import accelerate,sys,torch,torch.utils,transformers,verl,vllm
import verl.utils.fsdp_utils as fsdp_utils
root=Path(sys.argv[1]).resolve(); venv=root.parents[1]/'.venv'
assert sys.executable==str(venv/'bin/python3'),sys.executable
assert sys.prefix==str(venv),sys.prefix
for module in (accelerate,torch,transformers,vllm):
    assert Path(module.__file__).resolve().is_relative_to(venv),(module.__name__,module.__file__,venv)
for module in (verl,fsdp_utils):
    assert Path(module.__file__).resolve().is_relative_to(root/'external/VAGEN/verl'),(module.__name__,module.__file__)
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$VWT:$NWT/src:$NWT" timeout 180s "$PY" - "$VWT" <<'PY'
from pathlib import Path
import sys,vagen,vagen.server.server
from experiments.training.sft1 import vagen_step60_collect
expected=Path(sys.argv[1]).resolve()
assert Path(vagen.__file__).resolve().is_relative_to(expected),(vagen.__file__,expected)
print("POST_QUEUE_IMPORTS_OK")
PY

# Produce and validate CPU-side source/runtime evidence only after the allocation is secured.
cd "$NWT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_data.py --source "$SOURCE" --output "$RUN/partition"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_runtime_contract.py --runtime-root "$VWT" --expected-head 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0 --expected-tree 58ef0eb66ad0bef7587c253c5c643af572c1d3a7 --expected-diff-sha256 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9 --output "$RUN/runtime_contract.json"
test "$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/hash_vagen_step60_runtime_contract.py --contract "$RUN/runtime_contract.json")" = cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_checkpoint.py inspect-source --actor-dir "$ACTOR" --hash-shards | tee "$RUN/logs/inspect-source.log"


# Merge wrapper re-hashes source and validates HF architecture/tokenizer/finite weights/artifact manifest.
# Capture each pipeline component: R5 proved the Slurm child can finish 0:0 while the srun client returns nonzero.
set +e
srun --input=none --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" bash -lc "set -euo pipefail; cd '$NWT'; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$NWT/src:$NWT' '$PY' experiments/training/sft1/vagen_step60_checkpoint.py merge --actor-dir '$ACTOR' --target-dir '$RUN/merge/hf_actor' --python '$PY' --merger-script '$NWT/external/VAGEN/verl/scripts/legacy_model_merger.py' --hash-shards --execute" 2>&1 | tee "$RUN/logs/merge.log"
MERGE_PIPESTATUS=("${PIPESTATUS[@]}")
set -e
test "${#MERGE_PIPESTATUS[@]}" -eq 2
MERGE_SRUN_RC=${MERGE_PIPESTATUS[0]}
MERGE_TEE_RC=${MERGE_PIPESTATUS[1]}
printf 'srun=%s\ntee=%s\n' "$MERGE_SRUN_RC" "$MERGE_TEE_RC" > "$RUN/control/merge-pipeline-exit-codes"
test "$MERGE_TEE_RC" -eq 0

# The scheduler child and complete artifact binding are authoritative continuation gates.
merge_state=
for _ in $(seq 1 60); do
  sacct -j "$HOLD" --format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList -P > "$RUN/control/sacct-after-merge.txt"
  merge_state=$(awk -F'|' -v id="${HOLD}.0" '$1==id {print $3}' "$RUN/control/sacct-after-merge.txt" | tail -n1)
  case "$merge_state" in COMPLETED) break;; FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED) exit 1;; esac
  sleep 2
done
test "$merge_state" = COMPLETED
test "$(awk -F'|' -v id="${HOLD}.0" '$1==id {print $5}' "$RUN/control/sacct-after-merge.txt" | tail -n1)" = 0:0
cd "$NWT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_checkpoint.py validate-export --target-dir "$RUN/merge/hf_actor" > "$RUN/control/merge-validate-export.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" - "$RUN/merge/hf_actor" <<'PY'
import json,sys
from pathlib import Path
from experiments.training.sft1.vagen_step60_checkpoint import validate_merge_manifest
manifest=validate_merge_manifest(Path(sys.argv[1]),verify_artifacts=True)
print(json.dumps({"manifest_sha256":manifest["manifest_sha256"],"artifact_manifest_sha256":manifest["validation"]["artifact_manifest_sha256"]},sort_keys=True))
PY

cat > "$RUN/control/smoke-step.sh" <<'SMOKE'
set -euo pipefail
ROOT=/project/peilab/atst/nimloth
NWT=$ROOT/.worktree/rollout-vagen-step60-sft1-sft2
VWT=$ROOT/.worktree/vagen-step60-runtime-reconstruction-vagen
PY=$ROOT/.venv/bin/python3
RUN=$ROOT/outputs/experiments/training/sft1-vagen-step60/20260904T220000Z_step60_batch1_v3_normal_queue_r12_7dac687b
PORT=18640
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
test "${#ALLOCATED_GPUS[@]}" -eq 4
test "$(printf '%s\n' "${ALLOCATED_GPUS[@]}" | sort -u | wc -l)" -eq 4
POLICY_GPUS=${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}
ENV_GPUS=${ALLOCATED_GPUS[2]},${ALLOCATED_GPUS[3]}
printf 'SLURM_JOB_GPUS=%s\nCUDA_VISIBLE_DEVICES=%s\nPOLICY_GPUS=%s\nENV_GPUS=%s\n' "${SLURM_JOB_GPUS:-}" "$CUDA_VISIBLE_DEVICES" "$POLICY_GPUS" "$ENV_GPUS" > "$RUN/control/gpu-binding.txt"
nvidia-smi -L >> "$RUN/control/gpu-binding.txt"
export PATH="$ROOT/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$VWT:$NWT/src:$NWT" HF_HOME=/project/peilab/atst/.cache/huggingface TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface TORCH_HOME=/project/peilab/atst/flower/.cache/torch AI2THOR_HOME_ROOT=/project/peilab/atst/flower/.ai2thor-home
source "$NWT/experiments/training/baseline/setup_ai2thor_env.sh"
if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "port already in use" >&2; exit 2; fi
cleanup_server(){ set +e; if test -n "${SERVER_PID:-}"; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup_server EXIT
trap 'cleanup_server; trap - EXIT; exit 130' INT TERM
cd "$VWT"
CUDA_VISIBLE_DEVICES="$ENV_GPUS" "$PY" -m vagen.server.server server.host=127.0.0.1 server.port="$PORT" use_state_reward=False navigation.devices='[0,1]' navigation.max_workers=2 hydra.run.dir="$RUN/control/hydra-smoke" hydra.output_subdir=.hydra hydra.job.chdir=False > "$RUN/logs/env-smoke.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 167); do if curl -fsS "http://127.0.0.1:$PORT/health" > "$RUN/control/smoke-health.json"; then ready=1; break; fi; if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi; sleep 3; done
test "$ready" -eq 1
cd "$NWT"
CUDA_VISIBLE_DEVICES="$POLICY_GPUS" "$PY" experiments/training/sft1/vagen_step60_collect.py --model-path "$RUN/merge/hf_actor" --partition-manifest "$RUN/partition/partition_manifest.json" --source-index 0 --shard-index 0 --shard-size 100 --output-dir "$RUN/smoke/source-index-00000" --env-url "http://127.0.0.1:$PORT" --run-id step60-b1-v3normalq12-smoke-source-00000 --source-runtime-root "$VWT" --source-runtime-contract "$RUN/runtime_contract.json" --expected-reconstruction-head 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0 --expected-reconstruction-tree 58ef0eb66ad0bef7587c253c5c643af572c1d3a7 --expected-reconstruction-diff-sha256 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9 --expected-runtime-contract-payload-sha256 cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf --format-failure-policy fail_shard --concurrency 1 --tensor-parallel-size 2 --gpu-memory-utilization 0.35 --engine-seed 0 2>&1 | tee "$RUN/logs/smoke.log"
SMOKE
chmod 0555 "$RUN/control/smoke-step.sh"
set +e
srun --input=none --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" bash "$RUN/control/smoke-step.sh" 2>&1 | tee "$RUN/logs/smoke-step.log"
SMOKE_PIPESTATUS=("${PIPESTATUS[@]}")
set -e
test "${#SMOKE_PIPESTATUS[@]}" -eq 2
SMOKE_SRUN_RC=${SMOKE_PIPESTATUS[0]}
SMOKE_TEE_RC=${SMOKE_PIPESTATUS[1]}
printf 'srun=%s\ntee=%s\n' "$SMOKE_SRUN_RC" "$SMOKE_TEE_RC" > "$RUN/control/smoke-pipeline-exit-codes"
test "$SMOKE_TEE_RC" -eq 0
test "$SMOKE_SRUN_RC" -eq 0

cd "$NWT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" - "$RUN/smoke/source-index-00000" <<'PY'
import json, sys
from pathlib import Path
from experiments.training.sft1.vagen_step60_data import validate_complete_shard
shard=Path(sys.argv[1]); manifest=validate_complete_shard(shard, expected_source_indices={0})
assert manifest["source_indices"] == [0]
assert manifest["counts"]["records"] == manifest["counts"]["eligible"] == 1
assert manifest["counts"]["excluded"] == 0
assert manifest["counts"]["terminal_generations"] == 1
assert manifest["counts"]["terminal_environment_steps"] == 0
assert manifest["counts"]["transitions"] >= 1
row=json.loads((shard/"raw.jsonl").read_text(encoding="utf-8").strip())
assert len(row["observation_texts"]) == len(row["turns"]) + 1
assert len(row["image_artifacts"]) == len(row["turns"]) + 1
assert len({x["sha256"] for x in row["image_artifacts"]}) > 1
assert all(turn["generation_exclusion_reason"] is None for turn in row["turns"])
assert row["terminal_generation"]["generation_exclusion_reason"] is None
assert row["terminal_generation"]["executed"] is False
assert row["terminal_generation"]["environment_step_after_generation"] is False
print(json.dumps({"counts":manifest["counts"],"source_index":row["source_index"],"rewards":row["rewards"],"terminal_finish_reason":row["terminal_generation"]["finish_reason"],"terminal_stop_reason":row["terminal_generation"]["stop_reason"]},sort_keys=True))
PY
srun --input=none --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" nvidia-smi > "$RUN/control/nvidia-smi-final.txt"
```

The detached main script's outer trap cancels only its recorded hold, waits for terminal state and prints final `sacct` to the launcher log. Monitoring is read-only and independent; use the exact command below. It proves launcher PID state, binds scheduler queries to `control/hold_job_id`, and reads only this run's evidence. Monitoring never sends a signal or cancellation.

```bash
ssh superpod-csejzhang 'bash -l -s' <<'MONITOR'
set -euo pipefail
module load slurm 2>/dev/null
ROOT=/project/peilab/atst/nimloth
RUN=$ROOT/outputs/experiments/training/sft1-vagen-step60/20260904T220000Z_step60_batch1_v3_normal_queue_r12_7dac687b
PIDFILE=$ROOT/.local/tmp/step60-r12-launch-20260904T220000Z.pid
LOG=$ROOT/.local/tmp/step60-r12-launch-20260904T220000Z.log
test -f "$PIDFILE" && test ! -L "$PIDFILE"
pid=$(cat "$PIDFILE"); case "$pid" in ''|*[!0-9]*) exit 1;; esac
if kill -0 "$pid" 2>/dev/null; then echo "LAUNCHER=RUNNING PID=$pid"; else echo "LAUNCHER=EXITED PID=$pid"; fi
if test -f "$RUN/control/hold_job_id"; then
  job=$(cat "$RUN/control/hold_job_id"); case "$job" in ''|*[!0-9]*) exit 1;; esac
  squeue -j "$job" -o '%.18i %.28j %.10T %.10M %.20N %.8C %.12m %b'
  sacct -j "$job" --format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList,AllocTRES,MaxRSS -P
fi
tail -n 80 "$LOG"
find "$RUN/control" "$RUN/logs" "$RUN/smoke" -maxdepth 2 -type f -printf '%p %s\n' 2>/dev/null | sort
MONITOR
```

## Resume, cancellation and mandatory end record

- Partition/runtime contract and merged actor are reusable only after all committed manifests, bytes and load gates are revalidated. They are artifacts, not optimizer resume state.
- Smoke partials are retained and never resumed. A retry uses a fresh run root.
- Cancellation targets only the recorded hold ID with `scancel "$HOLD"`; no partial output is deleted.
- Completion, failure, cancellation or pause immediately triggers `on-experiment-end` with scheduler/runtime status, actual commands/commits, output, anomaly, validity and exact reuse/retry boundary.
- Passing smoke does not authorize the 100-row gate.
