# Exact venv-safe R4 launch contract — step60 actor merge + one-row GPU smoke

Date: 2026-09-02
Status: **candidate exact contract; not launch authorization**

This contract replaces terminal failed job `543910` with W-014 code commit `7dac687b733cccffaf0a211ef0a602ec001749dd` and a new run identity. Remote inert planning retained `/project/peilab/atst/nimloth/.venv/bin/python3` in both provenance and `command[0]`; that interpreter loaded Torch `2.6.0+cu124`, `torch.utils`, Accelerate `1.14.0` and the full legacy merger module. The plan SHA256 is `00198dc3116da488129a6b3cb88391de6a5d588e79ce8459e1be00b5ae748700`. The approved R3 contract passed the inert merge preflight but one isolated `vagen.server.server` import exited before `IMPORT_OK`; audit proved no run root or matching Slurm job, while three immediate identical reproductions all returned `IMPORT_OK`/0. R4 therefore adds only a bounded three-attempt policy to each already reviewed side-effect-free import gate and uses a distinct run root, preflight target, port and job name. All real merge/smoke semantics remain unchanged. It authorizes nothing until exact commit/push and separate launch approval. Scope remains actor merge/load plus source-index-0 smoke only; the 100-row gate and all later stages are excluded.

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

Unique run root, absent at 2026-09-02 preflight:

`/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260902T160000Z_step60_batch1_v3_venv_r4_7dac687b`

Allowed children: `partition/`, `runtime_contract.json`, `merge/hf_actor/`, `smoke/source-index-00000/`, failed `smoke/source-index-00000.partial-<12 hex>/`, `logs/`, `control/`, `metadata.json`, `LAUNCH_CONTRACT.md`, `RESOLVED_LAUNCH_CONTRACT.md`, `README.md`, and `END.json`. Existing paths are never reused or removed. W&B is disabled.

Slurm: account `peilab`, partition `normal`, any healthy single node, one task, four GPUs, 112 CPUs, 256 GiB scheduler memory, walltime `03:00:00`. The one smoke step receives all four GPUs and fails unless its initial `CUDA_VISIBLE_DEVICES` resolves to four distinct entries. Policy is restricted to logical `0,1` with TP2; service is restricted to logical `2,3`, exposing its two devices internally as `[0,1]`, `max_workers=2`. The 2026-09-02 pre-contract snapshot found `dgx-14` and `dgx-35` satisfying the four-GPU/112-CPU/256-GiB gates; this is transient evidence only. `dgx-51` is excluded because two prior navigation runs passed HTTP health but timed out the first real AI2-THOR prewarm, and it has not been requalified. Availability and requested CPU/memory must be rechecked immediately before submission. Expected duration is under two hours, including repeated source hashing, merge and startup margin.

## Exact execution script

Immediately before the remote script, run and enforce the repository resource query:

```bash
bash /workspace/remote2/nimloth/.local/scripts/query-resources.sh --partition normal --min-free-gpu 4
```

Then this entire script runs through `ssh superpod-csejzhang 'bash -l -s -- <EXACT_APPROVED_DOCS_COMMIT>'` in one login shell after exact experiment-launch approval, replacing the displayed angle-bracket argument with the literal pushed docs commit from that approval receipt. Inside the script `$1` is mandatory and hash-checked before use. Any error after `HOLD` is assigned triggers cancellation and proves a terminal Slurm state. The metadata copy is read from the hash-checked approved task ref, never from a mutable working tree.

```bash
set -euo pipefail
ROOT=/project/peilab/atst/nimloth
NWT=$ROOT/.worktree/rollout-vagen-step60-sft1-sft2
VWT=$ROOT/.worktree/vagen-step60-runtime-reconstruction-vagen
PY=$ROOT/.venv/bin/python3
RUN=$ROOT/outputs/experiments/training/sft1-vagen-step60/20260902T160000Z_step60_batch1_v3_venv_r4_7dac687b
ACTOR=/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60/actor
SOURCE=/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/train.parquet
TASK_REF=refs/remotes/origin/task/rollout-vagen-step60-sft1-sft2
TASK_DOC=.trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2/research/exact-merge-smoke-launch-contract-venv-r4-2026-09-02.md
APPROVED_DOCS_COMMIT=${1:?missing-approved-docs-commit}
HOLD=
NODE=
cleanup_hold() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$HOLD"; then
    scancel "$HOLD" 2>/dev/null || true
    terminal=0
    state=
    for _ in $(seq 1 60); do
      state=$(sacct -n -X -j "$HOLD" --format=State -P | awk -F'|' 'NR==1 {value=$1} END {print value}')
      case "$state" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) terminal=1; break;; esac
      sleep 2
    done
    sacct -j "$HOLD" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList,AllocTRES,MaxRSS -P || true
    if test "$terminal" -ne 1; then echo "hold failed to reach terminal state: ${state:-missing}" >&2; rc=1; fi
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
PREFLIGHT_TARGET=$ROOT/.local/tmp/step60-r4-inert-target-20260902T160000Z-7dac687b
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
test ! -e "$RUN" && test ! -L "$RUN"

git -C "$ROOT" fetch origin refs/heads/task/rollout-vagen-step60-sft1-sft2:refs/remotes/origin/task/rollout-vagen-step60-sft1-sft2
test "$(git -C "$ROOT" rev-parse "$TASK_REF")" = "$APPROVED_DOCS_COMMIT"
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir "$RUN/logs" "$RUN/control" "$RUN/merge" "$RUN/smoke"
git -C "$ROOT" show "$APPROVED_DOCS_COMMIT:$TASK_DOC" > "$RUN/LAUNCH_CONTRACT.md"
cp "$RUN/LAUNCH_CONTRACT.md" "$RUN/RESOLVED_LAUNCH_CONTRACT.md"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" - "$RUN/metadata.json" "$APPROVED_DOCS_COMMIT" <<'PY'
import json, sys
payload={"format":"vagen_step60_merge_smoke_run_v1","purpose":"frozen step60 actor merge and one-row source-protocol smoke","nimloth_code_commit":"7dac687b733cccffaf0a211ef0a602ec001749dd","launch_contract_docs_commit":sys.argv[2],"vagen_reconstruction_commit":"170a673d1bf5855fc0ea6fbed0744b3d7168f8f0","checkpoint_component":"global_step_60/actor","trainable_modules":[],"objectives":[],"partition":"normal","gpus":4,"cpus":112,"memory":"256G","walltime":"03:00:00","wandb":None,"resume":"validated partition/runtime/merged actor only; smoke partials are not resumable","validity":"one-row path smoke only"}
open(sys.argv[1],"x",encoding="utf-8").write(json.dumps(payload,indent=2)+"\n")
PY

cd "$NWT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_data.py --source "$SOURCE" --output "$RUN/partition"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_runtime_contract.py --runtime-root "$VWT" --expected-head 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0 --expected-tree 58ef0eb66ad0bef7587c253c5c643af572c1d3a7 --expected-diff-sha256 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9 --output "$RUN/runtime_contract.json"
test "$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/hash_vagen_step60_runtime_contract.py --contract "$RUN/runtime_contract.json")" = cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT/src:$NWT" "$PY" experiments/training/sft1/vagen_step60_checkpoint.py inspect-source --actor-dir "$ACTOR" --hash-shards | tee "$RUN/logs/inspect-source.log"

# Recheck the same repository resource surface again immediately before sbatch.
module load slurm 2>/dev/null
python3 "$NWT/experiments/training/baseline/slurm_gpu_resources.py" --partition normal --min-free-gpu 4 | tee "$RUN/control/resources-immediately-before-sbatch.txt"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NWT" python3 - <<'PY' > "$RUN/control/eligible-nodes-immediately-before-sbatch.json"
import json
from experiments.training.baseline.slurm_gpu_resources import parse_nodes
eligible=[]
for row in parse_nodes():
    scheduler_free=(row.real_mem_mb or 0)-(row.alloc_mem_mb or 0)
    if row.partition=="normal" and row.node!="dgx-51" and row.state in {"IDLE","MIXED"} and row.free_gpu>=4 and row.free_cpu>=112 and scheduler_free>=256*1024 and (row.free_mem_mb or 0)>=256*1024:
        eligible.append({"node":row.node,"free_gpu":row.free_gpu,"free_cpu":row.free_cpu,"scheduler_free_mem_mb":scheduler_free,"observed_free_mem_mb":row.free_mem_mb})
eligible.sort(key=lambda item:(-item["free_gpu"],-item["free_cpu"],-item["observed_free_mem_mb"],item["node"]))
assert eligible, "no one-node normal topology satisfies 4 GPU / 112 CPU / 256G scheduler+observed memory"
print(json.dumps(eligible,sort_keys=True))
PY
ELIGIBLE_NODE=$("$PY" - "$RUN/control/eligible-nodes-immediately-before-sbatch.json" <<'PY'
import json,sys
items=json.load(open(sys.argv[1],encoding="utf-8"))
assert items
print(items[0]["node"])
PY
)
test -n "$ELIGIBLE_NODE"
printf '%s\n' "$ELIGIBLE_NODE" > "$RUN/control/eligible-node"
HOLD=$(sbatch --parsable --account=peilab --partition=normal --nodelist="$ELIGIBLE_NODE" --nodes=1 --ntasks=1 --cpus-per-task=112 --gres=gpu:4 --mem=256G --time=03:00:00 --job-name=step60-b1-v3venv4-smoke --output="$RUN/logs/hold_%j.out" --error="$RUN/logs/hold_%j.err" --wrap='sleep infinity')
printf '%s\n' "$HOLD" > "$RUN/control/hold_job_id"
for _ in $(seq 1 180); do
  state=$(squeue -h -j "$HOLD" -o '%T')
  case "$state" in RUNNING) break;; FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) exit 1;; esac
  sleep 2
done
test "$(squeue -h -j "$HOLD" -o '%T')" = RUNNING
NODE=$(squeue -h -j "$HOLD" -o '%N')
test -n "$NODE" && test "$NODE" != '(null)'
test "$NODE" = "$ELIGIBLE_NODE"
printf '%s\n' "$NODE" > "$RUN/control/node"
scontrol show job -dd "$HOLD" > "$RUN/control/scontrol-job.txt"

# Merge wrapper re-hashes source by design and validates HF architecture/tokenizer/finite weights/artifact manifest.
srun --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" bash -lc "set -euo pipefail; cd '$NWT'; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$NWT/src:$NWT' '$PY' experiments/training/sft1/vagen_step60_checkpoint.py merge --actor-dir '$ACTOR' --target-dir '$RUN/merge/hf_actor' --python '$PY' --merger-script '$NWT/external/VAGEN/verl/scripts/legacy_model_merger.py' --hash-shards --execute" 2>&1 | tee "$RUN/logs/merge.log"

srun --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" bash -s <<'SMOKE'
set -euo pipefail
ROOT=/project/peilab/atst/nimloth
NWT=$ROOT/.worktree/rollout-vagen-step60-sft1-sft2
VWT=$ROOT/.worktree/vagen-step60-runtime-reconstruction-vagen
PY=$ROOT/.venv/bin/python3
RUN=$ROOT/outputs/experiments/training/sft1-vagen-step60/20260902T160000Z_step60_batch1_v3_venv_r4_7dac687b
PORT=18560
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
CUDA_VISIBLE_DEVICES="$POLICY_GPUS" "$PY" experiments/training/sft1/vagen_step60_collect.py --model-path "$RUN/merge/hf_actor" --partition-manifest "$RUN/partition/partition_manifest.json" --source-index 0 --shard-index 0 --shard-size 100 --output-dir "$RUN/smoke/source-index-00000" --env-url "http://127.0.0.1:$PORT" --run-id step60-b1-v3venv4-smoke-source-00000 --source-runtime-root "$VWT" --source-runtime-contract "$RUN/runtime_contract.json" --expected-reconstruction-head 170a673d1bf5855fc0ea6fbed0744b3d7168f8f0 --expected-reconstruction-tree 58ef0eb66ad0bef7587c253c5c643af572c1d3a7 --expected-reconstruction-diff-sha256 7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9 --expected-runtime-contract-payload-sha256 cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf --format-failure-policy fail_shard --concurrency 1 --tensor-parallel-size 2 --gpu-memory-utilization 0.35 --engine-seed 0 2>&1 | tee "$RUN/logs/smoke.log"
SMOKE

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
srun --jobid="$HOLD" --overlap --nodes=1 --ntasks=1 -w "$NODE" nvidia-smi > "$RUN/control/nvidia-smi-final.txt"
```

The outer trap then cancels the hold, waits for a terminal state and prints final `sacct`. During execution the session records `squeue -j "$HOLD"`, `sacct`, `scontrol`, merge/service/smoke logs, service health and node `nvidia-smi`; it remains attached until success or terminal failure.

## Resume, cancellation and mandatory end record

- Partition/runtime contract and merged actor are reusable only after all committed manifests, bytes and load gates are revalidated. They are artifacts, not optimizer resume state.
- Smoke partials are retained and never resumed. A retry uses a fresh run root.
- Cancellation targets only the recorded hold ID with `scancel "$HOLD"`; no partial output is deleted.
- Completion, failure, cancellation or pause immediately triggers `on-experiment-end` with scheduler/runtime status, actual commands/commits, output, anomaly, validity and exact reuse/retry boundary.
- Passing smoke does not authorize the 100-row gate.
