# New host bootstrap handoff — 2026-09-05

## Scope and authorization boundary

- Human directed preparation of a new execution environment reachable as
  `ssh yt_ust_jiezhang_1@121.46.19.4 -p 8001 -i ~/.ssh/yt_ust_jiezhang_1.id`
  and identified remote `~/HDD_POOL` as the storage root.
- This bootstrap does not authorize a new rollout launch, checkpoint mutation,
  deletion, or cancellation of the still-pending old-cluster job.

## Verified new-cluster facts

- Remote home: `/HOME/yt_ust_jiezhang/yt_ust_jiezhang_1`.
- `~/HDD_POOL` resolves to
  `/XYAIFS00/HDD_POOL/yt_ust_jiezhang/yt_ust_jiezhang_1` on Lustre; the
  observed filesystem had about 394 TB available.
- Slurm cluster `tianhexy-ai` exposes `h100x` (11 nodes) and `hx` (2 nodes),
  each node with 8 H100 GPUs, 128 CPUs and about 2 TB RAM. The user association
  is account `yt_ust_jiezhang` with QOS names `normal`, `standby`, and
  `emergency`. No user jobs were present at the time of the query.
- Login node Python is 3.10.12 but lacks `ensurepip`; the system Anaconda module
  is available, but the login node could not reach `repo.anaconda.com`.

## Code bootstrap completed

- Local source snapshot is the committed task ref
  `69a0cef6be9b81837f296eadc4775df94c9bc19d`.
- Verified bundles were uploaded without overwriting an existing path to
  `~/HDD_POOL/nimloth/bootstrap/20260905-69a0cef6`:
  - Nimloth bundle SHA256 `2216e4e574e6b4b5255fbf5a91e963171f79ae31e1d4d11f270f27f864012cb1`;
  - VAGEN bundle SHA256 `0313ff69daaa93b7b250443f5f527985781edc916d8e097e1c99894af5b9db32`;
  - VERL bundle SHA256 `c12aa72ec5dd064f21e085e0bb26cfd388b98f36da7f1214326d3317a77f34fa`;
  - le-wm bundle SHA256 `bcfe90da520d7a84fc8532ec66c7e5ebd701c8e1dc108bf3f01c0dae7bbd4c87`.
- Clean bundle-backed repositories now exist at:
  - `~/HDD_POOL/nimloth/src/step60-69a0cef6`: Nimloth `69a0cef6`, embedded
    VAGEN `9f1e89eb`, embedded VERL `494f2644`, le-wm `8edfeb33`;
  - `~/HDD_POOL/nimloth/runtimes/vagen-step60-170a673d`: reconstruction VAGEN
    `170a673d` with its declared nested VERL `65316156`.
- These clones require no remote Git credentials; the human can replace bundle
  remotes after configuring Git.

## Environment attempts and remaining input transfer

- Failed, preserved environment prefix:
  `~/HDD_POOL/nimloth/envs/step60-vllm082-py310`; creation stopped because
  system Python lacks `ensurepip`.
- Failed, preserved conda prefix/log identity:
  `~/HDD_POOL/nimloth/envs/step60-vllm082-py310-conda`; conda stopped before
  package installation because the login node cannot reach Anaconda channels.
- The verified old executable environment is Python 3.10.12 / glibc 2.35,
  about 15 GB, with PyTorch 2.6.0, Transformers 4.49.0 and vLLM 0.8.2. The new
  host is also Ubuntu/glibc-compatible in principle, but binary compatibility
  still requires import and H100 smoke validation after copying.
- Source actor scope is about 19 GB: eight model shards, eight extra-state
  shards and tokenizer/config files. The train parquet is about 2.9 MB with
  SHA256 `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`.
- Copying the old `.venv` and actor through local `/tmp` is not yet authorized.
  No environment or checkpoint bytes have been relayed.

## Old-cluster collision gate

- Old Slurm job `548155` was freshly verified as `PENDING (Priority)`, elapsed
  `00:00:00`, with no node assigned and no run root. It has not been cancelled.
- Do not launch on the new cluster while `548155` can still start. Exact human
  approval is required to cancel that job and to relay the environment/model
  inputs to the new storage.

## `yhrun` 8-GPU resource probe — 2026-09-06

- Human explicitly requested trying `yhrun -N 1:1 -G 8`. The bounded probe
  added a 30-second immediate-allocation limit, a two-minute job limit, and an
  `nvidia-smi -L` payload that would exit immediately after reporting the
  allocation.
- The launcher exited before submission with
  `yhrun: error: "1:1" is not a valid node count`. No job ID was created, no
  queue wait or GPU allocation occurred, and there is no output or resumable
  state.
- A subsequent probe must use this cluster's accepted node-count syntax (for
  example, a single integer after verifying the intended semantics); this
  failed invocation does not authorize an automatic replacement run.
- Human then approved the corrected single-node probe. `yhrun -N 1 -G 8`
  first failed before submission because this account has no default
  partition. The same bounded probe was completed with the already verified
  scheduler identity: partition `h100x`, account `yt_ust_jiezhang`, QOS
  `normal`, 112 CPUs, 30-second immediate-allocation limit, and two-minute job
  limit.
- Slurm accepted that complete request but returned `Requested nodes are busy`;
  the immediate mode prevented it from entering the queue. A fresh `squeue`
  check showed no user jobs afterward. All 11 `h100x` nodes were either
  `mixed` or `draining`; no GPU was allocated to this probe and no job ID,
  output, metric, or resumable state exists.
