# Design

## Fixed contract

- Host: `a100-1` (direct execution, no Slurm).
- Python: `/mnt/nimloth/venv/bin/python3`.
- Nimloth source: `/mnt/nimloth/.worktree/fix-sft1-termination-retrain` at `2a8ac7122c8e1f389e01cd360cb592cb9d308d80`.
- VAGEN source: `/mnt/nimloth/sources/vagen` at `844378ce8a5727d8274b0c7024573031f9b1296d`.
- Format-gate data: `/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T115012Z_success_only_termination_w8/data/format_eval.jsonl`.

## Checkpoints and outputs

- Epoch 5 adapter: `.../train/epoch_005`; output root: `/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T161800Z_epoch5_test120`.
- Epoch 6 adapter: `.../train/epoch_006`; output root: `/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T161800Z_epoch6_test120`.
- Each root contains `checkpoint/` for the exported full HF model, `eval/` for the unified evaluation artifact, and control logs/PIDs outside the eval directory.

## Evaluation parameters

Both arms use `--stage stage1 --eval-sets base common_sense --split test --episodes-per-eval-set 60 --seed-offset 1 --max-steps 20 --temperature 0 --top-p 1 --max-response-tokens 512 --tensor-parallel-size 2 --history-turns 5 --generation-seed 0 --success-threshold 1.5 --step-length 0.5 --max-model-len 32768 --gpu-memory-utilization 0.85 --max-pixels 100352 --vllm-mm-processor-cache-gb 0`.

## Resource isolation

- Export epoch 5 then epoch 6 sequentially with one visible spare GPU so checkpoint export chooses BF16; export itself remains the official module.
- Epoch 5: environment GPU 0 on port 18005; vLLM GPUs 1,2.
- Epoch 6: environment GPU 3 on port 18006; vLLM GPUs 4,5.
- Each environment has its own HOME and symlink to the verified existing AI2-THOR release cache. GPUs 6,7 remain free after export.
- Each arm calls only the repository `serve_environment.sh` and `run.sh` entrypoints. The controller shell only establishes process ownership, logs and cleanup.

## Failure handling

Format gate failure returns 2 and preserves evidence without starting episodes. Any service, export, vLLM or rollout failure stops that arm and preserves its log; it is not automatically retried. Cleanup terminates only recorded owned process groups. Partial summaries remain partial evidence.


## User-requested throughput revision

The user stopped the sequential evaluations and requested faster evaluation while
training continues. The code change is confined to early Stage 1/2 (and the shared
VAGEN direct protocol): `EvaluationConfig`/CLI expose positive
`--episode-concurrency` (default 1); the environment owner schedules a bounded set
of independent episodes and batches the existing VAGEN operations; the backbone
owner batches vLLM prompts, including independent Stage 2 query continuations.
Episode identity, seeds, prompts/history, raw termination validation, terminal
answers, and atomic record/resume rules remain unchanged. Concurrency is recorded
in the contract and cannot silently change on resume. No legacy planner runner,
training implementation, checkpoint, or real GPU launch is changed by this patch.
CPU tests compare serial/batched records, request count, mixed termination,
continuation budgets, and completed-record reuse; actual speed awaits remote use.

## Epoch7 protocol-weight continuation design boundary

Smallest gap: current loss weights only eight action-number tokens; expose independently configurable action-boundary/EOS weight through the established CLI/YAML path, default1 for compatibility. Use16 for both groups in this run. Preserve full-vocabulary CE and per-microbatch sum-of-weights normalization. Training loss.py owns weights, CLI/config own validation, trainer/checkpoint/export/eval own objective identity. Need inspect existing workflow for a safe committed epoch-boundary objective-change continuation before choosing the smallest interface. Strict same-objective resume remains strict. Do not copy/edit training_state to pretend compatibility. Preserve epoch7 model exactly; no semantic reinitialization or new LoRA-on-top of already-merged LoRA unless explicitly chosen. Main agent owns spec and launch; implementation agent owns source/tests/README.

## Latest user correction: existing resume only

User explicitly forbids a new entry and requests replacing weight-change rejection with warnings. This supersedes the prior separate-run/explicit-continuation design above. Use original workflow --resume with CLI action16 and boundary16 in the original run, loading epoch7 and preserving optimizer/scheduler/RNG/data cursor AND unweighted-LM convergence state. Only recognized loss-weight changes are warnings; stage/data/model/other hyperparameter mismatches remain errors. No new continuation option, phase offset, optimizer reset, new model initialization, or edited training_state. New checkpoints record actual weight identity; existing epoch checkpoints stay immutable. Workflow records the accepted parameter transition.

## Confirmed 2026-09-12: weighted-loss continuation after epoch12
The current user confirmation supersedes the former unweighted-only stopping design. Continue the same run through the same official --resume entry. Keep all trained parameters, BF16, weights16/16, optimizer/scheduler/RNG/data/cache. Explicit CLI monitor transition from unweighted LM to weighted LM reinitializes only convergence/best selection, using a fresh epoch12 validation as baseline. Subsequent same-monitor resumes retain that history. Weighted improvement below1% for2 consecutive epochs stops; sampled format at least31/32 means ready, otherwise terminal status explicitly says format unmet. No hard epoch ceiling. Preserve historical terminal evidence. Epoch12 is not altered. New sampling validation settings .7/.95/top_k0/max512/seed0 and raw metadata are already implemented. Do not reinterpret teacher-forced loss convergence as policy success; no automatic test eval.

## 2026-09-12 clarified: synchronous success rate after every epoch
User clarifies success evaluation belongs directly after EACH epoch, before the next epoch; previous separate-job resource question is superseded. Use a100-1 and existing full120test sampling config, no newentrypoint. Reuse loaded model via reusable HF RawGeneration adapter and shared run_direct_episodes, same strict invalid-noop andfull denominator. Saveepoch before eval; on epochresume finish missing measurement before next training (even if loss-stopped). Artifact-bound contract preventsmixing. Independent metric, not new stopping target. Allranks enter existing fullparam context; rank0 owns environments/inference and CPU Gloo control group communicates result/error before scope exit, so NCCL is not left pending through120episodes. Gloo timeout24h is controloperation bound, not convergence or experimenttime budget; existing HTTP500s bounds ordinary server waits. Parent owns verified existing envserver script+cleanup, no servercode fork. Restore RNG/modes before continuation. Preserve staticformat32 and bothlosses.

## 2026-09-12 distributed inline success acceleration
User stops slow rank0 evaluation and resumes training while acceleration is implemented. Protocol v2 assigns canonical120 identities by index modulo world; each loaded full-parameter rank runs its own disjoint subset and independent HF batches on its current GPU. Separate success-eval-concurrency defaults4 per rank, not format batch size. Existing server namespaces are UUIDs; deployment must budget world*concurrency environments. Each rank owns ranks/rank_NNN records/contracts/timings; rank0 aggregate reads original records without copying and rejects duplicate/foreign identities. Root binds world, assignments and exact checkpoint identity. New success_eval_distributed_v2 output leaves prior epoch15 partial untouched. Gloo gathers completion/errors from every rank before shared fullparameter context exits; no NCCL pending during episodes. Existing strict parsing, full120 denominator, sampling and training RNG restoration retained. Phase timings distinguish generation, environment operations and cleanup. No new executable or server lifecycle code.
