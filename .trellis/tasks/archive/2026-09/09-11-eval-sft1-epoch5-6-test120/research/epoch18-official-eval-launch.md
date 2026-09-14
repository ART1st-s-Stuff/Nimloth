# Research: epoch18 official evaluation launch

- Query: Evaluate epoch18 after training has advanced to epoch20, without rewriting history.
- Scope: internal
- Date: 2026-09-12

## Findings

Inspected source under `.worktree/fix-sft1-termination-retrain`.

- `src/nimloth/training/sft/stage1/trainer.py:1011`: --resume calls find_latest_resume_dir(output_dir), no historical checkpoint selector. Inline evaluation runs loaded resumed epoch and cannot independently choose epoch18. Do not alter latest checkpoint or run history to achieve selection.
- `src/nimloth/training/sft/stage1/checkpoint_export.py`: existing module exports exact adapter via --base-model, --adapter-dir, --out-dir; verifies adapter tensors and restores independently trained embeddings/head.
- `experiments/training/sft/evaluation/run.sh`: existing official standalone success entry.
- `src/nimloth/training/sft/evaluation/cli.py:34`: exact standalone flags below.
- `src/nimloth/training/sft/evaluation/README.md`: sampled format diagnostic below31/32 warns but does not block rollout. API is VAGEN844378c BatchEnvServer; no WM/value/MCTS. All120 denominator remains required.
- `experiments/training/sft/evaluation/serve_environment.sh`: ENV_DEVICES, ENV_PORT, ENV_MAX_WORKERS control renderer topology.
- `src/nimloth/training/sft/stage1/README.md:121`: inline v2 all8rank concurrency4 requires32 renderer instances and GPU preflight, unnecessary for isolated historical export.

Existing official command template (BASE, ADAPTER18, OUT, HELDOUT must be resolved from live original manifest/checkpoint):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=7 /mnt/nimloth/venv/bin/python3 -m nimloth.training.sft.stage1.checkpoint_export --base-model "$BASE" --adapter-dir "$ADAPTER18" --out-dir "$OUT/checkpoint"
PYTHON=/mnt/nimloth/venv/bin/python3 VAGEN_DIR=/mnt/nimloth/sources/vagen ENV_DEVICES='[0]' ENV_MAX_WORKERS=4 ENV_PORT=18018 HOME=/mnt/nimloth/env_home bash experiments/training/sft/evaluation/serve_environment.sh
PYTHON=/mnt/nimloth/venv/bin/python3 VAGEN_DIR=/mnt/nimloth/sources/vagen CUDA_VISIBLE_DEVICES=1,2 bash experiments/training/sft/evaluation/run.sh --stage stage1 --checkpoint "$OUT/checkpoint" --format-gate-jsonl "$HELDOUT" --env-url http://127.0.0.1:18018 --output-dir "$OUT/eval" --eval-sets base common_sense --split test --episodes-per-eval-set 60 --seed-offset 1 --max-steps 20 --temperature 0.7 --top-p 0.95 --max-response-tokens 512 --tensor-parallel-size 2 --episode-concurrency 4 --history-turns 5 --generation-seed 0 --success-threshold 1.5 --step-length 0.5 --max-pixels 100352
```

The recorded actual epoch12 launch used GPU0 renderer capacity4 and GPU1,2 TP2/concurrency4, maxmodel32768,memory0.85. This is the established topology, not a claim that4 is optimal. For speed, explicit --episode-concurrency16 and ENV_MAX_WORKERS16 with ENV_DEVICES='[0,3,4,5,6,7]' can exploit remaining GPUs after live capacity/reset preflight; current source supports batch generation and environments. This is a proposed resource override, not previously GPU validated. Preserve TP2 rather than assume arbitrary TP8 supported by model attention head counts. Do not co-reside with all8GPU training until memory availability is confirmed. Separate checkpoint output preserves all training progress.

## Related specs

`.trellis/spec/domains/reconstruction-and-evaluation.md`: identify checkpoint, dataset, provenance, actual completed denominator; model evaluation must use real environment evidence. `.trellis/workflow.md` research persistence and scoped tasks apply.

## External references

No external lookup required; interfaces verified in local current source. Runtime assumptions supplied by task progress: torch2.6.0, VAGEN844378c.

## Caveats / Not Found

No remote launch performed by researcher. Source supports configuration but higher concurrency memory/throughput needs live preflight. checkpoint_export chooses BF16 only when CUDA is visible; avoid hiding CUDA and accidentally getting FP32 export. CPU library memory keyword search returned unrelated legacy topic and was not used as evidence.
