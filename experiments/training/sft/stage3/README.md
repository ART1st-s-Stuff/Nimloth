# Stage3 data preparation

`prepare_eval_records.py` consumes a hash-pinned VAGEN raw-record manifest and verifies
record bytes, each image's hash/mode/size, transcript/history alignment, executed actions,
rewards and outcomes. Image paths are relative to their raw record file when not absolute.
It uses the existing source action conversion followed by the unchanged B prompt rewrite
in `prompt_conversion.py`. The shared `rollout.tail_drop` converter owns finite-horizon
returns and removing the final transition with no retained next observation.

```bash
python -m experiments.training.sft.stage3.prepare_eval_records \
  --manifest records_manifest.json --latent-token-count 64 \
  --max-action-horizon 20 --check-only
```

Replace `--check-only` with `--output-root NEW_DIRECTORY` to write `data.jsonl`,
`manifest.json` and `COMMITTED`. The output directory must not exist. Failure writes
rejection evidence without a completion marker. The manifest preserves the original raw
record hashes and every verified image identity; record and image sources are not modified.

`prepare_transition_cache.py` builds one split with the production CPU cache API,
without loading Qwen weights. Specify source/output/processor, token count, max
length and max pixels explicitly. It refuses existing outputs and never filters
failed trajectories. Run separately for `preprocess/train` and `preprocess/val`.
