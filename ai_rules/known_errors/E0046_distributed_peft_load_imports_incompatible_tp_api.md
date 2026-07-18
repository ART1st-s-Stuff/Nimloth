# E0046: Distributed PEFT adapter load imports incompatible TP API

## Symptom

After `torch.distributed.init_process_group`, calling PEFT 0.19
`PeftModel.from_pretrained(...)` fails while loading a LoRA checkpoint:

```text
ImportError: cannot import name 'EmbeddingParallel'
from transformers.integrations.tensor_parallel
```

This occurred in reconstruction query-cache job `479375`. The preceding
single-GPU/non-distributed smoke succeeded, so a single-process loader test did
not expose it.

## Cause

PEFT 0.19 checks only whether Torch distributed is initialized before calling
`_maybe_shard_state_dict_for_tp`. That helper imports tensor-parallel classes
from a newer Transformers API before checking whether this model actually has
a TP plan. The installed Transformers lacks `EmbeddingParallel`.

## Rule

For Nimloth distributed cache/evaluation jobs, do not call
`PeftModel.from_pretrained` after process-group initialization. Recreate the
exact LoRA topology with `configure_qwen_tuning`, then load the saved adapter
state with the project checkpoint loader. Continue to require zero unexpected
adapter keys and exact vision-EMA key/shape matching.

A distributed smoke is required for future loader changes; a single-process
smoke is insufficient for this failure mode.
