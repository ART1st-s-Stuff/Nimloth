# E0046: Recon must use the canonical merged SFT2 handoff

## What happened

New-SFT2 reconstruction cache job `479375` called PEFT 0.19
`PeftModel.from_pretrained(...)` after Torch distributed initialization. PEFT
unconditionally entered its tensor-parallel adapter path and failed before any
cache rows were written:

```text
ImportError: cannot import name 'EmbeddingParallel'
from transformers.integrations.tensor_parallel
```

A first workaround recreated the LoRA topology and called raw
`model.load_state_dict`. Distributed smoke `479384` completed mechanically but
reported `unexpected_keys=826`: none of the 826 saved adapter tensors were
restored because PEFT save keys omit the runtime adapter-name component. That
cache was declared invalid and never entered training.

## Why RL can load this SFT2

The current RL path does not hand the raw epoch PEFT directory to distributed
training. `prepare_k8_sft2_init.py` first snapshots a complete, stable epoch;
`prepare_k8_sft2_init.slurm` then merges all 826 adapters into a full HF model
and verifies k=8/inject metadata, query IDs and non-empty model shards. RL loads
that canonical merged model.

## Rule

Reconstruction must use the same validated snapshot+merge handoff as RL.

- Do not load a raw SFT2 PEFT epoch directory in a distributed recon job.
- Do not treat `strict=False` as successful adapter loading; any unexpected
  saved adapter key invalidates the representation.
- Require the immutable merged HF model plus its readiness/provenance record.
- Run a distributed cache smoke and compare it with the same merged model in a
  single-process smoke before full extraction.
