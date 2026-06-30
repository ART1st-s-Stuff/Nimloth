#!/usr/bin/env bash
# Shared vLLM rollout Hydra overrides for VAGEN navigation baseline.
# Prefer over sglang for training (see project memory on sglang issues).

VAGEN_ROLLOUT_VLLM_ARGS=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_num_batched_tokens=24000
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.multi_turn.enable=True
)
