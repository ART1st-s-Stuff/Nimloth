# Historical SFT1 format comparison (read-only audit)

Historical source: f65ed859 (Formal39 lineage), also checked b066dc3b. File: src/nimloth/backbone/qwen25vl/turn_generation.py.
Historical production backend calls run_fsdp_greedy_turn_probe, which applies apply_turn_response_logits before argmax.
The prompt prefills <think>; reasoning closure can be forced at its budget; latent/action-start tokens are forced; action choice is restricted to 8 IDs; action-end is forced. Therefore historical exact-parser 4/4 is constrained-generation protocol evidence, not unconstrained format learning.
Task 08-29-train-sft1-query-state/progress.md records Formal39 step0 and later update6420/terminal 4/4. The initialization was action-repaired ID176 and original query-state training adapted full language/head using 2*DINO+LM. Later visual-only fork froze the language path. These stages cannot be conflated with current original-hf_actor LoRA pure-answer training.
Current source f62caf7e stage1/trainer.py:116-177 uses plain greedy generate(max_new_tokens=128), without the historical logits constraints, and regex-checks the decoded continuation. Epoch1 0/32 was verified in previous live query. No causal claim that precision alone explains the difference.
Superpod live artifact refresh failed with SSH channel connect timeout; no remote mutations or retries. Conclusions above are from locally preserved Git source and experiment records, not fresh remote artifact verification.

## Exact historical constraint code (original lines 412-453)

```python
def allowed_turn_token_ids(
    output_ids: Sequence[int],
    spec: TurnGenerationSpec,
    *,
    decoded_close_end: int | None = None,
) -> tuple[int, ...] | None:
    """Return the constrained next-token set; ``None`` means reasoning vocab."""

    if decoded_close_end is not None:
        if not 1 <= decoded_close_end <= len(output_ids):
            raise ValueError("decoded close boundary is outside generated output")
        close_end = decoded_close_end
    else:
        close_start = find_token_subsequence(output_ids, spec.close_token_ids)
        close_end = (
            close_start + len(spec.close_token_ids)
            if close_start is not None
            else None
        )
    if close_end is None:
        if len(output_ids) < spec.max_reasoning_tokens:
            return None
        matched = _close_prefix_length(output_ids, spec.close_token_ids)
        return (spec.close_token_ids[matched],)

    suffix = output_ids[close_end:]
    if len(suffix) < len(spec.injected_token_ids):
        expected = spec.injected_token_ids[len(suffix)]
        if tuple(suffix) != spec.injected_token_ids[: len(suffix)]:
            raise ValueError("generated turn diverged from injected token prefix")
        return (expected,)
    if tuple(suffix[: len(spec.injected_token_ids)]) != spec.injected_token_ids:
        raise ValueError("generated turn has invalid injected token prefix")
    action_suffix = suffix[len(spec.injected_token_ids) :]
    if not action_suffix:
        return spec.action_token_ids
    if len(action_suffix) == 1:
        if action_suffix[0] not in spec.action_token_ids:
            raise ValueError("generated turn has an invalid action token")
        return (spec.action_end_token_id,)
    raise ValueError("generation continued after the action end boundary")

```
