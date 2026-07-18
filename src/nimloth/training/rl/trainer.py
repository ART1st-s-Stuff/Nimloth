"""Online RL training loop: rollout → encode → train → repeat.

Qwen model loading is handled inside ``train_rl`` via
``configure_qwen_tuning`` (supports LLM freeze/lora/full +
vision freeze/lora/full).  Resume from a previous RL checkpoint
(``--resume``) reloads the Qwen model, WM heads, and optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen_tuning import (
    configure_qwen_tuning,
    resolve_tune_modes,
    uses_lora,
)
from nimloth.backbone.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.rl.checkpoint import (
    load_lora_adapter_state,
    load_rl_wm_checkpoint,
    save_rl_checkpoint,
)
from nimloth.training.rl.loss import compute_predictor_loss, compute_value_loss
from nimloth.training.rl.rollout import (
    RolloutCollector,
    RolloutTrajectory,
    validate_rl_policy_protocol,
    validate_rollout_trajectory,
)
from nimloth.wm.dataset import discounted_action_value_targets
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


# ---------------------------------------------------------------------------
# Latent encoding (Qwen → hidden states)
# ---------------------------------------------------------------------------


def encode_trajectory_hiddens(
    trajectory: RolloutTrajectory,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    latent_token_count: int = 1,
    history_window: int = 112,
) -> list[torch.Tensor]:
    """Replay each real assistant turn and return its latent query block.

    There is one query state per generated action response.  The terminal image
    has no assistant response in VAGEN and therefore no fabricated query state.
    """
    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_latent_state_block,
        find_last_latent_state_block,
        last_hidden_state,
    )
    from nimloth.training.rl.rollout import (
        build_nimloth_policy_messages,
        multimodal_policy_messages,
    )

    if trajectory.latent_token_count != latent_token_count:
        raise ValueError(
            f"trajectory latent_token_count={trajectory.latent_token_count} but "
            f"runtime requested {latent_token_count}"
        )
    states: list[torch.Tensor] = []
    tokens = LatentActionTokens()

    for index, current_response in enumerate(trajectory.assistant_responses):
        messages, images = build_nimloth_policy_messages(
            trajectory.image_paths[: index + 1],
            trajectory.system_prompt,
            trajectory.observation_texts[: index + 1],
            trajectory.assistant_responses[:index],
            history_window=history_window,
        )
        messages.append({"role": "assistant", "content": current_response})
        messages, images = multimodal_policy_messages(messages, images)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            output = qwen_model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = last_hidden_state(output)
        latent_indices = find_last_latent_state_block(
            enc["input_ids"][0],
            token_id_map,
            tokens,
            latent_token_count=latent_token_count,
        )
        latent = extract_latent_state_block(
            hidden[0:1], latent_indices
        )  # (k, hidden_dim)
        states.append(latent.detach().cpu())

    return states


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def build_rl_transitions(
    trajectories: list[RolloutTrajectory],
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    gamma: float = 1.0,
    *,
    latent_token_count: int = 1,
    history_window: int = 112,
    use_kl_in_reward: bool = False,
    kl_reward_coef: float = 0.0,
    kl_penalty_type: str = "kl",
) -> list[dict[str, Any]]:
    """Encode trajectories → list of transition dicts (CPU tensors)."""

    transitions: list[dict[str, Any]] = []
    for traj in trajectories:
        hiddens = encode_trajectory_hiddens(
            traj,
            qwen_model,
            processor,
            token_id_map,
            device,
            latent_token_count=latent_token_count,
            history_window=history_window,
        )
        if len(hiddens) != traj.num_steps:
            raise RuntimeError(
                f"trajectory {traj.record_id}: encoded {len(hiddens)} states for "
                f"{traj.num_steps} actions"
            )

        record = traj.to_record()
        kl_reward_penalties = [0.0] * traj.num_steps
        if use_kl_in_reward:
            if kl_reward_coef <= 0:
                raise ValueError(
                    f"KL-in-reward requires positive coefficient, got {kl_reward_coef}"
                )
            if len(traj.reference_token_log_probs) != traj.num_steps:
                raise ValueError(
                    f"trajectory {traj.record_id} has no complete reference token log-probs"
                )
            from nimloth.training.rl.loss import compute_kl_penalty

            adjusted_rewards: list[float] = []
            for step in range(traj.num_steps):
                if len(traj.ppo_old_token_log_probs) != traj.num_steps:
                    raise ValueError(
                        f"trajectory {traj.record_id} has no complete PPO-old log-probs"
                    )
                behavior = torch.tensor(
                    traj.ppo_old_token_log_probs[step], dtype=torch.float32
                )
                reference = torch.tensor(
                    traj.reference_token_log_probs[step], dtype=torch.float32
                )
                penalty = float(compute_kl_penalty(
                    behavior,
                    reference,
                    penalty_type=kl_penalty_type,
                ).sum().item())
                kl_reward_penalties[step] = kl_reward_coef * penalty
                adjusted_rewards.append(
                    float(traj.step_rewards[step]) - kl_reward_penalties[step]
                )
            record["step_rewards"] = adjusted_rewards
            record["reward"] = sum(adjusted_rewards) + float(traj.final_reward)
        value_targets = discounted_action_value_targets(record, gamma=gamma)

        for t in range(traj.num_steps):
            action_log_probs = traj.action_log_probs[t]
            old_action_log_prob = float(
                action_log_probs[traj.action_indices[t]]
            )
            behavior_token_log_probs = [
                *[float(value) for value in traj.thought_token_log_probs[t]],
                old_action_log_prob,
            ]
            old_token_log_probs = (
                [float(value) for value in traj.ppo_old_token_log_probs[t]]
                if traj.ppo_old_token_log_probs
                else behavior_token_log_probs
            )

            from nimloth.training.rl.vagen_protocol import thought_from_assistant_response

            transitions.append({
                "qwen_hidden_current": hiddens[t],
                "qwen_hidden_next": hiddens[t + 1] if t + 1 < len(hiddens) else None,
                "action_index": torch.tensor(traj.action_indices[t], dtype=torch.long),
                "value_target": torch.tensor(value_targets[t], dtype=torch.float32),
                "old_token_log_probs": old_token_log_probs,
                "behavior_token_log_probs": behavior_token_log_probs,
                "thought_token_ids": list(traj.thought_token_ids[t]),
                "reference_token_log_probs": (
                    list(traj.reference_token_log_probs[t])
                    if traj.reference_token_log_probs else []
                ),
                "old_token_values": (
                    list(traj.critic_token_values[t])
                    if traj.critic_token_values else []
                ),
                "token_advantages": [],
                "token_returns": [],
                "kl_reward_penalty": kl_reward_penalties[t],
                "system_prompt": traj.system_prompt,
                "observation_texts": traj.observation_texts[:t + 1],
                "assistant_responses": traj.assistant_responses[:t],
                "current_thought": thought_from_assistant_response(
                    traj.assistant_responses[t]
                ),
                "image_history_paths": traj.image_paths[:t + 1],
            })

    return transitions


def normalize_turn_advantages_for_policy_tokens(
    raw_advantages: torch.Tensor,
    token_counts: list[int],
) -> torch.Tensor:
    """Whiten turn advantages with VAGEN's response-token loss-mask weighting."""

    if raw_advantages.ndim != 1 or raw_advantages.numel() != len(token_counts):
        raise ValueError(
            "turn advantage/token count mismatch: "
            f"advantages={tuple(raw_advantages.shape)}, counts={token_counts}"
        )
    if any(count <= 0 for count in token_counts):
        raise ValueError(f"policy token counts must be positive, got {token_counts}")
    weighted = torch.cat([
        advantage.reshape(1).expand(count)
        for advantage, count in zip(raw_advantages, token_counts)
    ])
    return (
        raw_advantages - weighted.mean()
    ) / (weighted.std(unbiased=False) + 1e-8)


def assign_masked_gae_to_transitions(
    trajectories: list[RolloutTrajectory],
    transitions: list[dict[str, Any]],
    *,
    gamma: float,
    lam: float,
    reward_placement: str,
) -> None:
    """Compute VAGEN token-level masked GAE and attach it to transitions."""

    from nimloth.training.rl.loss import compute_masked_gae_advantage_return

    if reward_placement not in {"final", "per_turn"}:
        raise ValueError(
            f"masked_gae reward_placement must be final or per_turn, got {reward_placement!r}"
        )
    per_trajectory_values: list[list[float]] = []
    per_trajectory_rewards: list[list[float]] = []
    token_counts_by_trajectory: list[list[int]] = []
    for trajectory in trajectories:
        if len(trajectory.critic_token_values) != trajectory.num_steps:
            raise RuntimeError(
                f"trajectory {trajectory.record_id} has no complete critic values"
            )
        counts = [len(values) for values in trajectory.critic_token_values]
        values = [value for step_values in trajectory.critic_token_values for value in step_values]
        rewards = [0.0] * len(values)
        cursor = 0
        if reward_placement == "per_turn":
            for step, count in enumerate(counts):
                reward = float(trajectory.step_rewards[step])
                if step == trajectory.num_steps - 1:
                    reward += float(trajectory.final_reward)
                rewards[cursor + count - 1] = reward
                cursor += count
        else:
            rewards[-1] = float(trajectory.reward)
        per_trajectory_values.append(values)
        per_trajectory_rewards.append(rewards)
        token_counts_by_trajectory.append(counts)

    max_tokens = max(len(values) for values in per_trajectory_values)
    values_tensor = torch.zeros((len(trajectories), max_tokens), dtype=torch.float32)
    rewards_tensor = torch.zeros_like(values_tensor)
    loss_mask = torch.zeros_like(values_tensor, dtype=torch.long)
    for row, (values, rewards) in enumerate(
        zip(per_trajectory_values, per_trajectory_rewards)
    ):
        length = len(values)
        values_tensor[row, :length] = torch.tensor(values)
        rewards_tensor[row, :length] = torch.tensor(rewards)
        loss_mask[row, :length] = 1
    advantages, returns = compute_masked_gae_advantage_return(
        token_level_rewards=rewards_tensor,
        values=values_tensor,
        loss_mask=loss_mask,
        gamma=gamma,
        lam=lam,
    )

    transition_cursor = 0
    for row, counts in enumerate(token_counts_by_trajectory):
        token_cursor = 0
        for count in counts:
            transition = transitions[transition_cursor]
            transition["token_advantages"] = advantages[
                row, token_cursor:token_cursor + count
            ].tolist()
            transition["token_returns"] = returns[
                row, token_cursor:token_cursor + count
            ].tolist()
            if len(transition["old_token_values"]) != count:
                raise RuntimeError("transition critic token count changed during GAE")
            token_cursor += count
            transition_cursor += 1
    if transition_cursor != len(transitions):
        raise RuntimeError(
            f"masked GAE assigned {transition_cursor} of {len(transitions)} transitions"
        )


def deterministic_transition_microbatches(
    num_transitions: int,
    microbatch_size: int,
    *,
    seed: int,
) -> list[list[int]]:
    """Shuffle once and partition every transition exactly once."""

    if num_transitions < 0:
        raise ValueError(f"num_transitions must be >= 0, got {num_transitions}")
    if microbatch_size <= 0:
        raise ValueError(f"microbatch_size must be > 0, got {microbatch_size}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(num_transitions, generator=generator).tolist()
    return [
        indices[start:start + microbatch_size]
        for start in range(0, num_transitions, microbatch_size)
    ]


def summarize_validation_trajectories(
    trajectories: list[RolloutTrajectory],
    *,
    expected_episodes: int,
) -> dict[str, float]:
    """Summarize a strict fixed-task heldout evaluation."""

    if len(trajectories) != int(expected_episodes):
        raise RuntimeError(
            "heldout evaluation returned incomplete data: "
            f"expected {expected_episodes} episodes, got {len(trajectories)}"
        )
    return {
        "val_success_rate": float(
            sum(1 for trajectory in trajectories if trajectory.success)
            / expected_episodes
        ),
        "val_avg_reward": float(
            sum(trajectory.reward for trajectory in trajectories) / expected_episodes
        ),
        "val_avg_steps": float(
            sum(trajectory.num_steps for trajectory in trajectories)
            / expected_episodes
        ),
        "val_num_episodes": float(expected_episodes),
    }


# ---------------------------------------------------------------------------
# PPO forward pass (Qwen with gradients)
# ---------------------------------------------------------------------------

def compute_policy_token_stats_for_batch(
    ppo_items: list[dict[str, Any]],
    model,
    processor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    history_window: int,
    temperature: float,
    latent_token_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Recompute VAGEN-style response-token log-probs and entropies.

    Every sampled thought token and sampled action token participates in PPO.
    Deterministically injected latent queries and action delimiters are context,
    not policy decisions, and therefore stay outside the loss mask.
    """

    from nimloth.latent.extraction import LatentActionTokens, latent_state_tokens
    from nimloth.training.rl.rollout import (
        _append_input_token,
        build_nimloth_policy_messages,
        policy_inputs_from_messages,
    )

    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    tokens = LatentActionTokens()
    action_token_ids = [token_id_map[token] for token in tokens.action_tokens]
    flattened_log_probs: list[torch.Tensor] = []
    flattened_entropies: list[torch.Tensor] = []
    token_counts: list[int] = []

    for item in ppo_items:
        messages, images = build_nimloth_policy_messages(
            item["image_history_paths"],
            item["system_prompt"],
            item["observation_texts"],
            item["assistant_responses"],
            history_window=history_window,
        )
        model_inputs, _ = policy_inputs_from_messages(
            model, processor, messages, images
        )
        prefix_length = int(model_inputs["input_ids"].shape[1])
        thought_ids = [int(token) for token in item["thought_token_ids"]]
        decoded_thought = processor.tokenizer.decode(
            thought_ids, skip_special_tokens=False
        ).strip()
        if decoded_thought != str(item["current_thought"]).strip():
            raise RuntimeError(
                "stored thought token IDs do not reproduce current_thought: "
                f"decoded={decoded_thought!r}, text={item['current_thought']!r}"
            )
        for token_id in thought_ids:
            model_inputs = _append_input_token(model_inputs, token_id)
        for query_name in latent_state_tokens(latent_token_count, tokens):
            model_inputs = _append_input_token(
                model_inputs, token_id_map[query_name]
            )
        model_inputs = _append_input_token(
            model_inputs, token_id_map[tokens.action_start]
        )

        logit_positions = torch.tensor(
            [
                *(prefix_length - 1 + offset for offset in range(len(thought_ids))),
                int(model_inputs["input_ids"].shape[1]) - 1,
            ],
            device=model_inputs["input_ids"].device,
            dtype=torch.long,
        )
        outputs = model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            logits_to_keep=logit_positions,
        )
        for offset, selected_id in enumerate(thought_ids):
            raw_logits = outputs.logits[0, offset].float()
            policy_logits = raw_logits if temperature == 0 else raw_logits / temperature
            token_log_probs = torch.log_softmax(policy_logits, dim=-1)
            token_probs = torch.softmax(policy_logits, dim=-1)
            flattened_log_probs.append(token_log_probs[selected_id])
            flattened_entropies.append(-(token_probs * token_log_probs).sum())

        action_ids = torch.tensor(
            action_token_ids, device=outputs.logits.device
        )
        raw_action_logits = outputs.logits[0, -1, action_ids].float()
        action_logits = (
            raw_action_logits
            if temperature == 0
            else raw_action_logits / temperature
        )
        action_log_probs = torch.log_softmax(action_logits, dim=-1)
        action_probs = torch.softmax(action_logits, dim=-1)
        taken_action = int(item["taken_action_idx"])
        flattened_log_probs.append(action_log_probs[taken_action])
        flattened_entropies.append(
            -(action_probs * action_log_probs).sum()
        )
        token_counts.append(len(thought_ids) + 1)

    if not flattened_log_probs:
        raise ValueError("PPO batch contains no stochastic response tokens")
    return (
        torch.stack(flattened_log_probs),
        torch.stack(flattened_entropies),
        token_counts,
    )


def _trajectory_policy_items(
    trajectories: list[RolloutTrajectory],
) -> tuple[list[dict[str, Any]], list[tuple[RolloutTrajectory, int]]]:
    """Flatten exact per-turn policy contexts without changing transcript order."""

    from nimloth.training.rl.vagen_protocol import thought_from_assistant_response

    items: list[dict[str, Any]] = []
    owners: list[tuple[RolloutTrajectory, int]] = []
    for trajectory in trajectories:
        for step in range(trajectory.num_steps):
            items.append({
                "image_history_paths": trajectory.image_paths[: step + 1],
                "system_prompt": trajectory.system_prompt,
                "observation_texts": trajectory.observation_texts[: step + 1],
                "assistant_responses": trajectory.assistant_responses[:step],
                "current_thought": thought_from_assistant_response(
                    trajectory.assistant_responses[step]
                ),
                "thought_token_ids": trajectory.thought_token_ids[step],
                "taken_action_idx": trajectory.action_indices[step],
            })
            owners.append((trajectory, step))
    return items, owners


def attach_policy_and_reference_token_log_probs(
    trajectories: list[RolloutTrajectory],
    model,
    processor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    history_window: int,
    temperature: float,
    latent_token_count: int,
    compute_reference: bool,
) -> dict[str, float]:
    """Recompute PPO-old and immutable-base reference token log-probs.

    VAGEN recomputes old log-probs with the actor after rollout rather than
    trusting inference-engine values. We retain generation-time behavior values
    for audit, but PPO ratios use this full-response replay.
    """

    items, owners = _trajectory_policy_items(trajectories)
    for trajectory in trajectories:
        trajectory.ppo_old_token_log_probs = []
        trajectory.reference_token_log_probs = []

    with _temporary_eval(model), torch.no_grad():
        old_log_probs, _old_entropies, old_counts = (
            compute_policy_token_stats_for_batch(
                items,
                model,
                processor,
                token_id_map,
                device,
                history_window=history_window,
                temperature=temperature,
                latent_token_count=latent_token_count,
            )
        )
    reference_log_probs: torch.Tensor | None = None
    reference_counts: list[int] = []
    if compute_reference:
        with _temporary_eval(model), disable_lora_adapters_for_reference(model):
            with torch.no_grad():
                reference_log_probs, _reference_entropies, reference_counts = (
                    compute_policy_token_stats_for_batch(
                        items,
                        model,
                        processor,
                        token_id_map,
                        device,
                        history_window=history_window,
                        temperature=temperature,
                        latent_token_count=latent_token_count,
                    )
                )
        if old_counts != reference_counts:
            raise RuntimeError(
                f"policy/reference token counts differ: {old_counts} != {reference_counts}"
            )

    def split_values(
        flattened: torch.Tensor,
        counts: list[int],
        *,
        label: str,
    ) -> dict[int, list[list[float]]]:
        cursor = 0
        result = {
            id(trajectory): [[] for _ in range(trajectory.num_steps)]
            for trajectory in trajectories
        }
        for (trajectory, step), count in zip(owners, counts):
            values = flattened[cursor:cursor + count].detach().cpu().tolist()
            result[id(trajectory)][step] = [float(value) for value in values]
            cursor += count
        if cursor != int(flattened.numel()):
            raise RuntimeError(
                f"{label} token split consumed {cursor} of {flattened.numel()} values"
            )
        return result

    old_by_trajectory = split_values(old_log_probs, old_counts, label="PPO-old")
    reference_by_trajectory = (
        split_values(reference_log_probs, reference_counts, label="reference")
        if reference_log_probs is not None else {
            id(trajectory): [] for trajectory in trajectories
        }
    )
    behavior_deltas: list[float] = []
    for trajectory in trajectories:
        trajectory.ppo_old_token_log_probs = old_by_trajectory[id(trajectory)]
        trajectory.reference_token_log_probs = reference_by_trajectory[id(trajectory)]
        for step in range(trajectory.num_steps):
            behavior = [
                *trajectory.thought_token_log_probs[step],
                trajectory.action_log_probs[step][trajectory.action_indices[step]],
            ]
            replayed = trajectory.ppo_old_token_log_probs[step]
            behavior_deltas.extend(
                abs(float(left) - float(right))
                for left, right in zip(behavior, replayed)
            )
        validate_rollout_trajectory(trajectory)
    return {
        "policy_tokens": float(old_log_probs.numel()),
        "behavior_replay_logprob_max_delta": max(behavior_deltas, default=0.0),
        "behavior_replay_logprob_mean_delta": (
            sum(behavior_deltas) / len(behavior_deltas)
            if behavior_deltas else 0.0
        ),
    }


def attach_critic_token_values(
    trajectories: list[RolloutTrajectory],
    critic: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    history_window: int,
    latent_token_count: int,
) -> dict[str, float]:
    """Attach old independent-critic values for every stochastic token."""

    from nimloth.training.rl.token_critic import compute_token_values_for_batch

    items, owners = _trajectory_policy_items(trajectories)
    with _temporary_eval(critic), torch.no_grad():
        flattened, counts = compute_token_values_for_batch(
            items,
            critic,
            processor,
            token_id_map,
            device,
            history_window=history_window,
            latent_token_count=latent_token_count,
        )
    cursor = 0
    by_trajectory = {
        id(trajectory): [[] for _ in range(trajectory.num_steps)]
        for trajectory in trajectories
    }
    for (trajectory, step), count in zip(owners, counts):
        by_trajectory[id(trajectory)][step] = [
            float(value)
            for value in flattened[cursor:cursor + count].detach().cpu().tolist()
        ]
        cursor += count
    if cursor != flattened.numel():
        raise RuntimeError(
            f"critic token split consumed {cursor} of {flattened.numel()} values"
        )
    for trajectory in trajectories:
        trajectory.critic_token_values = by_trajectory[id(trajectory)]
        validate_rollout_trajectory(trajectory)
    return {
        "critic_tokens": float(flattened.numel()),
        "critic_value_mean": float(flattened.float().mean().item()),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


@contextmanager
def disable_lora_adapters_for_reference(model: torch.nn.Module):
    """Expose the immutable merged SFT2 base as PPO's reference policy.

    PEFT's public ``disable_adapter()`` changes ``requires_grad`` and has had
    FSDP flat-parameter incompatibilities. LoRA forward layers already consult
    ``_disable_adapters``; toggling only that boolean preserves FSDP ownership.
    """

    root = _unwrap(model)
    adapter_layers = [
        module for module in root.modules()
        if hasattr(module, "_disable_adapters") and hasattr(module, "base_layer")
    ]
    if not adapter_layers:
        raise RuntimeError(
            "KL reference requires LoRA adapter layers over the immutable SFT2 base"
        )
    previous = [bool(module._disable_adapters) for module in adapter_layers]
    try:
        for module in adapter_layers:
            module._disable_adapters = True
        yield
    finally:
        for module, was_disabled in zip(adapter_layers, previous):
            module._disable_adapters = was_disabled


def validate_policy_tune_combination(
    *,
    llm_tune: str,
    vision_tune: str,
) -> None:
    """Reject mixed full/LoRA policies that PEFT cannot checkpoint completely."""

    modes = {llm_tune, vision_tune}
    if "lora" in modes and "full" in modes:
        raise ValueError(
            "mixed full/LoRA Qwen tuning is unsupported: PEFT checkpoints only "
            "the adapters and would drop full-tuned backbone parameters"
        )


def normalize_policy_parameter_dtype(
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
) -> None:
    """Cast PEFT-created FP32 parameters to the BF16 policy dtype for FSDP."""

    module.to(dtype=dtype)
    dtypes = {
        parameter.dtype
        for parameter in module.parameters()
        if parameter.is_floating_point()
    }
    if dtypes != {dtype}:
        raise RuntimeError(
            f"policy parameters must have one FSDP dtype {dtype}, got {dtypes}"
        )


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


@contextmanager
def _temporary_eval(module: torch.nn.Module):
    """Run inference without permanently changing per-submodule train modes."""

    training_modes = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, was_training in training_modes:
            submodule.training = was_training


@contextmanager
def _temporary_deterministic_train(module: torch.nn.Module):
    """Enable gradient checkpointing while keeping PPO recompute deterministic.

    Hugging Face Qwen only applies gradient checkpointing when ``training`` is
    true. Actor recompute previously used ``eval()``, silently disabling the
    requested checkpointing and retaining every long-history activation. Set
    train mode but temporarily turn module-based dropout into an identity so
    the recomputed policy remains equivalent to deterministic evaluation.
    """

    training_modes = [(submodule, submodule.training) for submodule in module.modules()]
    dropouts = [
        (submodule, float(submodule.p))
        for submodule in module.modules()
        if isinstance(submodule, torch.nn.modules.dropout._DropoutNd)
    ]
    module.train()
    try:
        for dropout, _probability in dropouts:
            dropout.p = 0.0
        yield
    finally:
        for dropout, probability in dropouts:
            dropout.p = probability
        for submodule, was_training in training_modes:
            submodule.training = was_training


def _maybe_init_wandb(
    *,
    rank: int,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
):
    """Initialize one resumable W&B run after distributed rank is known."""

    if rank != 0:
        return None
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
        print(json.dumps({"wandb": "skipped", "reason": "WANDB_API_KEY not set"}))
        return None

    import wandb

    run_id_path = output_dir / "wandb_run_id.txt"
    requested_run_id = os.environ.get("WANDB_RUN_ID")
    if requested_run_id is None and run_id_path.is_file():
        requested_run_id = run_id_path.read_text(encoding="utf-8").strip() or None
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nimloth-rl"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME") or args.experiment_name,
        id=requested_run_id,
        resume="allow" if requested_run_id is not None else None,
        mode=mode,
        config=config,
        dir=os.environ.get("WANDB_DIR"),
    )
    run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    print(json.dumps({"wandb": "initialized", "run_id": run.id, "resume": requested_run_id is not None}))
    return run


def _require_optimizer_progress(global_step: int) -> None:
    """Refuse to materialize a misleading final checkpoint with zero updates."""

    if int(global_step) <= 0:
        raise RuntimeError(
            "RL completed zero optimizer steps; refusing to write final checkpoint"
        )


def _broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    """Synchronize a small non-FSDP module across ranks.

    JSONL/FSDP mode intentionally makes every rank consume identical data so the
    small WM/value modules can remain local replicas.  This only works if their
    initial parameters are identical; CLI construction happens before
    ``setup_dist()``, so we explicitly broadcast rank-0 state after device setup.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def train_rl(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    collector: RolloutCollector,
    output_dir: Path,
    validation_collector: RolloutCollector | None = None,
) -> int:
    """Run the online RL training loop."""

    # --- unpack config -------------------------------------------------------
    rl_cfg: dict = config.get("rl", {})
    freeze_cfg: dict = config.get("freeze", {})
    pred_cfg: dict = config.get("predictor", {})
    vh_cfg: dict = config.get("value_head", {})
    algo_cfg: dict = config.get("algorithm", {})
    critic_cfg: dict = config.get("critic", {})
    val_cfg: dict = config.get("validation", {})
    train_cfg: dict = config.get("training", {})

    iterations: int = rl_cfg.get("iterations", 1000)
    envs_per_iter: int = rl_cfg.get("envs_per_iteration", 8)
    max_steps_per_ep: int = rl_cfg.get("max_steps_per_episode", 20)
    gamma: float = rl_cfg.get("gamma", 0.99)
    batch_size: int = rl_cfg.get("batch_size", 32)
    if batch_size <= 0:
        raise ValueError(f"rl.batch_size must be > 0, got {batch_size}")

    pred_lr: float = float(pred_cfg.get("lr", 1e-3))
    vh_lr: float = float(vh_cfg.get("lr", 1e-3))
    rank_margin: float = float(vh_cfg.get("rank_margin", 0.0))
    lambda_rank: float = float(vh_cfg.get("lambda_rank", 0.0))

    # Actor (Qwen PPO) config
    actor_cfg: dict = config.get("actor", {})
    actor_enabled: bool = bool(actor_cfg) and not freeze_cfg.get("qwen", True)
    actor_lr: float = float(actor_cfg.get("lr", 1e-6))
    entropy_coeff: float = float(actor_cfg.get("entropy_coeff", 0.0))
    clip_ratio: float = float(actor_cfg.get("clip_ratio", 0.2))
    use_kl_loss: bool = bool(actor_cfg.get("use_kl_loss", False))
    kl_loss_coef: float = float(actor_cfg.get("kl_loss_coef", 0.0))
    kl_loss_type: str = str(actor_cfg.get("kl_loss_type", "low_var_kl"))
    use_kl_in_reward: bool = bool(actor_cfg.get("use_kl_in_reward", False))
    kl_reward_coef: float = float(actor_cfg.get("kl_reward_coef", 0.0))
    kl_reward_type: str = str(actor_cfg.get("kl_reward_type", "kl"))
    if use_kl_loss and use_kl_in_reward:
        raise ValueError(
            "VAGEN applies actor KL instead of reward KL when use_kl_loss=true; "
            "enable only one KL placement"
        )
    if use_kl_loss and kl_loss_coef <= 0:
        raise ValueError("actor KL loss requires actor.kl_loss_coef > 0")
    if use_kl_in_reward and kl_reward_coef <= 0:
        raise ValueError("reward KL requires actor.kl_reward_coef > 0")
    if (use_kl_loss or use_kl_in_reward) and not uses_lora(args):
        raise ValueError(
            "KL reference currently requires LoRA so the immutable merged SFT2 "
            "base can be evaluated with adapters disabled"
        )

    # Config-controlled freeze is advisory — actual tuning is via --llm-tune / --vision-tune
    freeze_qwen: bool = freeze_cfg.get("qwen", True)
    freeze_state_proj: bool = freeze_cfg.get("state_proj", True)

    log_interval: int = train_cfg.get("log_interval", 10)
    save_interval: int = train_cfg.get("save_interval", 50)
    val_enabled: bool = val_cfg.get("enabled", True)
    val_interval: int = val_cfg.get("interval", 50)
    val_envs: int = val_cfg.get("envs", 16)
    val_baseline: bool = bool(val_cfg.get("baseline", False))
    val_max_steps: int = int(val_cfg.get("max_steps_per_episode", max_steps_per_ep))
    stop_if_no_success_by = int(
        train_cfg.get("stop_if_no_success_by_iteration", 0)
    )
    evaluation_only = bool(train_cfg.get("evaluation_only", False))
    if evaluation_only and not (val_enabled and val_baseline):
        raise ValueError(
            "training.evaluation_only requires validation.enabled=true and baseline=true"
        )
    if evaluation_only and (iterations != 0 or envs_per_iter != 0):
        raise ValueError(
            "training.evaluation_only requires rl.iterations=0 and envs_per_iteration=0"
        )
    if evaluation_only and bool(getattr(args, "resume", False)):
        raise ValueError("evaluation-only baseline cannot resume a training checkpoint")
    seed: int = train_cfg.get("seed", 42)

    adv_estimator = str(algo_cfg.get("adv_estimator", "turn_mc"))
    if adv_estimator not in {"turn_mc", "masked_gae"}:
        raise ValueError(
            f"algorithm.adv_estimator must be turn_mc or masked_gae, got {adv_estimator!r}"
        )
    gae_gamma = float(algo_cfg.get("gamma", gamma))
    gae_lam = float(algo_cfg.get("lam", 1.0))
    reward_placement = str(algo_cfg.get("reward_placement", "per_turn"))
    critic_backend = str(critic_cfg.get("backend", "independent_qwen"))
    critic_lr = float(critic_cfg.get("lr", 1e-5))
    critic_micro_batch_size = int(critic_cfg.get("micro_batch_size", 1))
    critic_cliprange_value = float(critic_cfg.get("cliprange_value", 0.5))
    critic_gradient_checkpointing = bool(
        critic_cfg.get("gradient_checkpointing", True)
    )
    if adv_estimator == "masked_gae":
        if not actor_enabled:
            raise ValueError("masked_gae requires an enabled actor")
        if critic_backend != "independent_qwen":
            raise ValueError(
                "masked_gae currently requires critic.backend=independent_qwen"
            )
        if critic_micro_batch_size != 1 or batch_size != 1:
            raise ValueError(
                "initial independent-Qwen masked_gae requires critic and transition "
                "microbatch size1 until a long-history memory gate passes"
            )
        if reward_placement not in {"final", "per_turn"}:
            raise ValueError(
                "masked_gae reward_placement must be final or per_turn"
            )
        if use_kl_in_reward:
            raise ValueError(
                "masked_gae token KL-in-reward is not implemented; use actor KL loss"
            )
        if critic_lr <= 0 or critic_cliprange_value <= 0:
            raise ValueError("masked_gae critic lr and cliprange_value must be positive")

    # --- tuning modes --------------------------------------------------------
    llm_tune, vision_tune = resolve_tune_modes(args)
    validate_policy_tune_combination(
        llm_tune=llm_tune, vision_tune=vision_tune
    )
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)

    # --- distributed setup ---------------------------------------------------
    rank, world, local_rank, device = setup_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    wandb_run = _maybe_init_wandb(
        rank=rank,
        output_dir=output_dir,
        args=args,
        config=config,
    )

    # --- Qwen model loading --------------------------------------------------
    policy_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    latent_token_count = validate_rl_policy_protocol(policy_config)
    attention_dropout = float(
        getattr(getattr(policy_config, "text_config", None), "attention_dropout", 0.0)
    )
    if actor_enabled and attention_dropout != 0.0:
        raise ValueError(
            "deterministic actor recompute requires text attention_dropout=0, "
            f"got {attention_dropout}"
        )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    n_added = add_special_tokens(
        processor.tokenizer, latent_token_count=latent_token_count
    )
    token_id_map = special_token_ids(
        processor.tokenizer, latent_token_count=latent_token_count
    )
    tokenizer_vocab = len(processor.tokenizer)

    resume_ckpt_dir = (
        Path(args.resume_checkpoint).resolve()
        if args.resume_checkpoint is not None
        else output_dir / "best"
    )
    resume_state_path = resume_ckpt_dir / "rl_state.pt"
    resume_adapter = resume_ckpt_dir / "adapter_config.json"
    base_model_path = str(args.model)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    loaded_latent_token_count = validate_rl_policy_protocol(model.config)
    if loaded_latent_token_count != latent_token_count:
        raise ValueError(
            "policy config changed while loading: "
            f"expected k={latent_token_count}, got k={loaded_latent_token_count}"
        )
    policy_dtype = model.get_input_embeddings().weight.dtype
    model_vocab_before = model.get_input_embeddings().weight.shape[0]
    # Log model embedding info before resize
    embed = model.get_input_embeddings()
    pad_idx = getattr(embed, "padding_idx", None)
    print(json.dumps({
        "rank": rank,
        "model_vocab_before": model_vocab_before,
        "tokenizer_vocab": tokenizer_vocab,
        "n_added": n_added,
        "padding_idx": pad_idx,
        "embed_weight_shape": list(embed.weight.shape),
    }), flush=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if n_added > 0:
        model.resize_token_embeddings(tokenizer_vocab)
        model_vocab_after = model.get_input_embeddings().weight.shape[0]
        print(json.dumps({"rank": rank, "resized": True,
                          "model_vocab_after": model_vocab_after}), flush=True)

    # Resume branches
    resume_aux_ckpt: Path | None = None  # for loading WM + optimizer later

    if args.resume and resume_state_path.exists() and resume_adapter.exists():
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires llm_tune and/or vision_tune lora")
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        saved_base = saved.get("base_model_path")
        if saved_base:
            base_model_path = str(saved_base)
        if is_main():
            print(json.dumps({"resume_lora_adapter": str(resume_ckpt_dir),
                              "base_model_path": base_model_path}))
        model = configure_qwen_tuning(model, args)
        load_lora_adapter_state(model, resume_ckpt_dir)
        resume_aux_ckpt = resume_ckpt_dir

    elif args.resume and resume_state_path.exists() and (resume_ckpt_dir / "config.json").exists():
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with lora tuning")
        if is_main():
            print(json.dumps({"resume_full": str(resume_ckpt_dir)}))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_ckpt_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.resize_token_embeddings(len(processor.tokenizer))
        model = configure_qwen_tuning(model, args)
        resume_aux_ckpt = resume_ckpt_dir

    else:
        model = configure_qwen_tuning(model, args)
        if is_main():
            print(json.dumps({"init": "configured_tuning",
                              "base_model_path": base_model_path,
                              "llm_tune": llm_tune,
                              "vision_tune": vision_tune}))

    normalize_policy_parameter_dtype(model, dtype=policy_dtype)
    model.to(device)
    if is_main():
        print(json.dumps({"policy_parameter_dtype": str(policy_dtype)}))

    # --- freeze WM-encoding pathway if requested -----------------------------
    if freeze_qwen and llm_tune == "freeze" and vision_tune == "freeze":
        _freeze(model)
    if freeze_state_proj:
        _freeze(state_proj)

    state_proj.to(device)
    wm_predictor.to(device)
    value_head.to(device)
    if world > 1:
        _broadcast_module_state(state_proj)
        _broadcast_module_state(wm_predictor)
        _broadcast_module_state(value_head)
        if is_main():
            print(json.dumps({"synced_local_wm_modules": True, "world_size": world}))

    # --- Vision EMA -----------------------------------------------------------
    vision_ema: VisionEncoderEMA | None = None
    if vision_ema_enabled:
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = resume_ckpt_dir / "vision_ema.pt"
        if args.resume and ema_path.is_file():
            loaded_ema = VisionEncoderEMA.load_checkpoint(ema_path, map_location=device)
            vision_ema.decay = loaded_ema.decay
            vision_ema.shadow = {k: v.to(device) for k, v in loaded_ema.shadow.items()}
        if is_main():
            print(json.dumps({"vision_ema": True,
                              "shadow_params": len(vision_ema.shadow),
                              "decay": vision_ema.decay}))

    # --- FSDP wrap ------------------------------------------------------------
    if world > 1:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        # FULL_SHARD splits the embedding across ranks. If the padding_idx
        # row doesn't fall on every rank's shard, FSDP forward hits:
        #   assert padding_idx < weight.size(0)
        # Clearing padding_idx is safe: it only zeroes the padding embedding
        # row during forward, which the model doesn't rely on.
        embed = model.get_input_embeddings()
        if hasattr(embed, "padding_idx") and embed.padding_idx is not None:
            embed.padding_idx = None
            if is_main():
                print(json.dumps({"cleared_padding_idx": True}))

        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            use_orig_params=True,
        )
        if is_main():
            print(json.dumps({"fsdp": "wrapped", "world_size": world}))
    # FSDP handles multi-GPU; small modules stay on device if world==1.

    critic_model: torch.nn.Module | None = None
    if adv_estimator == "masked_gae":
        from nimloth.training.rl.token_critic import (
            load_independent_qwen_critic,
        )

        critic_source = (
            resume_ckpt_dir / "critic"
            if args.resume and (resume_ckpt_dir / "critic").is_dir()
            else Path(args.model)
        )
        if args.resume and critic_source == Path(args.model):
            raise FileNotFoundError(
                f"masked_gae resume missing independent critic at {resume_ckpt_dir / 'critic'}"
            )
        critic_model = load_independent_qwen_critic(
            critic_source,
            dtype=policy_dtype,
            attn_implementation=args.attn_implementation,
            gradient_checkpointing=critic_gradient_checkpointing,
        )
        critic_model.to(device)
        critic_embedding = critic_model.get_input_embeddings()
        if (
            hasattr(critic_embedding, "padding_idx")
            and critic_embedding.padding_idx is not None
        ):
            critic_embedding.padding_idx = None
        if world > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
            )

            critic_model = FSDP(
                critic_model,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                sync_module_states=True,
                use_orig_params=True,
            )
        if is_main():
            print(json.dumps({
                "critic": "independent_qwen",
                "source": str(critic_source),
                "fsdp": world > 1,
                "gradient_checkpointing": critic_gradient_checkpointing,
            }))

    # --- Wire online env rollout after FSDP wrapping -------------------------
    from nimloth.training.rl.rollout import EnvRolloutCollector

    if isinstance(collector, EnvRolloutCollector) and lambda_rank != 0.0:
        raise ValueError(
            "dynamic online RL forbids unconditional chosen-action ranking; "
            "VAGEN uses per-turn returns without this auxiliary objective"
        )

    def _wire_env_collector(
        env_collector: RolloutCollector,
        *,
        role: str,
    ) -> RolloutCollector:
        if not isinstance(env_collector, EnvRolloutCollector):
            raise TypeError(f"{role} collector must be EnvRolloutCollector")
        if env_collector._latent_token_count != latent_token_count:
            raise ValueError(
                f"{role} collector/model latent-token mismatch: "
                f"collector={env_collector._latent_token_count}, "
                f"model={latent_token_count}"
            )
        if world > 1:
            from nimloth.training.rl.distributed_rollout import (
                DistributedEnvRolloutCollector,
            )
            env_collector = DistributedEnvRolloutCollector.from_collector(
                env_collector
            )
        env_collector._model = model
        env_collector._processor = processor
        env_collector._device = device
        if is_main():
            print(json.dumps({
                "env_collector": role,
                "distributed": world > 1,
                "device": str(device),
                "world_size": world,
                "history_window": env_collector._history_window,
                "max_think_tokens": env_collector._max_think_tokens,
                "eval_sets": env_collector._eval_sets,
                "split": env_collector._split,
            }))
        return env_collector

    if isinstance(collector, EnvRolloutCollector):
        collector = _wire_env_collector(collector, role="train")
    if val_enabled:
        if validation_collector is None:
            raise ValueError(
                "validation.enabled requires a separate heldout env collector"
            )
        validation_collector = _wire_env_collector(
            validation_collector, role="heldout_validation"
        )
    elif validation_collector is not None:
        raise ValueError(
            "validation collector was provided while validation.enabled is false"
        )

    if isinstance(collector, EnvRolloutCollector):
        rollout_protocol: dict[str, Any] = {
            "mode": "dynamic_env",
            "split": collector._split,
            "eval_sets": list(collector._eval_sets),
            "history_window": int(collector._history_window),
            "max_think_tokens": int(collector._max_think_tokens),
            "prompt_protocol": "vagen-source-eval-to-nimloth-inject-v2",
            "image_protocol": "vagen-process-image-min512-max2048-v1",
            "reward_protocol": "vagen-per-turn-plus-final-v1",
            "optimization_protocol": "all-stochastic-response-tokens-one-ppo-epoch-v2",
            "policy_loss_mask": "sampled-thought-plus-action;injected-scaffold-excluded",
            "actor_recompute_protocol": (
                "deterministic-train-gradient-checkpointing-selected-logits-v1"
            ),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "functional_attention_dropout": attention_dropout,
            "transition_microbatch_size": batch_size,
            "advantage": {
                "estimator": adv_estimator,
                "gamma": gae_gamma if adv_estimator == "masked_gae" else gamma,
                "lam": gae_lam if adv_estimator == "masked_gae" else None,
                "reward_placement": reward_placement,
            },
            "critic": {
                "backend": critic_backend if adv_estimator == "masked_gae" else None,
                "lr": critic_lr if adv_estimator == "masked_gae" else None,
                "micro_batch_size": (
                    critic_micro_batch_size if adv_estimator == "masked_gae" else None
                ),
                "cliprange_value": (
                    critic_cliprange_value if adv_estimator == "masked_gae" else None
                ),
                "gradient_checkpointing": (
                    critic_gradient_checkpointing
                    if adv_estimator == "masked_gae" else None
                ),
            },
            "value_ranking_weight": 0.0,
            "trajectory_schema": 4 if adv_estimator == "masked_gae" else 3,
            "kl": {
                "reference": "immutable-merged-sft2-base-adapters-disabled",
                "use_actor_loss": use_kl_loss,
                "actor_coef": kl_loss_coef,
                "actor_penalty": kl_loss_type,
                "use_in_reward": use_kl_in_reward,
                "reward_coef": kl_reward_coef,
                "reward_penalty": kl_reward_type,
            },
            "attention_implementation": str(args.attn_implementation),
            "environment_config": collector._environment_config(
                str(collector._eval_sets[0])
            )["env_config"] | {"eval_set": "<per-episode>"},
            "temperature": float(rl_cfg.get("temperature", 1.0)),
            "top_p": float(rl_cfg.get("top_p", 1.0)),
            "seed_offset": int(collector._base_seed_offset),
            "env_timeout": int(collector._env_timeout),
            "control_backend": "gloo" if world > 1 else "local",
            "latent_token_count": latent_token_count,
            "latent_query_mode": "inject",
        }
        if isinstance(validation_collector, EnvRolloutCollector):
            rollout_protocol["validation"] = {
                "split": validation_collector._split,
                "eval_sets": list(validation_collector._eval_sets),
                "history_window": int(validation_collector._history_window),
                "max_think_tokens": int(validation_collector._max_think_tokens),
                "temperature": float(validation_collector._temperature),
                "top_p": float(validation_collector._top_p),
                "seed_offset": int(validation_collector._base_seed_offset),
                "env_timeout": int(validation_collector._env_timeout),
                "baseline": val_baseline,
                "interval": val_interval,
                "envs": val_envs,
                "max_steps_per_episode": val_max_steps,
            }
    else:
        rollout_protocol = {
            "mode": "jsonl",
            "sources": [str(path) for path in getattr(collector, "_sources", [])],
        }
    if is_main():
        (output_dir / "rollout_protocol.json").write_text(
            json.dumps(rollout_protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ppo_temperature = (
        float(rl_cfg.get("temperature", 1.0))
        if rollout_protocol["mode"] == "dynamic_env"
        else 1.0
    )

    # --- optimizer ------------------------------------------------------------
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": actor_lr, "name": "qwen"},
        {"params": state_proj.parameters(), "lr": pred_lr, "name": "state_proj"},
        {"params": wm_predictor.parameters(), "lr": pred_lr, "name": "wm_predictor"},
    ]
    if adv_estimator == "turn_mc":
        param_groups.append({
            "params": value_head.parameters(), "lr": vh_lr, "name": "value_head"
        })
    else:
        _freeze(value_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    critic_optimizer = (
        torch.optim.AdamW(
            [parameter for parameter in critic_model.parameters() if parameter.requires_grad],
            lr=critic_lr,
            weight_decay=1e-2,
        )
        if critic_model is not None else None
    )

    # --- resume training state ------------------------------------------------
    start_iteration = 1
    global_step = 0
    best_value_loss = float("inf")
    if resume_aux_ckpt is not None:
        resume_state = load_rl_wm_checkpoint(
            resume_aux_ckpt, state_proj, wm_predictor, value_head, device
        )
        if resume_state:
            saved_rollout_protocol = resume_state.get("rollout_protocol")
            if saved_rollout_protocol is None:
                raise RuntimeError(
                    "RL checkpoint has no rollout_protocol metadata; refusing an unverifiable resume"
                )
            if saved_rollout_protocol != rollout_protocol:
                raise RuntimeError(
                    "rollout protocol mismatch on resume: "
                    f"saved={saved_rollout_protocol}, current={rollout_protocol}"
                )
            start_iteration = int(resume_state.get("iteration", 0)) + 1
            global_step = int(resume_state.get("global_step", 0))
            best_value_loss = float(resume_state.get("best_value_loss", float("inf")))
            if world > 1:
                saved_world = int(resume_state.get("optimizer_world_size", 0))
                if saved_world != world:
                    raise RuntimeError(
                        f"FSDP optimizer checkpoint world_size={saved_world}, current={world}"
                    )
                optimizer_path = resume_aux_ckpt / f"optimizer_rank_{rank:05d}.pt"
                if not optimizer_path.is_file():
                    raise FileNotFoundError(f"missing rank optimizer checkpoint: {optimizer_path}")
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location="cpu", weights_only=False)
                )
                if critic_optimizer is not None:
                    saved_critic_world = int(
                        resume_state.get("critic_optimizer_world_size", 0)
                    )
                    if saved_critic_world != world:
                        raise RuntimeError(
                            "critic optimizer world-size mismatch: "
                            f"saved={saved_critic_world}, current={world}"
                        )
                    critic_optimizer_path = (
                        resume_aux_ckpt / f"critic_optimizer_rank_{rank:05d}.pt"
                    )
                    if not critic_optimizer_path.is_file():
                        raise FileNotFoundError(
                            f"missing rank critic optimizer: {critic_optimizer_path}"
                        )
                    critic_optimizer.load_state_dict(torch.load(
                        critic_optimizer_path,
                        map_location="cpu",
                        weights_only=False,
                    ))
            elif resume_state.get("optimizer") is not None:
                optimizer.load_state_dict(resume_state["optimizer"])
                if critic_optimizer is not None:
                    critic_state = resume_state.get("critic_optimizer")
                    if critic_state is None:
                        raise RuntimeError("masked_gae resume has no critic optimizer")
                    critic_optimizer.load_state_dict(critic_state)
            if is_main():
                print(json.dumps({"resume": True, "start_iteration": start_iteration,
                                  "global_step": global_step}))

    if isinstance(collector, EnvRolloutCollector):
        collector.set_resume_iteration(
            start_iteration=start_iteration,
            envs_per_iteration=envs_per_iter,
            validation_enabled=False,
            validation_interval=val_interval,
            validation_envs=val_envs,
        )
        if is_main():
            print(json.dumps({
                "env_seed_cursor": collector._ep_counter,
                "base_seed_offset": collector._base_seed_offset,
                "start_iteration": start_iteration,
            }))

    # --- logging --------------------------------------------------------------
    log_path = output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "time", "iteration", "global_step",
                "wm_mse", "value_loss", "total_loss",
                "num_rollouts", "num_transitions", "optimizer_steps", "success_rate",
                "val_success_rate", "val_avg_reward", "val_avg_steps",
                "actor_loss", "entropy", "clip_fraction", "mean_advantage",
                "policy_tokens", "ppo_kl", "kl_loss", "kl_reward_penalty",
                "critic_loss", "value_clip_fraction", "critic_value_mean",
            ])

    validation_log_path = output_dir / "validation_log.csv"
    if is_main() and val_enabled and not validation_log_path.exists():
        with validation_log_path.open("w", newline="") as stream:
            csv.writer(stream).writerow([
                "time", "label", "iteration", "global_step",
                "val_success_rate", "val_avg_reward", "val_avg_steps",
                "val_num_episodes",
            ])

    def _run_heldout_validation(
        *,
        label: str,
        iteration: int,
        current_global_step: int,
    ) -> dict[str, float]:
        assert isinstance(validation_collector, EnvRolloutCollector)
        validation_collector.reset_seed_cursor()
        with _temporary_eval(model):
            trajectories = validation_collector.collect(
                num_episodes=val_envs,
                max_steps_per_episode=val_max_steps,
                output_dir=output_dir / "validation" / label,
            )
        metrics = summarize_validation_trajectories(
            trajectories, expected_episodes=val_envs
        )
        if is_main():
            with validation_log_path.open("a", newline="") as stream:
                csv.writer(stream).writerow([
                    time.time(), label, iteration, current_global_step,
                    metrics["val_success_rate"], metrics["val_avg_reward"],
                    metrics["val_avg_steps"], metrics["val_num_episodes"],
                ])
            print(json.dumps({
                "validation": label,
                "iteration": iteration,
                "global_step": current_global_step,
                **metrics,
            }))
            if wandb_run is not None:
                wandb_run.log({
                    **{f"validation/{key[4:]}": value for key, value in metrics.items()},
                    "global_step": current_global_step,
                    "iteration": iteration,
                    "validation/label": label,
                })
        return metrics

    if val_enabled and val_baseline and start_iteration == 1:
        _run_heldout_validation(
            label="baseline", iteration=0, current_global_step=global_step
        )

    if evaluation_only:
        if is_main():
            print(json.dumps({
                "evaluation_only": "complete",
                "global_step": global_step,
                "optimizer_steps": 0,
            }))
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_dist()
        return 0

    successful_rollouts_total = 0
    last_completed_iteration = start_iteration - 1
    if log_path.is_file() and start_iteration > 1:
        with log_path.open() as stream:
            for row in csv.DictReader(stream):
                if row.get("success_rate") and row.get("num_rollouts"):
                    successful_rollouts_total += round(
                        float(row["success_rate"]) * float(row["num_rollouts"])
                    )

    # --- main loop ------------------------------------------------------------
    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # 1. Collect trajectories -------------------------------------------------
        if is_main():
            print(json.dumps({"iteration": iteration, "phase": "rollout",
                              "num_episodes": envs_per_iter}))
        if isinstance(collector, EnvRolloutCollector):
            with _temporary_eval(model):
                trajectories = collector.collect(
                    num_episodes=envs_per_iter,
                    max_steps_per_episode=max_steps_per_ep,
                    output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
                )
        else:
            trajectories = collector.collect(
                num_episodes=envs_per_iter,
                max_steps_per_episode=max_steps_per_ep,
                output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
            )
        successful_rollouts_total += sum(
            1 for trajectory in trajectories if trajectory.success
        )
        if is_main():
            print(json.dumps({
                "iteration": iteration,
                "trajectories_collected": len(trajectories),
                "successful_rollouts_total": successful_rollouts_total,
            }))

        if not trajectories:
            if is_main():
                print(json.dumps({"iteration": iteration,
                                  "warning": "no trajectories collected, skipping"}))
            continue

        if actor_enabled:
            replay_metrics = attach_policy_and_reference_token_log_probs(
                trajectories,
                model,
                processor,
                token_id_map,
                device,
                history_window=int(getattr(collector, "_history_window", 112)),
                temperature=ppo_temperature,
                latent_token_count=latent_token_count,
                compute_reference=use_kl_loss or use_kl_in_reward,
            )
            if is_main():
                print(json.dumps({
                    "iteration": iteration,
                    "policy_replay": replay_metrics,
                }))
        if critic_model is not None:
            critic_replay_metrics = attach_critic_token_values(
                trajectories,
                critic_model,
                processor,
                token_id_map,
                device,
                history_window=int(getattr(collector, "_history_window", 112)),
                latent_token_count=latent_token_count,
            )
            if is_main():
                print(json.dumps({
                    "iteration": iteration,
                    "critic_replay": critic_replay_metrics,
                }))
        if is_main():
            from nimloth.training.rl.rollout import save_trajectories

            save_trajectories(
                trajectories,
                output_dir / f"rollouts/iter_{iteration:04d}",
            )

        # 2. Encode → transitions ------------------------------------------------
        with _temporary_eval(model):
            transitions = build_rl_transitions(
                trajectories,
                model,
                processor,
                token_id_map,
                device,
                gamma=gamma,
                latent_token_count=latent_token_count,
                history_window=int(getattr(collector, "_history_window", 112)),
                use_kl_in_reward=use_kl_in_reward,
                kl_reward_coef=kl_reward_coef,
                kl_penalty_type=kl_reward_type,
            )
        # Free GPU memory before PPO forward (Qwen+LoRA+gradients needs extra VRAM).
        torch.cuda.empty_cache()
        if not transitions:
            if is_main():
                print(json.dumps({"iteration": iteration, "warning": "no transitions"}))
            continue

        # Compute one normalized advantage population before any mini-batch update,
        # then consume every transition exactly once (VAGEN ppo_epochs=1).
        sp = _unwrap(state_proj)
        if adv_estimator == "masked_gae":
            assign_masked_gae_to_transitions(
                trajectories,
                transitions,
                gamma=gae_gamma,
                lam=gae_lam,
                reward_placement=reward_placement,
            )
        else:
            raw_advantages: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(transitions), batch_size):
                    items = transitions[start:start + batch_size]
                    hidden = torch.stack([
                        item["qwen_hidden_current"] for item in items
                    ]).to(device)
                    actions = torch.stack([
                        item["action_index"] for item in items
                    ]).to(device)
                    targets = torch.stack([
                        item["value_target"] for item in items
                    ]).to(device)
                    state = sp(hidden).float()
                    values = value_head(state).float().gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)
                    raw_advantages.extend(
                        (targets - values).detach().cpu().unbind()
                    )
            raw_advantage_tensor = torch.stack(raw_advantages)
            policy_token_counts = [
                len(item["old_token_log_probs"]) for item in transitions
            ]
            advantage_tensor = normalize_turn_advantages_for_policy_tokens(
                raw_advantage_tensor, policy_token_counts
            )
            for item, advantage in zip(transitions, advantage_tensor):
                item["advantage"] = advantage

        transition_batches = deterministic_transition_microbatches(
            len(transitions), batch_size, seed=seed + iteration
        )
        metric_sums: dict[str, float] = {}
        optimizer_steps_this_iteration = 0

        for batch_indices in transition_batches:
            batch = [transitions[index] for index in batch_indices]
            hidden_cur = torch.stack([
                item["qwen_hidden_current"] for item in batch
            ]).to(device)
            actions = torch.stack([item["action_index"] for item in batch]).to(device)
            value_targets = torch.stack([
                item["value_target"] for item in batch
            ]).to(device)

            wm_batch = [item for item in batch if item["qwen_hidden_next"] is not None]
            if wm_batch:
                pred_loss, pred_metrics = compute_predictor_loss(
                    qwen_hidden_current=torch.stack([
                        item["qwen_hidden_current"] for item in wm_batch
                    ]).to(device),
                    qwen_hidden_next=torch.stack([
                        item["qwen_hidden_next"] for item in wm_batch
                    ]).to(device),
                    action_indices=torch.stack([
                        item["action_index"] for item in wm_batch
                    ]).to(device),
                    state_proj=state_proj,
                    wm_predictor=wm_predictor,
                )
            else:
                pred_loss = torch.zeros((), device=device)
                pred_metrics = {"wm_mse": 0.0}

            if adv_estimator == "turn_mc":
                wm_state = sp(hidden_cur).float().detach()
                val_loss, val_metrics = compute_value_loss(
                    state_emb=wm_state,
                    action_indices=actions,
                    action_value_targets=value_targets,
                    value_head=value_head,
                    rank_margin=rank_margin,
                    lambda_rank=lambda_rank,
                )
            else:
                val_loss = torch.zeros((), device=device)
                val_metrics = {"value_loss": 0.0}

            actor_metrics: dict[str, float] = {}
            if actor_enabled:
                import gc
                from nimloth.training.rl.loss import (
                    compute_actor_loss,
                    compute_kl_penalty,
                )

                torch.cuda.empty_cache()
                gc.collect()
                ppo_items = [{
                    "image_history_paths": item["image_history_paths"],
                    "system_prompt": item["system_prompt"],
                    "observation_texts": item["observation_texts"],
                    "assistant_responses": item["assistant_responses"],
                    "current_thought": item["current_thought"],
                    "thought_token_ids": item["thought_token_ids"],
                    "taken_action_idx": int(item["action_index"].item()),
                } for item in batch]
                with _temporary_deterministic_train(model):
                    new_log_probs, token_entropies, token_counts = (
                        compute_policy_token_stats_for_batch(
                            ppo_items,
                            model,
                            processor,
                            token_id_map,
                            device,
                            history_window=int(
                                getattr(collector, "_history_window", 112)
                            ),
                            temperature=ppo_temperature,
                            latent_token_count=latent_token_count,
                        )
                    )
                expected_counts = [
                    len(item["old_token_log_probs"]) for item in batch
                ]
                if token_counts != expected_counts:
                    raise RuntimeError(
                        "PPO stochastic-token count mismatch: "
                        f"recomputed={token_counts}, rollout={expected_counts}"
                    )
                old_log_probs = torch.tensor(
                    [
                        value
                        for item in batch
                        for value in item["old_token_log_probs"]
                    ],
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                )
                if adv_estimator == "masked_gae":
                    advantages = torch.cat([
                        torch.tensor(item["token_advantages"])
                        for item in batch
                    ]).to(
                        device=new_log_probs.device,
                        dtype=new_log_probs.dtype,
                    )
                else:
                    advantages = torch.cat([
                        item["advantage"].reshape(1).expand(count)
                        for item, count in zip(batch, token_counts)
                    ]).to(
                        device=new_log_probs.device,
                        dtype=new_log_probs.dtype,
                    )
                actor_loss, actor_metrics = compute_actor_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    clip_ratio=clip_ratio,
                )
                entropy = token_entropies.mean()
                policy_loss = actor_loss - entropy_coeff * entropy
                if use_kl_loss:
                    reference_log_probs = torch.tensor(
                        [
                            value
                            for item in batch
                            for value in item["reference_token_log_probs"]
                        ],
                        device=new_log_probs.device,
                        dtype=new_log_probs.dtype,
                    )
                    kl_loss = compute_kl_penalty(
                        new_log_probs,
                        reference_log_probs,
                        penalty_type=kl_loss_type,
                    ).mean()
                    policy_loss = policy_loss + kl_loss_coef * kl_loss
                    actor_metrics["kl_loss"] = float(kl_loss.detach().item())
                    actor_metrics["kl_coef"] = kl_loss_coef
                total_loss = pred_loss + val_loss + policy_loss
                actor_metrics["entropy"] = float(entropy.detach().item())
                actor_metrics["mean_advantage"] = float(advantages.mean().item())
                actor_metrics["policy_tokens"] = float(
                    new_log_probs.numel() / len(batch)
                )
                actor_metrics["kl_reward_penalty"] = float(sum(
                    item["kl_reward_penalty"] for item in batch
                ) / len(batch))
            else:
                total_loss = pred_loss + val_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], 1.0
            )
            optimizer.step()
            if vision_ema is not None:
                vision_ema.update(model)

            critic_metrics: dict[str, float] = {}
            if critic_model is not None:
                from nimloth.training.rl.loss import (
                    compute_clipped_token_value_loss,
                )
                from nimloth.training.rl.token_critic import (
                    compute_token_values_for_batch,
                )

                assert critic_optimizer is not None
                critic_optimizer.zero_grad(set_to_none=True)
                with _temporary_deterministic_train(critic_model):
                    predicted_values, critic_counts = (
                        compute_token_values_for_batch(
                            ppo_items,
                            critic_model,
                            processor,
                            token_id_map,
                            device,
                            history_window=int(
                                getattr(collector, "_history_window", 112)
                            ),
                            latent_token_count=latent_token_count,
                        )
                    )
                if critic_counts != token_counts:
                    raise RuntimeError(
                        "actor/critic stochastic-token count mismatch: "
                        f"actor={token_counts}, critic={critic_counts}"
                    )
                old_values = torch.cat([
                    torch.tensor(item["old_token_values"])
                    for item in batch
                ]).to(device=predicted_values.device, dtype=predicted_values.dtype)
                token_returns = torch.cat([
                    torch.tensor(item["token_returns"])
                    for item in batch
                ]).to(device=predicted_values.device, dtype=predicted_values.dtype)
                critic_loss, critic_metrics = compute_clipped_token_value_loss(
                    predicted_values=predicted_values,
                    old_values=old_values,
                    returns=token_returns,
                    cliprange_value=critic_cliprange_value,
                )
                critic_loss.backward()
                if hasattr(critic_model, "clip_grad_norm_"):
                    critic_model.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
                critic_optimizer.step()
                val_metrics = {"value_loss": critic_metrics["critic_loss"]}

            global_step += 1
            optimizer_steps_this_iteration += 1

            batch_metrics = {
                "wm_mse": float(pred_metrics.get("wm_mse", 0.0)),
                "value_loss": float(val_metrics.get(
                    "value_loss", val_metrics.get("value_total", 0.0)
                )),
                "total_loss": float(total_loss.detach().item()),
                "actor_loss": float(actor_metrics.get("actor_loss", 0.0)),
                "entropy": float(actor_metrics.get("entropy", 0.0)),
                "clip_fraction": float(actor_metrics.get("clip_fraction", 0.0)),
                "mean_advantage": float(actor_metrics.get("mean_advantage", 0.0)),
                "policy_tokens": float(actor_metrics.get("policy_tokens", 0.0)),
                "ppo_kl": float(actor_metrics.get("ppo_kl", 0.0)),
                "kl_loss": float(actor_metrics.get("kl_loss", 0.0)),
                "kl_reward_penalty": float(
                    actor_metrics.get("kl_reward_penalty", 0.0)
                ),
                "critic_loss": float(critic_metrics.get("critic_loss", 0.0)),
                "value_clip_fraction": float(
                    critic_metrics.get("value_clip_fraction", 0.0)
                ),
                "critic_value_mean": float(
                    critic_metrics.get("critic_value_mean", 0.0)
                ),
            }
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * len(batch)

        last_completed_iteration = iteration
        iter_metrics = {
            key: value / len(transitions) for key, value in metric_sums.items()
        }
        iter_metrics.update({
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(len(transitions)),
            "optimizer_steps": float(optimizer_steps_this_iteration),
            "success_rate": float(
                sum(1 for trajectory in trajectories if trajectory.success)
                / max(1, len(trajectories))
            ),
        })

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- fixed heldout validation -------------------------------------------
        if val_enabled and iteration % val_interval == 0:
            iter_metrics.update(_run_heldout_validation(
                label=f"iter_{iteration:04d}",
                iteration=iteration,
                current_global_step=global_step,
            ))

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- logging -----------------------------------------------------------
        current_val = iter_metrics.get("value_loss", float("inf"))

        if is_main() and (iteration % log_interval == 0 or iteration == 1):
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    time.time(), iteration, global_step,
                    iter_metrics.get("wm_mse", ""),
                    iter_metrics.get("value_loss", ""),
                    iter_metrics.get("total_loss", ""),
                    iter_metrics.get("num_rollouts", ""),
                    iter_metrics.get("num_transitions", ""),
                    iter_metrics.get("optimizer_steps", ""),
                    iter_metrics.get("success_rate", ""),
                    iter_metrics.get("val_success_rate", ""),
                    iter_metrics.get("val_avg_reward", ""),
                    iter_metrics.get("val_avg_steps", ""),
                    iter_metrics.get("actor_loss", ""),
                    iter_metrics.get("entropy", ""),
                    iter_metrics.get("clip_fraction", ""),
                    iter_metrics.get("mean_advantage", ""),
                    iter_metrics.get("policy_tokens", ""),
                    iter_metrics.get("ppo_kl", ""),
                    iter_metrics.get("kl_loss", ""),
                    iter_metrics.get("kl_reward_penalty", ""),
                    iter_metrics.get("critic_loss", ""),
                    iter_metrics.get("value_clip_fraction", ""),
                    iter_metrics.get("critic_value_mean", ""),
                ])
            elapsed = time.time() - iter_start
            print(json.dumps({
                "iteration": iteration,
                "global_step": global_step,
                "metrics": iter_metrics,
                "elapsed_s": round(elapsed, 1),
            }))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{key}": value for key, value in iter_metrics.items()},
                        "global_step": global_step,
                        "iteration": iteration,
                    }
                )

        # --- checkpoint --------------------------------------------------------
        if iteration % save_interval == 0:
            save_rl_checkpoint(
                output_dir / f"iter_{iteration:04d}",
                state_proj=state_proj,
                wm_predictor=wm_predictor,
                value_head=value_head,
                model=model,
                critic_model=critic_model,
                processor=processor,
                vision_ema=vision_ema,
                optimizer=optimizer,
                critic_optimizer=critic_optimizer,
                iteration=iteration,
                global_step=global_step,
                best_value_loss=best_value_loss,
                lora=uses_lora(args),
                llm_tune=llm_tune,
                vision_tune=vision_tune,
                base_model_path=base_model_path,
                rollout_protocol=rollout_protocol,
            )
            if current_val < best_value_loss:
                best_value_loss = current_val
                save_rl_checkpoint(
                    resume_ckpt_dir,  # "best/"
                    state_proj=state_proj,
                    wm_predictor=wm_predictor,
                    value_head=value_head,
                    model=model,
                    critic_model=critic_model,
                    processor=processor,
                    vision_ema=vision_ema,
                    optimizer=optimizer,
                    critic_optimizer=critic_optimizer,
                    iteration=iteration,
                    global_step=global_step,
                    best_value_loss=best_value_loss,
                    lora=uses_lora(args),
                    llm_tune=llm_tune,
                    vision_tune=vision_tune,
                    base_model_path=base_model_path,
                    rollout_protocol=rollout_protocol,
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if (
            stop_if_no_success_by > 0
            and iteration >= stop_if_no_success_by
            and successful_rollouts_total == 0
        ):
            if is_main():
                print(json.dumps({
                    "early_stop": "no_successful_training_trajectory",
                    "iteration": iteration,
                    "global_step": global_step,
                }))
            break

    # --- final checkpoint -----------------------------------------------------
    try:
        _require_optimizer_progress(global_step)
    except RuntimeError:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        cleanup_dist()
        raise
    save_rl_checkpoint(
        output_dir / "final",
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        model=model,
        critic_model=critic_model,
        processor=processor,
        vision_ema=vision_ema,
        optimizer=optimizer,
        critic_optimizer=critic_optimizer,
        iteration=last_completed_iteration,
        global_step=global_step,
        best_value_loss=best_value_loss,
        lora=uses_lora(args),
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        base_model_path=base_model_path,
        rollout_protocol=rollout_protocol,
    )
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0
