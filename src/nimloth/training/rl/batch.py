"""RL transition 的采样与张量化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nimloth.rollout import EncodedTransition


@dataclass(frozen=True)
class RLBatch:
    """RL objective 消费的 hidden、return 与 policy provenance。"""

    transitions: tuple[EncodedTransition, ...]
    current_hidden: torch.Tensor
    next_hidden: torch.Tensor
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor


def select_transition_batch(
    transitions: Sequence[EncodedTransition],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> RLBatch:
    """使用独立 CPU generator 做可复现的 transition 子采样。"""

    if len(transitions) < batch_size:
        raise ValueError(
            f"only {len(transitions)} transitions are available, need {batch_size}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(transitions), generator=generator)[:batch_size]
    selected = tuple(transitions[int(index)] for index in indices)
    return RLBatch(
        transitions=selected,
        current_hidden=torch.stack(
            [transition.current_hidden for transition in selected]
        ).to(device),
        next_hidden=torch.stack(
            [transition.next_hidden for transition in selected]
        ).to(device),
        action_indices=torch.tensor(
            [transition.action_index for transition in selected],
            dtype=torch.long,
            device=device,
        ),
        return_targets=torch.tensor(
            [transition.value_target for transition in selected],
            dtype=torch.float32,
            device=device,
        ),
        old_log_probs=torch.tensor(
            [transition.old_log_prob for transition in selected],
            dtype=torch.float32,
            device=device,
        ),
    )


__all__ = ["RLBatch", "select_transition_batch"]
