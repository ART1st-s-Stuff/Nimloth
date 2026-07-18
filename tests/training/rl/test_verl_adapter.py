from __future__ import annotations

import numpy as np
import pytest
import torch


def _row(
    *,
    trajectory_id: str,
    input_ids: list[int],
    prompt_length: int,
    loss_mask: list[int],
    reward_positions: dict[int, float],
    position_channels: int = 1,
):
    from nimloth.training.rl.verl_adapter import VerlReplayRow

    length = len(input_ids)
    if position_channels == 1:
        position_ids = torch.arange(length)
    else:
        position_ids = torch.arange(length).expand(position_channels, -1).clone()
    rewards = torch.zeros(length)
    end_mask = torch.zeros(length, dtype=torch.long)
    for position, reward in reward_positions.items():
        rewards[position] = reward
        end_mask[position] = 1
    return VerlReplayRow(
        trajectory_id=trajectory_id,
        input_ids=torch.tensor(input_ids),
        attention_mask=torch.ones(length, dtype=torch.long),
        position_ids=position_ids,
        prompt_length=prompt_length,
        loss_mask=torch.tensor(loss_mask, dtype=torch.long),
        token_level_rewards=rewards,
        end_of_response_position_mask=end_mask,
        multi_modal_inputs={"pixel_values": torch.ones(2, 3)},
    )


def test_build_verl_replay_dataproto_preserves_scaffold_and_masks() -> None:
    from nimloth.training.rl.verl_adapter import build_verl_replay_dataproto

    # Row A response: sampled thought 20,21; deterministic latent scaffold 30,31;
    # sampled action 40. Only sampled tokens participate in PPO/GAE.
    row_a = _row(
        trajectory_id="a",
        input_ids=[10, 11, 20, 21, 30, 31, 40],
        prompt_length=2,
        loss_mask=[0, 0, 1, 1, 0, 0, 1],
        reward_positions={6: 1.0},
        position_channels=3,
    )
    row_b = _row(
        trajectory_id="b",
        input_ids=[12, 22, 32, 42],
        prompt_length=1,
        loss_mask=[0, 1, 0, 1],
        reward_positions={3: -0.25},
        position_channels=3,
    )
    data = build_verl_replay_dataproto(
        [row_a, row_b],
        pad_token_id=0,
        temperature=0.7,
        micro_batch_size=1,
    )

    assert tuple(data.batch.batch_size) == (2,)
    assert data.batch["prompts"].tolist() == [[10, 11], [0, 12]]
    assert data.batch["responses"].tolist() == [
        [20, 21, 30, 31, 40],
        [22, 32, 42, 0, 0],
    ]
    assert data.batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
    ]
    assert data.batch["loss_mask"].tolist() == [
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
    ]
    assert torch.equal(data.batch["gae_mask"], data.batch["loss_mask"])
    assert data.batch["token_level_scores"].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -0.25, 0.0, 0.0],
    ]
    assert data.batch["multi_turn_token_level_rewards"].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0],
    ]
    assert data.batch["position_ids"].shape == (2, 3, 7)
    assert data.meta_info == {
        "temperature": 0.7,
        "micro_batch_size": 1,
        "use_dynamic_bsz": False,
    }
    assert data.non_tensor_batch["trajectory_id"].tolist() == ["a", "b"]
    assert all(
        isinstance(value, dict)
        for value in data.non_tensor_batch["multi_modal_inputs"]
    )

    from vagen.trainer.ppo.ray_trainer import (
        AdvantageEstimator,
        compute_advantage,
    )

    data.batch["token_level_rewards"] = data.batch["token_level_scores"].clone()
    data.batch["values"] = torch.zeros_like(data.batch["token_level_scores"])
    compute_advantage(
        data,
        AdvantageEstimator.MASKED_GAE,
        gamma=1.0,
        lam=1.0,
    )
    response_loss_mask = data.batch["loss_mask"][:, -5:].bool()
    assert torch.isfinite(data.batch["advantages"]).all()
    # VAGEN masked_whiten writes normalized filler values outside the mask;
    # actor/critic losses must continue to apply response_loss_mask.
    assert torch.isfinite(
        data.batch["advantages"].masked_select(response_loss_mask)
    ).all()
    assert torch.equal(
        data.batch["returns"].masked_select(~response_loss_mask),
        torch.zeros_like(data.batch["returns"].masked_select(~response_loss_mask)),
    )


def test_build_nimloth_verl_trajectory_row_preserves_cross_turn_gae() -> None:
    from nimloth.training.rl.verl_adapter import (
        build_nimloth_verl_trajectory_replay_row,
        build_verl_replay_dataproto,
    )
    from vagen.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage

    response_a = [20, 21, 30, 31, 32, 40, 33]
    response_b = [20, 22, 30, 31, 32, 41, 33]
    transcript = torch.tensor(
        [1, 2, *response_a, 5, 6, *response_b, 7, 8], dtype=torch.long
    )
    row = build_nimloth_verl_trajectory_replay_row(
        trajectory_id="episode-1",
        transcript_input_ids=transcript,
        transcript_attention_mask=torch.ones_like(transcript),
        transcript_position_ids=torch.arange(transcript.numel()).repeat(3, 1),
        thought_token_ids_by_turn=[[20, 21], [20, 22]],
        latent_query_token_ids=[30, 31],
        action_start_token_id=32,
        action_token_ids=[40, 41],
        action_end_token_id=33,
        turn_rewards=[0.01, 1.01],
        pad_token_id=0,
        multi_modal_inputs={"pixel_values": torch.ones(2, 4)},
    )
    assert row.prompt_length == 1
    assert row.input_ids.tolist() == [0, *transcript.tolist()]
    policy_positions = row.loss_mask.nonzero(as_tuple=True)[0].tolist()
    assert policy_positions == [3, 4, 8, 12, 13, 17]
    assert row.end_of_response_position_mask.nonzero(as_tuple=True)[0].tolist() == [8, 17]
    assert row.token_level_rewards[8].item() == pytest.approx(0.01)
    assert row.token_level_rewards[17].item() == pytest.approx(1.01)

    data = build_verl_replay_dataproto([row], pad_token_id=0)
    data.batch["token_level_rewards"] = data.batch["token_level_scores"].clone()
    data.batch["values"] = torch.zeros_like(data.batch["token_level_scores"])
    compute_advantage(data, AdvantageEstimator.MASKED_GAE, gamma=1.0, lam=1.0)
    response_mask = data.batch["loss_mask"][:, -transcript.numel():].bool()
    returns = data.batch["returns"].masked_select(response_mask)
    assert returns.tolist() == pytest.approx([1.02, 1.02, 1.02, 1.01, 1.01, 1.01])


def test_build_nimloth_verl_row_masks_only_sampled_thought_and_action() -> None:
    from nimloth.training.rl.verl_adapter import build_nimloth_verl_replay_row

    row = build_nimloth_verl_replay_row(
        trajectory_id="trajectory-7-turn-3",
        prompt_input_ids=torch.tensor([10, 11, 12]),
        prompt_attention_mask=torch.ones(3, dtype=torch.long),
        prompt_position_ids=torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long
        ),
        thought_token_ids=[20, 21],
        latent_query_token_ids=[30, 31],
        action_start_token_id=32,
        action_token_id=40,
        action_end_token_id=33,
        reward=1.25,
        multi_modal_inputs={"pixel_values": torch.ones(1, 4)},
    )
    assert row.prompt_length == 3
    assert row.input_ids.tolist() == [10, 11, 12, 20, 21, 30, 31, 32, 40, 33]
    assert row.loss_mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    assert row.token_level_rewards.tolist() == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.25, 0.0
    ]
    assert row.end_of_response_position_mask.tolist() == [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ]
    assert row.position_ids.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ]


def test_verl_replay_rejects_reward_or_loss_on_prompt_and_empty_policy_mask() -> None:
    from nimloth.training.rl.verl_adapter import (
        VerlReplayRow,
        build_verl_replay_dataproto,
    )

    base = dict(
        trajectory_id="bad",
        input_ids=torch.tensor([1, 2, 3]),
        attention_mask=torch.ones(3, dtype=torch.long),
        position_ids=torch.arange(3),
        prompt_length=1,
        end_of_response_position_mask=torch.tensor([0, 0, 1]),
        multi_modal_inputs=None,
    )
    with pytest.raises(ValueError, match="prompt loss mask"):
        build_verl_replay_dataproto(
            [VerlReplayRow(
                **base,
                loss_mask=torch.tensor([1, 0, 1]),
                token_level_rewards=torch.tensor([0.0, 0.0, 1.0]),
            )],
            pad_token_id=0,
        )
    with pytest.raises(ValueError, match="no sampled policy tokens"):
        build_verl_replay_dataproto(
            [VerlReplayRow(
                **base,
                loss_mask=torch.tensor([0, 0, 0]),
                token_level_rewards=torch.tensor([0.0, 0.0, 1.0]),
            )],
            pad_token_id=0,
        )
    with pytest.raises(ValueError, match="prompt reward"):
        build_verl_replay_dataproto(
            [VerlReplayRow(
                **base,
                loss_mask=torch.tensor([0, 1, 1]),
                token_level_rewards=torch.tensor([1.0, 0.0, 0.0]),
            )],
            pad_token_id=0,
        )


def test_verl_replay_requires_uniform_position_id_rank() -> None:
    from nimloth.training.rl.verl_adapter import build_verl_replay_dataproto

    row_1d = _row(
        trajectory_id="one",
        input_ids=[1, 2],
        prompt_length=1,
        loss_mask=[0, 1],
        reward_positions={1: 1.0},
        position_channels=1,
    )
    row_3d = _row(
        trajectory_id="three",
        input_ids=[1, 2],
        prompt_length=1,
        loss_mask=[0, 1],
        reward_positions={1: 1.0},
        position_channels=3,
    )
    with pytest.raises(ValueError, match="position_ids rank"):
        build_verl_replay_dataproto([row_1d, row_3d], pad_token_id=0)
