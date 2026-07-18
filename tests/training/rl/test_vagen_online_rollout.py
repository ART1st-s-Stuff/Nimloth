from __future__ import annotations

import numpy as np
import pytest
import torch


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.mapping = {
            "<|latent_state|>": 30,
            "<|latent_state_1|>": 31,
            "<|action_start|>": 32,
            "<|action_end|>": 33,
            **{f"<|action_({index})|>": 40 + index for index in range(8)},
        }

    def convert_tokens_to_ids(self, token: str):
        return self.mapping.get(token, -1)

    @property
    def unk_token_id(self):
        return -1

    def encode(self, text: str, add_special_tokens: bool = False):
        if text == "<think>inspect</think>":
            return [20, 21]
        if text in self.mapping:
            return [self.mapping[text]]
        raise AssertionError(f"unexpected encode: {text!r}")

    def decode(self, token_ids, **_kwargs):
        if list(token_ids) == [20, 21]:
            return "<think>inspect</think>"
        raise AssertionError(f"unexpected decode: {token_ids!r}")


class _Actor:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, batch):
        from verl import DataProto

        self.calls.append(batch)
        sampling = batch.meta_info["sampling_params"]
        if sampling["max_tokens"] != 1:
            return DataProto.from_dict(
                tensors={
                    "responses": torch.tensor([[20, 21, 0, 0]]),
                    "response_lengths": torch.tensor([2]),
                }
            )
        return DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[43, 0, 0, 0]]),
                "response_lengths": torch.tensor([1]),
            }
        )


def test_staged_rollout_generates_thought_then_forced_action() -> None:
    from nimloth.latent.extraction import LatentActionTokens
    from nimloth.training.rl.vagen_online_rollout import (
        _NimlothStagedRolloutMixin,
    )
    from verl import DataProto

    manager = object.__new__(_NimlothStagedRolloutMixin)
    manager.tokenizer = _Tokenizer()
    manager.actor_rollout_wg = _Actor()
    manager.latent_token_count = 2
    manager.max_think_tokens = 64
    manager._tokens = LatentActionTokens()
    manager._token_ids = dict(manager.tokenizer.mapping)
    manager._query_ids = [30, 31]
    manager._action_ids = list(range(40, 48))
    manager._action_id_to_index = {
        token_id: index for index, token_id in enumerate(manager._action_ids)
    }
    raw = np.empty(1, dtype=object)
    raw[0] = [10, 11]
    generation = DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.zeros((1, 1), dtype=torch.long),
            "position_ids": torch.zeros((1, 1), dtype=torch.long),
        },
        non_tensors={"raw_prompt_ids": raw},
    )

    thoughts, actions, policy, env = manager._generate_staged_responses(generation)

    assert thoughts == [[20, 21]]
    assert actions == [3]
    assert policy == [
        "<think>inspect</think><|latent_state|><|latent_state_1|>"
        "<|action_start|><|action_(3)|><|action_end|>"
    ]
    assert env == ["<think>inspect</think><action>move_left</action>"]
    assert manager.actor_rollout_wg.calls[0].meta_info["sampling_params"] == {
        "max_tokens": 64,
        "stop": ["</think>"],
        "include_stop_str_in_output": True,
        "detokenize": True,
    }
    action_call = manager.actor_rollout_wg.calls[1]
    assert action_call.non_tensor_batch["raw_prompt_ids"][0] == [
        10, 11, 20, 21, 30, 31, 32
    ]
    assert action_call.meta_info["sampling_params"]["allowed_token_ids"] == list(
        range(40, 48)
    )


def test_pinned_vagen_verl_exposes_staged_rollout_hooks() -> None:
    from pathlib import Path

    worker = Path(
        "external/VAGEN/verl/verl/workers/fsdp_workers.py"
    ).read_text(encoding="utf-8")
    assert "sampling_overrides = prompts.meta_info.get('sampling_params', {})" in worker
    assert "unknown_sampling_overrides" in worker
    rollout = Path(
        "external/VAGEN/verl/verl/workers/rollout/vllm_rollout/"
        "vllm_rollout_spmd.py"
    ).read_text(encoding="utf-8")
    assert "'response_lengths': response_lengths" in rollout
    trainer = Path(
        "external/VAGEN/vagen/trainer/ppo/ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert "def resolve_rollout_manager_class" in trainer
    assert 'manager_kwargs["split"] = "train"' in trainer
    assert 'manager_kwargs["split"] = "val"' in trainer


def test_staged_rollout_rejects_missing_response_length() -> None:
    from nimloth.training.rl.vagen_online_rollout import _response_token_ids
    from verl import DataProto

    output = DataProto.from_dict(tensors={"responses": torch.tensor([[20, 21]])})
    with pytest.raises(ValueError, match="requires vLLM response_lengths"):
        _response_token_ids(output, 0)
