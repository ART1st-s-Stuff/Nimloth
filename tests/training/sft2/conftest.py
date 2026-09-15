"""Small real-schema inputs for trajectory-native Stage3 interface tests."""
from types import SimpleNamespace

import pytest
import torch

from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.rollout.transitions import TransitionSample
from nimloth.training.sft.stage3.data.trajectory import TrajectoryItem, TrajectoryCollator


@pytest.fixture
def trajectory_factory():
    tokenizer = SimpleNamespace(pad_token_id=0,
        convert_tokens_to_ids=lambda token: {"<|latent_state|>": 2, "<|latent_state_1|>": 3}.get(token, -1))
    builder = Qwen25VLInputBuilder(SimpleNamespace(tokenizer=tokenizer, image_token=None),
                                  max_length=256, latent_token_count=2)
    def make(record="a", length=6, horizon=4, success=True, weight=1.0, missing_outcome=False):
        samples, encodings = [], []
        for step in range(length + 1):
            ids = torch.tensor([token for _ in range(step + 1) for token in (7, 2, 3, 8)])
            row = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            if step < length:
                labels = torch.full_like(ids, -100)
                labels[-1] = ids[-1]
                row["labels"] = labels
                samples.append(TransitionSample(record, step, [], [], step % 3,
                    f"{record}_{step}.png", f"{record}_{step+1}.png", [], [],
                    action_value_target=length-step, success=success,
                    action_success=None if missing_outcome else bool(step % 2)))
            encodings.append(row)
        item = TrajectoryItem(tuple(samples), tuple(encodings), weight)
        encoded = TrajectoryCollator(builder, prediction_horizon=horizon)([item])[0]
        return builder, item, encoded
    return make
