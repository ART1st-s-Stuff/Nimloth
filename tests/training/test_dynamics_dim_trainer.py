from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from nimloth.training.wm_heads.dynamics_dim_trainer import (
    DynamicsDimTrainer,
    DynamicsTrainerConfig,
    EpochBatchSampler,
)
from nimloth.wm.dynamics_dim_heads import DynamicsDimHeadSpec, DynamicsDimWMHeads


class TinyFlatTransitions(Dataset):
    def __init__(self, count: int = 12) -> None:
        generator = torch.Generator().manual_seed(81)
        self.states = torch.randn(count, 32, generator=generator).half()
        self.actions = torch.arange(count) % 8
        self.targets = (self.states + self.actions[:, None] / 20).half()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        return {"state": self.states[index], "next_state": self.targets[index], "action": self.actions[index], "id": f"row:{index}"}


def tiny_heads() -> DynamicsDimWMHeads:
    spec = DynamicsDimHeadSpec(external_dim=32, full_dynamics_dim=32, factorized_dynamics_dim=8, predictor_hidden_dim=8, predictor_depth=1, predictor_heads=2, predictor_mlp_dim=16, history_size=1)
    return DynamicsDimWMHeads.create(spec)


def test_epoch_sampler_visits_every_row_once_per_epoch_and_resumes() -> None:
    sampler = EpochBatchSampler(size=10, batch_size=4, epochs=5, seed=17)
    first = sampler.next_indices()
    saved = sampler.state_dict()
    expected = sampler.next_indices()
    resumed = EpochBatchSampler(size=10, batch_size=4, epochs=5, seed=17)
    resumed.load_state_dict(saved)
    batches = [first, expected]
    while not sampler.done:
        batches.append(sampler.next_indices())

    assert torch.equal(resumed.next_indices(), expected)
    assert len(batches) == 15
    visits = torch.bincount(torch.cat(batches), minlength=10)
    assert visits.tolist() == [5] * 10
    assert sampler.epoch == 5


def test_dynamics_trainer_uses_shared_batch_and_resumes_exactly(tmp_path: Path) -> None:
    dataset = TinyFlatTransitions()
    config = DynamicsTrainerConfig(seed=31, batch_size=4, epochs=2, learning_rate=0.001, weight_decay=0.0)
    torch.manual_seed(config.seed)
    trainer = DynamicsDimTrainer.create(tiny_heads(), dataset, config, torch.device("cpu"))

    first = trainer.train_step()
    trainer.save_checkpoint(tmp_path)
    expected = trainer.train_step()
    expected_state = {key: value.clone() for key, value in trainer.heads.state_dict().items()}
    resumed = DynamicsDimTrainer.resume(tmp_path, dataset, torch.device("cpu"))
    actual = resumed.train_step()
    metrics = resumed.evaluate(dataset)

    assert first["sample_ids"] == ["row:2", "row:11", "row:0", "row:6"]
    assert actual["sample_ids"] == expected["sample_ids"]
    assert all(torch.equal(expected_state[key], value) for key, value in resumed.heads.state_dict().items())
    assert set(metrics) == {"full", "factorized"}
    assert set(metrics["full"]) == {"mse", "cosine", "shuffled_mse", "shuffled_cosine"}
    assert all(torch.isfinite(torch.tensor(value)) for branch in metrics.values() for value in branch.values())
