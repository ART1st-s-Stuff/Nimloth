from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from nimloth.training.wm_heads.data import DeterministicBatchStream, FrozenStateTransitions
from nimloth.training.wm_heads.trainer import MatchedTrainerConfig, MatchedWMTrainer
from nimloth.wm.matched_heads import MatchedHeadSpec, MatchedWMHeads


class TinyTransitions(Dataset):
    def __init__(self, count: int = 12) -> None:
        generator = torch.Generator().manual_seed(71)
        self.states = torch.randn(count, 8, 4, generator=generator)
        self.actions = torch.arange(count) % 8
        effects = self.actions.float().view(-1, 1, 1) / 20
        self.targets = self.states + effects

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        return {"state": self.states[index], "next_state": self.targets[index], "action": self.actions[index], "id": f"row:{index}"}


def write_source_cache(path: Path) -> None:
    rows, states = [], []
    for record in range(2):
        for step in range(3):
            rows.append({"id": f"r{record}:{step}", "record_id": f"r{record}", "step_index": step, "action_index": step})
            states.append(torch.full((8, 4), record * 10 + step, dtype=torch.float16))
    torch.save({"state_emb": torch.stack(states), "rows": rows}, path / "shard.pt")
    manifest = {"count": 6, "cond_dim": 32, "state_dtype": "float16", "compression": "none", "shard_size": 6, "shards": [{"file": "shard.pt", "count": 6}], "fingerprint": "fixture"}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def tiny_spec() -> MatchedHeadSpec:
    return MatchedHeadSpec(state_tokens=8, token_dim=4, vector_hidden_dim=8, token_hidden_dim=8, depth=1, heads=2, mlp_ratio=2)


def test_frozen_state_transitions_drop_only_terminal_rows(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    write_source_cache(cache)

    dataset = FrozenStateTransitions(cache)

    assert len(dataset) == 4
    assert dataset.transition_ids == ("r0:0", "r0:1", "r1:0", "r1:1")
    assert torch.equal(dataset[0]["next_state"], torch.full((8, 4), 1.0, dtype=torch.float16))
    assert dataset[2]["id"] == "r1:0"


def test_batch_stream_resumes_exact_order() -> None:
    stream = DeterministicBatchStream(size=11, batch_size=4, seed=19)
    stream.next_indices()
    saved = stream.state_dict()
    expected = stream.next_indices()

    resumed = DeterministicBatchStream(size=11, batch_size=4, seed=19)
    resumed.load_state_dict(saved)

    assert torch.equal(resumed.next_indices(), expected)


def test_trainer_uses_shared_ids_reports_controls_and_resumes(tmp_path: Path) -> None:
    dataset = TinyTransitions()
    config = MatchedTrainerConfig(seed=23, batch_size=4, learning_rate=0.001, weight_decay=0.0)
    torch.manual_seed(config.seed)
    trainer = MatchedWMTrainer.create(MatchedWMHeads.create(tiny_spec()), dataset, config, torch.device("cpu"))

    first = trainer.train_step()
    trainer.save_checkpoint(tmp_path, "latest")
    trainer.save_checkpoint(tmp_path, "best")
    trainer.save_checkpoint(tmp_path, "final")
    expected = trainer.train_step()
    expected_state = {key: value.detach().clone() for key, value in trainer.heads.state_dict().items()}
    resumed = MatchedWMTrainer.resume(tmp_path / "latest", dataset, torch.device("cpu"))
    actual = resumed.train_step()
    metrics = resumed.evaluate(dataset)

    assert first["sample_ids"] == ["row:3", "row:5", "row:8", "row:4"]
    assert actual["sample_ids"] == expected["sample_ids"]
    assert all(torch.equal(expected_state[key], value) for key, value in resumed.heads.state_dict().items())
    assert set(metrics) == {"vector", "token"}
    assert set(metrics["vector"]) == {"mse", "cosine", "shuffled_mse", "shuffled_cosine"}
    assert all(torch.isfinite(torch.tensor(value)) for branch in metrics.values() for value in branch.values())
    assert all((tmp_path / tag / "trainer.pt").is_file() for tag in ("best", "latest", "final"))
