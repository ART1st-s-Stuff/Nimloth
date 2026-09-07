from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler, TensorDataset

from nimloth.training.sft.stage1.checkpoint import (
    COMMITTED_MARKER,
    find_latest_resume_dir,
    restore_rng_state,
    save_resume_checkpoint,
    validate_resume_state,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dropout = torch.nn.Dropout(0.25)
        self.config = SimpleNamespace(nimloth_training_stage="format")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(value))

    def save_pretrained(self, path: Path, **_kwargs) -> None:
        torch.save(self.state_dict(), path / "model.pt")


class TinyProcessor:
    def save_pretrained(self, path: Path) -> None:
        (path / "processor.json").write_text("{}\n", encoding="utf-8")


def _seed() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


def _run(tmp_path: Path, *, interrupt: bool):
    tmp_path.mkdir(parents=True)
    _seed()
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    dataset = TensorDataset(torch.arange(1, 13, dtype=torch.float32).view(-1, 1))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=23, shuffle=True)
    sampler.set_epoch(1)
    ordered = list(iter(sampler))
    identity = {"stage": "format", "world_size": 1, "dataset": "tiny"}
    consumed: list[int] = []
    global_step = 0
    next_batch = 0

    def train_from(cursor: int) -> None:
        nonlocal global_step, next_batch
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for position, index in enumerate(ordered[cursor:], start=cursor):
            consumed.append(index)
            loss = model(dataset[index][0]).square().mean()
            loss.backward()
            micro += 1
            next_batch = position + 1
            if micro == 2:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(micro)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                micro = 0
                if interrupt and global_step == 2:
                    save_resume_checkpoint(
                        model,
                        TinyProcessor(),
                        tmp_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=1,
                        next_micro_batch=next_batch,
                        best_val=1.0,
                        identity=identity,
                        rank=0,
                        world=1,
                        lora=False,
                        base_model_path=tmp_path / "base",
                        latent_token_count=1,
                        mask_latent_query_labels=False,
                        latent_query_mode="generate",
                    )
                    return

    train_from(0)
    if interrupt:
        checkpoint = find_latest_resume_dir(tmp_path)
        assert checkpoint is not None
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=False
        )
        validate_resume_state(state, expected_identity=identity, rank=0, world=1)
        resumed = TinyModel()
        resumed.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=True))
        model = resumed
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.load_state_dict(state["optimizer"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scheduler.load_state_dict(state["scheduler"])
        global_step = state["step"]
        restore_rng_state(state["rank_rng_states"][0])
        train_from(state["next_micro_batch"])
    return model, optimizer, scheduler, consumed, global_step


def test_interrupted_resume_matches_uninterrupted_optimizer_boundary(tmp_path):
    uninterrupted = _run(tmp_path / "full", interrupt=False)
    resumed = _run(tmp_path / "resumed", interrupt=True)
    for left, right in zip(uninterrupted[0].parameters(), resumed[0].parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(uninterrupted[1].state_dict(), resumed[1].state_dict())
    assert uninterrupted[2].state_dict() == resumed[2].state_dict()
    assert resumed[3] == uninterrupted[3]
    assert resumed[4] == uninterrupted[4] == 6


def test_resume_rejects_world_identity_partial_accumulation_and_missing_rank():
    base = {
        "resume_schema": "nimloth_early_stage_resume_v1",
        "identity": {"stage": "format"},
        "world_size": 2,
        "micro_accum": 0,
        "epoch": 1,
        "next_micro_batch": 4,
        "rank_rng_states": [{}, {}],
    }
    validate_resume_state(base, expected_identity=base["identity"], rank=1, world=2)
    with pytest.raises(ValueError, match="world size mismatch"):
        validate_resume_state(base, expected_identity=base["identity"], rank=0, world=1)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_state(
            base, expected_identity={"stage": "query"}, rank=0, world=2
        )
    with pytest.raises(ValueError, match="partial gradient"):
        validate_resume_state(
            {**base, "micro_accum": 1},
            expected_identity=base["identity"],
            rank=0,
            world=2,
        )
    with pytest.raises(ValueError, match="per-rank"):
        validate_resume_state(
            {**base, "rank_rng_states": [{}]},
            expected_identity=base["identity"],
            rank=0,
            world=2,
        )


def test_latest_resume_ignores_half_written_directory(tmp_path):
    incomplete = tmp_path / "resume_step_00000009"
    incomplete.mkdir(parents=True)
    torch.save({}, incomplete / "training_state.pt")
    complete = tmp_path / "resume_step_00000008"
    complete.mkdir()
    torch.save({}, complete / "training_state.pt")
    (complete / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    assert find_latest_resume_dir(tmp_path) == complete


def test_latest_resume_ignores_half_written_epoch(tmp_path):
    incomplete = tmp_path / "epoch_002"
    incomplete.mkdir()
    torch.save(
        {"step": 9, "identity": {"stage": "format"}, "world_size": 4},
        incomplete / "training_state.pt",
    )
    complete = tmp_path / "epoch_001"
    complete.mkdir()
    torch.save({"step": 8}, complete / "training_state.pt")
    (complete / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    assert find_latest_resume_dir(tmp_path) == complete


def test_latest_resume_keeps_legacy_epoch_compatibility(tmp_path):
    legacy = tmp_path / "epoch_001"
    legacy.mkdir()
    torch.save(
        {"step": 8, "epoch": 1, "best_val": 1.0, "lora": True},
        legacy / "training_state.pt",
    )
    assert find_latest_resume_dir(tmp_path) == legacy
