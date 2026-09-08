from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage1.checkpoint import (
    COMMITTED_MARKER,
    initialize_new_scheduler_segment,
    prune_numbered_checkpoints,
    publish_checkpoint_alias,
    save_resume_checkpoint,
    update_early_stopping,
    validate_new_scheduler_segment_source,
    validate_resume_state,
)


def _source_state(identity: dict[str, object]) -> dict[str, object]:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    return {
        "training_stage": "format",
        "world_size": 1,
        "epoch": 1,
        "step": 20,
        "best_val": 5.593,
        "lora": True,
        "identity": identity,
        "optimizer": optimizer.state_dict(),
    }


def test_new_segment_restores_optimizer_moments_but_keeps_fresh_scheduler(
    tmp_path,
) -> None:
    identity = {
        "stage": "format",
        "world_size": 4,
        "epochs": 1,
        "seed": 42,
        "batch_size": 1,
        "grad_accum": 8,
    }
    source = tmp_path / "epoch_001"
    source.mkdir()
    (source / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    state = _source_state(identity)
    state["world_size"] = 4
    validate_new_scheduler_segment_source(
        state,
        source,
        expected_stage="format",
        current_identity={
            "stage": "format",
            "world_size": 8,
            "epochs": 19,
            "seed": 42,
            "batch_size": 1,
            "grad_accum": 4,
        },
    )

    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    state["optimizer"]["param_groups"][0]["lr"] = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=19)
    fresh_scheduler_state = scheduler.state_dict()
    start_epoch, global_step, best_val = initialize_new_scheduler_segment(
        state, optimizer, fresh_lrs=[0.1]
    )

    assert (start_epoch, global_step, best_val) == (2, 20, 5.593)
    assert optimizer.state[parameter]["step"].item() == 1
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.tensor([0.2]))
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["initial_lr"] == 0.1
    assert scheduler.state_dict() == fresh_scheduler_state


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda state: state.update(resume_schema="nimloth_early_stage_resume_v1"),
            "mid-epoch",
        ),
        (lambda state: state.pop("optimizer"), "optimizer moments"),
        (lambda state: state["identity"].update(seed=7), "identity mismatch"),
    ],
)
def test_new_segment_source_fails_closed(tmp_path, mutation, message) -> None:
    identity = {
        "stage": "format",
        "world_size": 4,
        "epochs": 1,
        "seed": 42,
        "batch_size": 1,
        "grad_accum": 8,
    }
    source = tmp_path / "epoch_001"
    source.mkdir()
    (source / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    state = _source_state(identity)
    state["world_size"] = 4
    mutation(state)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_new_scheduler_segment_source(
            state,
            source,
            expected_stage="format",
            current_identity={
                "stage": "format",
                "world_size": 8,
                "epochs": 19,
                "seed": 42,
                "batch_size": 1,
                "grad_accum": 4,
            },
        )


def test_new_segment_rejects_effective_batch_change(tmp_path) -> None:
    identity = {
        "stage": "format",
        "world_size": 4,
        "epochs": 1,
        "seed": 42,
        "batch_size": 1,
        "grad_accum": 8,
    }
    source = tmp_path / "epoch_001"
    source.mkdir()
    (source / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    state = _source_state(identity)
    state["world_size"] = 4

    with pytest.raises(ValueError, match="preserve effective batch size"):
        validate_new_scheduler_segment_source(
            state,
            source,
            expected_stage="format",
            current_identity={
                "stage": "format",
                "world_size": 8,
                "epochs": 19,
                "seed": 42,
                "batch_size": 1,
                "grad_accum": 8,
            },
        )


def test_early_stopping_uses_absolute_delta_and_consecutive_patience() -> None:
    best, bad = 5.593, 0
    best, bad, improved, stop = update_early_stopping(
        val_loss=5.5925,
        best_val=best,
        bad_epochs=bad,
        patience=3,
        min_delta=0.001,
    )
    assert (best, bad, improved, stop) == (5.593, 1, False, False)
    best, bad, improved, stop = update_early_stopping(
        val_loss=5.591,
        best_val=best,
        bad_epochs=bad,
        patience=3,
        min_delta=0.001,
    )
    assert (best, bad, improved, stop) == (5.591, 0, True, False)
    for expected_bad, expected_stop in ((1, False), (2, False), (3, True)):
        best, bad, improved, stop = update_early_stopping(
            val_loss=5.5905,
            best_val=best,
            bad_epochs=bad,
            patience=3,
            min_delta=0.001,
        )
        assert (bad, improved, stop) == (expected_bad, False, expected_stop)


def test_same_segment_resume_rejects_changed_segment_identity() -> None:
    expected = {
        "stage": "format",
        "scheduler_segment": {"source_step": 20, "additional_epochs": 19},
    }
    state = {
        "resume_schema": "nimloth_early_stage_resume_v1",
        "identity": {
            "stage": "format",
            "scheduler_segment": {"source_step": 20, "additional_epochs": 20},
        },
        "world_size": 1,
        "micro_accum": 0,
        "epoch": 2,
        "next_micro_batch": 3,
        "rank_rng_states": [SimpleNamespace().__dict__],
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_state(state, expected_identity=expected, rank=0, world=1)


def test_resume_checkpoint_persists_segment_identity_and_patience(tmp_path) -> None:
    class Model(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1)
            self.config = SimpleNamespace(nimloth_training_stage="format")

        def save_pretrained(self, path, **_kwargs) -> None:
            torch.save(self.state_dict(), path / "model.pt")

    class Processor:
        def save_pretrained(self, path) -> None:
            (path / "processor.json").write_text("{}\n", encoding="utf-8")

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    segment = {
        "schema": "nimloth_scheduler_segment_v1",
        "source_epoch": 1,
        "additional_epochs": 19,
    }
    identity = {"stage": "format", "scheduler_segment": segment}
    checkpoint = save_resume_checkpoint(
        model,
        Processor(),
        tmp_path,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=25,
        epoch=2,
        next_micro_batch=40,
        best_val=5.5,
        identity=identity,
        rank=0,
        world=1,
        lora=False,
        base_model_path=tmp_path / "base",
        latent_token_count=1,
        mask_latent_query_labels=False,
        latent_query_mode="generate",
        scheduler_segment=segment,
        segment_bad_epochs=2,
    )
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    assert state["scheduler_segment"] == segment
    assert state["segment_bad_epochs"] == 2
    validate_resume_state(state, expected_identity=identity, rank=0, world=1)


def test_checkpoint_retention_and_atomic_alias_stay_inside_new_run(tmp_path) -> None:
    for step in (5, 10, 15):
        checkpoint = tmp_path / f"resume_step_{step:08d}"
        checkpoint.mkdir()
        (checkpoint / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    incomplete = tmp_path / "resume_step_00000020"
    incomplete.mkdir()

    prune_numbered_checkpoints(tmp_path, "resume_step_", keep=2)

    assert not (tmp_path / "resume_step_00000005").exists()
    assert (tmp_path / "resume_step_00000010").is_dir()
    assert (tmp_path / "resume_step_00000015").is_dir()
    assert incomplete.is_dir()

    epoch = tmp_path / "epoch_002"
    epoch.mkdir()
    (epoch / COMMITTED_MARKER).write_text("{}\n", encoding="utf-8")
    publish_checkpoint_alias(tmp_path, "best", epoch)
    assert (tmp_path / "best").resolve() == epoch.resolve()
