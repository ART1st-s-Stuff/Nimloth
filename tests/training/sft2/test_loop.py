"""SFT2 循环恢复状态测试。"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from nimloth.training.sft2.loop import (
    SFT2LoopState,
    SFT2TrainingLoop,
    load_sft2_loop_state,
)


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-3)


def test_load_loop_state_restores_partial_epoch(tmp_path) -> None:
    optimizer = _optimizer()
    state_path = tmp_path / "training_state.pt"
    torch.save(
        {
            "step": 7,
            "epoch": 3,
            "epoch_complete": False,
            "micro_step_in_epoch": 4,
            "best_val_wm_mse": 0.25,
            "training_invariants": {"seed": 42},
            "optimizer": optimizer.state_dict(),
        },
        state_path,
    )

    state = load_sft2_loop_state(
        resume=True,
        resume_state_path=state_path,
        resume_checkpoint_dir=tmp_path,
        optimizer=optimizer,
        training_invariants={"seed": 42},
    )

    assert state.global_step == 7
    assert state.start_epoch == 3
    assert state.resume_micro_step == 4
    assert state.best_val_wm_mse == 0.25


def test_load_loop_state_rejects_invariant_mismatch(tmp_path) -> None:
    state_path = tmp_path / "training_state.pt"
    torch.save(
        {
            "training_invariants": {"world_size": 2},
        },
        state_path,
    )

    with pytest.raises(ValueError, match="training invariants mismatch"):
        load_sft2_loop_state(
            resume=True,
            resume_state_path=state_path,
            resume_checkpoint_dir=tmp_path,
            optimizer=_optimizer(),
            training_invariants={"world_size": 1},
        )


def test_train_microbatch_backwards_primary_before_sigreg_forward() -> None:
    events: list[str] = []
    current_state = torch.randn(2, 4, requires_grad=True)

    class FakeAlgorithm:
        has_sigreg_stage = True

        def wm_weight(self, _global_step: int, _total_steps: int) -> float:
            return 0.5

        def training_primary_step(self, _runtime, batch, *, wm_weight: float):
            events.append("primary_forward")
            assert batch == "prepared"
            assert wm_weight == 0.5
            return SimpleNamespace(
                current_state=current_state,
                metrics={"total_loss": 2.0},
                sample_count=2,
                loss=torch.tensor(2.0, requires_grad=True),
            )

        def training_sigreg_step(
            self,
            _runtime,
            batch,
            *,
            detached_current_state: torch.Tensor,
            sigreg_seed: int,
        ):
            events.append("sigreg_forward")
            assert batch == "prepared"
            assert detached_current_state.requires_grad is False
            assert sigreg_seed == 1_010_052
            return SimpleNamespace(
                loss=torch.tensor(0.3, requires_grad=True),
                metrics={"sigreg_loss": 3.0},
            )

        def merge_training_metrics(self, primary_metrics, sigreg):
            events.append("merge_metrics")
            return {
                **primary_metrics,
                **sigreg.metrics,
                "total_loss": primary_metrics["total_loss"] + sigreg.loss.item(),
            }

    class FakeOptimizationRuntime:
        def backward(self, _loss: torch.Tensor, *, grad_accum: int) -> None:
            events.append("backward")
            assert grad_accum == 4

    class FakeBatchBuilder:
        def prepare(self, samples):
            events.append("prepare")
            assert samples == "raw"
            return "prepared"

    loop = SFT2TrainingLoop(
        config=SimpleNamespace(
            step_timing=False,
            step_timing_interval=1,
            grad_accum=4,
            seed=42,
        ),
        rank=0,
        train_loader=[],
        val_loader=[],
        train_batch_sampler=None,
        algorithm=FakeAlgorithm(),
        model_runtime=object(),
        optimization_runtime=FakeOptimizationRuntime(),
        batch_builder=FakeBatchBuilder(),
        checkpoint_runtime=None,
        reporter=None,
        state=SFT2LoopState(),
        total_steps=10,
    )

    wm_weight, metrics, sample_count = loop._train_microbatch(
        "raw",
        epoch=1,
        micro_step=1,
    )

    assert events == [
        "prepare",
        "primary_forward",
        "backward",
        "sigreg_forward",
        "backward",
        "merge_metrics",
    ]
    assert wm_weight == 0.5
    assert metrics["total_loss"] == pytest.approx(2.3)
    assert sample_count == 2
