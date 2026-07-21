"""SFT2 循环恢复状态测试。"""

from __future__ import annotations

import torch
import pytest

from nimloth.training.sft2.loop import load_sft2_loop_state


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
