from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.rl.algorithm import PLANNER_TRAINING_OBJECTIVE
from nimloth.training.rl.checkpoint import (
    _save_collected_fsdp_backbone,
    link_checkpoint_snapshot,
    save_rl_checkpoint,
)
from nimloth.training.rl.trainer import _load_resume_state


def test_checkpoint_snapshot_hardlinks_complete_tree(tmp_path: Path) -> None:
    source = tmp_path / "latest"
    (source / "wm_predictor").mkdir(parents=True)
    (source / "rl_state.pt").write_bytes(b"optimizer")
    (source / "wm_predictor" / "predictor.pt").write_bytes(b"predictor")

    snapshot = tmp_path / "iter_0001"
    link_checkpoint_snapshot(source, snapshot)

    for relative in (Path("rl_state.pt"), Path("wm_predictor/predictor.pt")):
        source_file = source / relative
        snapshot_file = snapshot / relative
        assert snapshot_file.read_bytes() == source_file.read_bytes()
        assert os.stat(snapshot_file).st_ino == os.stat(source_file).st_ino


def test_fsdp_export_uses_collected_state_without_reading_live_wrapper(
    tmp_path: Path,
) -> None:
    class RawModel:
        saved_state: dict[str, torch.Tensor] | None = None

        def save_pretrained(
            self,
            output_dir: Path,
            *,
            state_dict: dict[str, torch.Tensor],
            safe_serialization: bool,
        ) -> None:
            assert output_dir == tmp_path
            assert safe_serialization is True
            self.saved_state = state_dict

    class LiveWrapper:
        def __init__(self, module: RawModel) -> None:
            self.module = module

        def state_dict(self) -> dict[str, torch.Tensor]:
            raise AssertionError("export must not start a second FSDP collective")

    raw = RawModel()
    state = {
        "model.embed_tokens.weight": torch.zeros(4, 2),
        "model.embed_tokens.nimloth_query_rows": torch.tensor([[1.0, 2.0]]),
        "model.embed_tokens.nimloth_protocol_rows": torch.tensor([[3.0, 4.0]]),
        "model.embed_tokens.nimloth_query_ids": torch.tensor([1]),
        "model.embed_tokens.nimloth_protocol_ids": torch.tensor([2]),
        "lm_head.weight": torch.zeros(4, 2),
        "lm_head.nimloth_query_rows": torch.tensor([[5.0, 6.0]]),
        "lm_head.nimloth_protocol_rows": torch.tensor([[7.0, 8.0]]),
        "lm_head.nimloth_query_ids": torch.tensor([1]),
        "lm_head.nimloth_protocol_ids": torch.tensor([2]),
    }
    agent = SimpleNamespace(
        backbone=SimpleNamespace(model=LiveWrapper(raw)),
    )

    _save_collected_fsdp_backbone(agent, tmp_path, state)  # type: ignore[arg-type]

    assert raw.saved_state is not None
    assert not any("nimloth_" in key for key in raw.saved_state)
    assert torch.equal(
        raw.saved_state["model.embed_tokens.weight"][1:3],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(
        raw.saved_state["lm_head.weight"][1:3],
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    saved_rows = torch.load(
        tmp_path / "selected_token_rows.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(saved_rows) == {key for key in state if ".nimloth_" in key}


def test_rl_checkpoint_reads_fsdp_wrapper_state_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from nimloth.training.rl import checkpoint

    events: list[str] = []

    class RawModel:
        def save_pretrained(self, _output_dir: Path, **kwargs: object) -> None:
            assert set(kwargs) == {"state_dict", "safe_serialization"}
            events.append("raw_save")

    class WrappedModel:
        module = RawModel()

        def state_dict(self) -> dict[str, torch.Tensor]:
            events.append("collect")
            return {"weight": torch.ones(2)}

    class Backbone:
        model = WrappedModel()

        def save_pretrained(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("FSDP export must not reread the live backbone")

    class CheckpointModule(torch.nn.Module):
        def save_checkpoint(self, _path: Path) -> None:
            events.append("head_save")

    class Processor:
        def save_pretrained(self, _path: Path) -> None:
            events.append("processor_save")

    @contextmanager
    def state_dict_type(*_args: object, **_kwargs: object):
        events.append("collective")
        yield

    monkeypatch.setattr(checkpoint, "_is_fsdp", lambda _model: True)
    monkeypatch.setattr(FSDP, "state_dict_type", state_dict_type)
    agent = SimpleNamespace(
        backbone=Backbone(),
        wm=SimpleNamespace(
            state_proj=torch.nn.Linear(1, 1),
            wm_predictor=CheckpointModule(),
            value_head=CheckpointModule(),
            planner_policy_head=None,
            outcome_head=None,
        ),
    )

    save_rl_checkpoint(
        tmp_path,
        agent=agent,  # type: ignore[arg-type]
        processor=Processor(),
        vision_ema=None,
    )

    assert events.count("collect") == 1
    assert events.index("collect") < events.index("raw_save")


def test_resume_rejects_old_incoming_action_planner_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    state = {
        "optimizer": optimizer.state_dict(),
        "planner_config": {"enabled": True},
        "planner_training_objective": "receding_horizon_transition_mc_v1",
        "reference_kl_config": {"weight": 0.0, "type": None},
        "train_world_model": True,
    }
    monkeypatch.setattr(
        "nimloth.training.rl.trainer.load_rl_wm_checkpoint",
        lambda *_args, **_kwargs: state,
    )

    with pytest.raises(ValueError, match="planner training objective mismatch"):
        _load_resume_state(
            checkpoint_dir=tmp_path,
            world_model=object(),  # type: ignore[arg-type]
            optimizer=optimizer,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            optimizer_state_sharded=False,
            expected_checkpoint_metric="success_rate",
            expected_credit_assignment="none",
            expected_token_credit_config={},
            expected_truncated_bootstrap=None,
            expected_planner_config={"enabled": True},
            expected_planner_training_objective=PLANNER_TRAINING_OBJECTIVE,
            expected_planner_value_config={
                "ppo_clip_range": 0.2,
                "ppo_epochs": 4,
            },
            expected_reference_kl_config={"weight": 0.0, "type": None},
            expected_train_world_model=True,
        )


def test_replicated_optimizer_can_resume_across_training_world_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_parameter = torch.nn.Parameter(torch.ones(()))
    source_optimizer = torch.optim.AdamW([source_parameter])
    source_parameter.square().backward()
    source_optimizer.step()
    planner_config = {"enabled": True}
    planner_value_config = {"ppo_clip_range": 0.2, "ppo_epochs": 4}
    state = {
        "iteration": 5,
        "global_step": 5,
        "optimizer": source_optimizer.state_dict(),
        "optimizer_world_size": 1,
        "training_world_size": 2,
        "optimizer_state_layout": "replicated",
        "planner_config": planner_config,
        "planner_training_objective": PLANNER_TRAINING_OBJECTIVE,
        "planner_value_config": planner_value_config,
        "reference_kl_config": {"weight": 0.0, "type": None},
        "train_world_model": True,
    }
    monkeypatch.setattr(
        "nimloth.training.rl.trainer.load_rl_wm_checkpoint",
        lambda *_args, **_kwargs: state,
    )
    resumed_parameter = torch.nn.Parameter(torch.ones(()))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter])

    resumed = _load_resume_state(
        checkpoint_dir=tmp_path,
        world_model=object(),  # type: ignore[arg-type]
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        rank=7,
        world_size=16,
        optimizer_state_sharded=False,
        expected_checkpoint_metric="success_rate",
        expected_credit_assignment="none",
        expected_token_credit_config={},
        expected_truncated_bootstrap=None,
        expected_planner_config=planner_config,
        expected_planner_training_objective=PLANNER_TRAINING_OBJECTIVE,
        expected_planner_value_config=planner_value_config,
        expected_reference_kl_config={"weight": 0.0, "type": None},
        expected_train_world_model=True,
    )

    assert resumed.loaded is True
    assert resumed.start_iteration == 6
    assert resumed.global_step == 5
    assert resumed_optimizer.state_dict()["state"]


def test_resume_rejects_changed_planner_value_ppo_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    state = {
        "optimizer": optimizer.state_dict(),
        "planner_config": {"enabled": True},
        "planner_training_objective": PLANNER_TRAINING_OBJECTIVE,
        "planner_value_config": {"ppo_clip_range": 0.2, "ppo_epochs": 2},
        "reference_kl_config": {"weight": 0.0, "type": None},
        "train_world_model": True,
    }
    monkeypatch.setattr(
        "nimloth.training.rl.trainer.load_rl_wm_checkpoint",
        lambda *_args, **_kwargs: state,
    )

    with pytest.raises(ValueError, match="planner value config mismatch"):
        _load_resume_state(
            checkpoint_dir=tmp_path,
            world_model=object(),  # type: ignore[arg-type]
            optimizer=optimizer,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            optimizer_state_sharded=False,
            expected_checkpoint_metric="success_rate",
            expected_credit_assignment="none",
            expected_token_credit_config={},
            expected_truncated_bootstrap=None,
            expected_planner_config={"enabled": True},
            expected_planner_training_objective=PLANNER_TRAINING_OBJECTIVE,
            expected_planner_value_config={
                "ppo_clip_range": 0.2,
                "ppo_epochs": 4,
            },
            expected_reference_kl_config={"weight": 0.0, "type": None},
            expected_train_world_model=True,
        )
