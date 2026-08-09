from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.rl.planner_verl_factory import (
    _planner_fsdp_wrap_policy,
    build_planner_worker_components,
)


def _trainer_args() -> dict[str, object]:
    return {
        "model": "/weights/id147/train/latest",
        "max_pixels": 256 * 256,
        "attn_implementation": "flash_attention_2",
        "llm_tune": "full",
        "vision_tune": "freeze",
        "gradient_checkpointing": True,
        "wm_checkpoint": "/weights/id147/train/latest/wm_predictor",
        "state_proj_checkpoint": "/weights/id147/train/latest/state_proj.pt",
        "value_head_checkpoint": "/weights/id147/train/latest/value_head",
        "planner_policy_head_checkpoint": (
            "/weights/id147/train/latest/planner_policy_head"
        ),
    }


class StateProjector(torch.nn.Module):
    pass


class WMPredictor(torch.nn.Module):
    pass


class ValueHead(torch.nn.Module):
    pass


class PlannerPolicyHead(torch.nn.Module):
    pass


def test_planner_fsdp_wrap_policy_separates_fp32_auxiliary_owners() -> None:
    model = SimpleNamespace(_no_split_modules=("QwenDecoderLayer",))
    world_model = SimpleNamespace(
        state_proj=StateProjector(),
        wm_predictor=WMPredictor(),
        value_head=ValueHead(),
        planner_policy_head=PlannerPolicyHead(),
    )

    assert _planner_fsdp_wrap_policy(model, world_model) == {
        "transformer_layer_cls_to_wrap": ("QwenDecoderLayer",),
        "fp32_module_paths": (
            "agent.wm.state_proj",
            "agent.wm.wm_predictor",
            "agent.wm.value_head",
            "agent.wm.planner_policy_head",
        ),
    }


def test_planner_worker_factory_rejects_optimizer_resume_before_model_load(
    tmp_path: Path,
) -> None:
    trainer_args = _trainer_args()
    trainer_args["resume"] = True

    with pytest.raises(ValueError, match="weights-only initialization"):
        build_planner_worker_components(
            config={
                "rl_config_path": str(tmp_path / "missing.yaml"),
                "trainer_args": trainer_args,
            },
            device=torch.device("cuda", 0),
            rank=0,
            world_size=4,
        )


def test_planner_worker_factory_requires_explicit_artifact_contract(
    tmp_path: Path,
) -> None:
    trainer_args = _trainer_args()
    del trainer_args["planner_policy_head_checkpoint"]

    with pytest.raises(ValueError, match="planner_policy_head_checkpoint"):
        build_planner_worker_components(
            config={
                "rl_config_path": str(tmp_path / "missing.yaml"),
                "trainer_args": trainer_args,
            },
            device=torch.device("cuda", 0),
            rank=0,
            world_size=4,
        )
