"""SFT2 variant 不能反向污染公共训练编排。"""

from __future__ import annotations

import inspect

import nimloth.training.rl.checkpoint as rl_checkpoint
import nimloth.training.rl.trainer as rl_trainer
import nimloth.training.sft2.checkpoint as sft2_checkpoint
import nimloth.training.sft2.trainer as sft2_trainer
from nimloth.training.sft2.variant import available_sft2_variants, resolve_sft2_variant


FORBIDDEN_CONCRETE_NAMES = (
    "CachedDINOGridTargets",
    "DINOGridBatchAssembler",
    "DINOGridSFT2Algorithm",
    "GridWorldModel",
    "TemporalSpatialGridPredictor",
    "dino_grid_decoder.pt",
    "dino_grid_config.json",
)


def test_common_trainers_and_checkpoints_do_not_name_dino_grid_types() -> None:
    for module in (sft2_trainer, sft2_checkpoint, rl_trainer, rl_checkpoint):
        source = inspect.getsource(module)
        leaked = [name for name in FORBIDDEN_CONCRETE_NAMES if name in source]
        assert leaked == [], f"{module.__name__} leaks variant details: {leaked}"


def test_sft2_variants_are_resolved_behind_one_registry() -> None:
    assert available_sft2_variants() == ("dino_grid", "latent")
    assert resolve_sft2_variant("latent").name == "latent"
    assert resolve_sft2_variant("dino_grid").name == "dino_grid"
