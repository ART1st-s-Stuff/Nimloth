"""Production model factory for the composite Planner VERL/FSDP worker."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from nimloth.agent import Agent
from nimloth.backbone import (
    build_input_builder,
    load_backbone,
    resolve_tune_modes,
)
from nimloth.backbone.qwen25vl.checkpoint import find_visual_module
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.config.rl import load_rl_config
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.planner_verl_worker import (
    PlannerObjectiveModule,
    PlannerWorkerModelComponents,
)
from nimloth.training.rl.trainer import (
    _build_optimizer,
    _build_world_model,
    _prepare_planner_qwen_training,
)


_REQUIRED_TRAINER_ARGS = {
    "model",
    "max_pixels",
    "attn_implementation",
    "llm_tune",
    "vision_tune",
    "gradient_checkpointing",
    "wm_checkpoint",
    "state_proj_checkpoint",
    "value_head_checkpoint",
    "planner_policy_head_checkpoint",
}


def _worker_args(raw: Mapping[str, Any]) -> argparse.Namespace:
    missing = sorted(_REQUIRED_TRAINER_ARGS - set(raw))
    if missing:
        raise ValueError(
            "planner worker trainer_args are missing: " + ", ".join(missing)
        )
    if raw.get("resume", False) or raw.get("resume_checkpoint") is not None:
        raise ValueError(
            "planner VERL migration accepts historical checkpoints only as "
            "weights-only initialization, never optimizer resume"
        )
    values = {
        "lora": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "query_tune": "freeze",
        "resume": False,
        "resume_checkpoint": None,
        **dict(raw),
    }
    return argparse.Namespace(**values)


def _planner_fsdp_wrap_policy(
    model: torch.nn.Module,
    world_model: torch.nn.Module,
) -> dict[str, Any]:
    """Declare BF16 Qwen and FP32 auxiliary FSDP boundaries separately."""

    qwen_layers = getattr(model, "_no_split_modules", None)
    if not qwen_layers:
        raise RuntimeError("Qwen model does not declare FSDP transformer layers")
    auxiliary_modules = (
        world_model.state_proj,
        world_model.wm_predictor,
        world_model.value_head,
        world_model.planner_policy_head,
    )
    if any(module is None for module in auxiliary_modules):
        raise RuntimeError("planner FSDP root is missing an auxiliary module")
    return {
        "transformer_layer_cls_to_wrap": tuple(str(name) for name in qwen_layers),
        "fp32_module_paths": (
            "agent.wm.state_proj",
            "agent.wm.wm_predictor",
            "agent.wm.value_head",
            "agent.wm.planner_policy_head",
        ),
    }


def build_planner_worker_components(
    *,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> PlannerWorkerModelComponents:
    """Load unwrapped weights and construct the one complete planner objective."""

    config_path = config.get("rl_config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("planner worker config requires rl_config_path")
    raw_args = config.get("trainer_args")
    if not isinstance(raw_args, Mapping):
        raise ValueError("planner worker config requires trainer_args mapping")
    args = _worker_args(raw_args)
    rl_config = load_rl_config(Path(config_path))

    if not rl_config.agent.planning.enabled or not rl_config.planner_policy.enabled:
        raise ValueError("planner VERL worker requires PlannerPolicyHead planning")
    if rl_config.planner_policy.ppo_epochs != 1:
        raise ValueError("planner VERL worker requires exactly one optimizer epoch")
    if (
        rl_config.gradient.state_source != "recompute"
        or not rl_config.gradient.representation_to_backbone
    ):
        raise ValueError(
            "planner VERL worker requires differentiable complete-prefix recomputation"
        )
    if rl_config.predictor.lambda_sigreg != 0.0:
        raise ValueError("planner FSDP root does not support external SIGReg")
    if rl_config.distributed.gpus_per_rank != 1:
        raise ValueError(
            "planner VERL FSDP uses one Ray/FSDP process per GPU; "
            "gpus_per_rank must be 1"
        )
    if rl_config.distributed.world_size != world_size:
        raise ValueError(
            "planner worker world size differs from RL config: "
            f"worker={world_size}, config={rl_config.distributed.world_size}"
        )

    llm_tune, vision_tune = resolve_tune_modes(args)
    if llm_tune == "freeze":
        raise ValueError("planner complete-prefix Qwen language body must be trainable")
    if vision_tune != "freeze":
        raise ValueError(
            "planner VERL first gate requires frozen vision parameters and no EMA"
        )

    from transformers import AutoConfig

    latent_token_count = validate_agent_policy_protocol(
        AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    )
    loaded = load_backbone(
        args,
        device=device,
        latent_token_count=latent_token_count,
        model_parallel_size=1,
        resume_dir=None,
        resume_state_path=None,
    )
    if loaded.pair_parallel:
        raise RuntimeError("planner FSDP factory loaded an unexpected TP model")
    model = loaded.backbone.model
    world_model = _build_world_model(
        args,
        rl_config,
        llm=model,
        device=device,
    )
    if not rl_config.predictor.train_wm:
        world_model.wm_predictor.requires_grad_(False).eval()

    checkpointed_modules = _prepare_planner_qwen_training(
        model,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        eval_modules=(find_visual_module(model),),
    )
    if bool(args.gradient_checkpointing) and checkpointed_modules < 1:
        raise RuntimeError("planner worker did not activate gradient checkpointing")

    agent = Agent(backbone=loaded.backbone, wm=world_model)
    input_builder = build_input_builder(
        loaded,
        max_length=999_999,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=True,
    )
    algorithm = RLAlgorithm(
        history_size=rl_config.predictor.history_size,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=rl_config.value_head.rank_margin,
        value_rank_weight=rl_config.value_head.lambda_rank,
        value_ppo_clip_range=rl_config.value_head.ppo_clip_range,
        planner_policy_enabled=True,
        planner_policy_clip_ratio=rl_config.planner_policy.clip_ratio,
        planner_policy_entropy_weight=rl_config.planner_policy.entropy_coeff,
        planner_policy_temperature=rl_config.planner_policy.temperature,
        ppo_clip_ratio=rl_config.actor.clip_ratio,
        entropy_weight=rl_config.actor.entropy_coeff,
        credit_assignment=rl_config.actor.credit_assignment,
        token_gamma=rl_config.token_credit.gamma,
        token_gae_lambda=rl_config.token_credit.gae_lambda,
        token_value_loss_weight=rl_config.token_credit.value_loss_weight,
        reference_kl_loss_weight=rl_config.actor.reference_kl_loss_weight,
        train_world_model=rl_config.predictor.train_wm,
        world_model_weight=rl_config.predictor.lambda_wm,
        dino_grid_weight=rl_config.predictor.lambda_dino,
    )
    objective = PlannerObjectiveModule(
        agent=agent,
        input_builder=input_builder,
        algorithm=algorithm,
        max_state_tokens=rl_config.actor.max_state_tokens,
    )

    def optimizer_factory(_root: torch.nn.Module) -> torch.optim.Optimizer:
        return _build_optimizer(
            objective.agent.backbone.model,
            objective.agent.wm,
            None,
            rl_config,
        )

    return PlannerWorkerModelComponents(
        objective_module=objective,
        optimizer_factory=optimizer_factory,
        wrap_policy=_planner_fsdp_wrap_policy(model, world_model),
        max_grad_norm=1.0,
    )


__all__ = ["build_planner_worker_components"]
