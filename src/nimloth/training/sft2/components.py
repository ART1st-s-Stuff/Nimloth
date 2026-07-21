"""构造 SFT2 的 Agent、分布式包装和 optimizer。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.agent import Agent, AgentBatchBuilder, AgentTarget
from nimloth.backbone import (
    BackboneEMA,
    build_sft2_batch_builder,
    build_vision_ema,
    load_sft2_backbone,
)
from nimloth.training.sft2.checkpoint import load_aux_checkpoint
from nimloth.wm import (
    LeWMConfig,
    LatentWMPredictor,
    SIGReg,
    StateProjector,
    ValueHead,
    WorldModel,
)


SFT2_WM_HISTORY_SIZE = 1


def require_sft2_wm_history(wm_predictor: LatentWMPredictor, source: Path) -> None:
    if wm_predictor.config.history_size != SFT2_WM_HISTORY_SIZE:
        raise ValueError(
            "SFT2 one-step WM requires a checkpoint with history_size=1; "
            f"got history_size={wm_predictor.config.history_size} from {source}"
        )


@dataclass(frozen=True)
class SFT2Components:
    """训练循环需要的完整 Agent 与优化状态。"""

    agent: Agent
    batch_builder: AgentBatchBuilder
    target: AgentTarget
    processor: Any
    vision_ema: BackboneEMA | None
    sigreg: SIGReg
    optimizer: torch.optim.Optimizer
    base_model_path: Path
    aux_device: torch.device
    qwen_pair_parallel: bool
    ddp_static_graph: bool


def build_sft2_components(
    args,
    resume_ckpt_dir: Path | None,
    *,
    device: torch.device,
    world_size: int,
    train_wm_predictor: bool,
    vision_ema_enabled: bool,
) -> SFT2Components:
    """由通用 backbone factory 加载 LLM，再装配完整 Agent。"""

    loaded = load_sft2_backbone(args, resume_ckpt_dir, device=device)
    model = loaded.backbone.model
    pair_parallel = loaded.pair_parallel
    base_model_path = Path(loaded.base_model_path)
    query_adapter = loaded.query_adapter

    wm_config = LeWMConfig(emb_dim=args.emb_dim, history_size=SFT2_WM_HISTORY_SIZE)
    if args.wm_predictor_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(
            args.wm_predictor_checkpoint,
            map_location=device,
        ).to(device)
        require_sft2_wm_history(wm_predictor, args.wm_predictor_checkpoint)
    else:
        wm_predictor = LatentWMPredictor.create(wm_config).to(device)
    if not train_wm_predictor:
        for parameter in wm_predictor.parameters():
            parameter.requires_grad = False

    aux_device = device
    if pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get("model.language_model.norm")
        if mapped is not None:
            aux_device = torch.device(f"cuda:{mapped}")
        wm_predictor = wm_predictor.to(aux_device)

    model_dtype = next(model.parameters()).dtype
    state_proj = StateProjector(
        model.config.hidden_size,
        wm_predictor.emb_dim,
        latent_token_count=args.latent_token_count,
    ).to(device=aux_device, dtype=model_dtype)
    value_head = ValueHead(wm_predictor.emb_dim).to(
        device=aux_device,
        dtype=model_dtype,
    )
    sigreg = SIGReg(
        knots=args.sigreg_knots,
        num_proj=args.sigreg_num_proj,
    ).to(device=aux_device)
    wm = WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )

    resume_state_path = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    if args.resume and resume_state_path is not None and resume_state_path.exists():
        load_aux_checkpoint(
            resume_ckpt_dir,
            wm,
            device,
            latent_query_mode=args.latent_query_mode,
            query_tune=args.query_tune,
        )

    static_graph = world_size > 1
    if world_size > 1:
        if pair_parallel:
            model = DDP(
                model,
                device_ids=None,
                output_device=None,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        else:
            device_index = int(str(device).split(":")[-1])
            model = DDP(
                model,
                device_ids=[device_index],
                output_device=device_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        aux_index = int(str(aux_device).split(":")[-1])
        state_proj = DDP(
            state_proj,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        value_head = DDP(
            value_head,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        if train_wm_predictor:
            wm_predictor = DDP(
                wm_predictor,
                device_ids=[aux_index],
                output_device=aux_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )

    vision_ema = build_vision_ema(
        enabled=vision_ema_enabled,
        decay=args.vision_ema_decay,
        llm=model,
        resume_path=(resume_ckpt_dir / "vision_ema.pt") if resume_ckpt_dir else None,
        device=device,
    )

    query_parameter = query_adapter.delta if query_adapter is not None else None
    qwen_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter is not query_parameter
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": qwen_parameters, "lr": args.lr_qwen_start, "name": "qwen"},
        {"params": state_proj.parameters(), "lr": args.state_proj_lr, "name": "state_proj"},
        {"params": value_head.parameters(), "lr": args.value_head_lr, "name": "value_head"},
    ]
    if query_parameter is not None:
        parameter_groups.append(
            {
                "params": [query_parameter],
                "lr": args.query_lr,
                "weight_decay": 0.0,
                "name": "query_adapter",
            }
        )
    if train_wm_predictor:
        predictor_parameters = (
            wm_predictor.module.parameters()
            if hasattr(wm_predictor, "module")
            else wm_predictor.parameters()
        )
        parameter_groups.append(
            {
                "params": list(predictor_parameters),
                "lr": args.wm_predictor_lr,
                "name": "wm_predictor",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    backbone = loaded.backbone.with_model(model)
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
        ),
    )
    batch_builder = build_sft2_batch_builder(
        loaded,
        device=aux_device,
        max_length=args.max_length,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
    )
    target = AgentTarget(
        agent,
        backbone_context=(
            (lambda: vision_ema.use_ema_weights(backbone.model))
            if vision_ema is not None
            else None
        ),
    )
    return SFT2Components(
        agent=agent,
        batch_builder=batch_builder,
        target=target,
        processor=loaded.processor,
        vision_ema=vision_ema,
        sigreg=sigreg,
        optimizer=optimizer,
        base_model_path=base_model_path,
        aux_device=aux_device,
        qwen_pair_parallel=pair_parallel,
        ddp_static_graph=static_graph,
    )


__all__ = ["SFT2Components", "build_sft2_components", "require_sft2_wm_history"]
