"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.agent import Agent
from nimloth.backbone import (
    DINOV2_LARGE_IDENTITY,
    CachedDINOGridTargets,
    build_input_builder,
    build_vision_ema,
    inspect_standalone_dino_cache,
    load_backbone,
    model_output_device,
    resolve_tune_modes,
    resolve_vision_ema,
    uses_lora,
)
from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.latent import (
    query_labels_are_masked,
    resolve_latent_query_mode,
)
from nimloth.training.sft.stage3.algorithm import (
    SFT2_VALUE_OBJECTIVE,
    SFT2Algorithm,
    require_sft2_wm_history,
)
from nimloth.training.sft.stage3.batch import Stage3BatchAssembler
from nimloth.training.sft.stage3.checkpoint import (
    SFT2CheckpointManager,
    SFT2CheckpointRuntime,
    load_world_model_checkpoint,
    resolve_resume_checkpoint_dir,
)
from nimloth.training.sft.stage3.cli import parse_sft2_args
from nimloth.training.sft.stage3.data.factory import build_data_bundle
from nimloth.training.sft.stage3.dino_grid import DINOGridBatchAssembler
from nimloth.training.sft.stage3.loop import (
    SFT2TrainingLoop,
    load_sft2_loop_state,
)
from nimloth.training.sft.stage3.reporting import STAGE3_LOG_COLUMNS, SFT2Reporter
from nimloth.training.sft.stage3.runtime import (
    SFT2ModelRuntime,
    SFT2OptimizationRuntime,
)
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.optim import OptimizationRuntime
from nimloth.util.wandb import init_wandb_run
from nimloth.wm import (
    LatentWMPredictor,
    LeWMConfig,
    SequenceSIGReg,
    StateProjector,
    ValueHead,
    WorldModel,
)
from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    ResidualTemporalSpatialGridPredictor,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
    load_sft1_slot_projector,
)
from nimloth.wm.layout import GridStateLayout
from nimloth.wm.outcome import ActionOutcomeHead


def _rl_eval_checkpoint_root(args: Any) -> Path | None:
    checkpoint = getattr(args, "rl_eval_checkpoint", None)
    return Path(checkpoint).resolve() if checkpoint is not None else None


def _frozen_wm_representation_identity(
    args: Any,
    resume_ckpt_dir: Path | None,
) -> dict[str, Any]:
    """Bind an offline state cache to the checkpoint that produced its states."""

    from nimloth.eval.stage3_outcome import file_sha256

    root = (resume_ckpt_dir or Path(args.model)).resolve()
    if resume_ckpt_dir is not None:
        relative_paths = (
            "training_state.pt",
            "state_proj.pt",
            "wm_predictor/config.json",
            "wm_predictor/predictor.pt",
        )
        kind = "stage3_resume"
    else:
        relative_paths = (
            "config.json",
            "training_state.pt",
            "grid_state_config.json",
            "slot_projector.pt",
        )
        kind = "stage2_initialization"
    missing = [name for name in relative_paths if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "frozen-WM representation checkpoint is incomplete: "
            f"root={root}, missing={missing}"
        )
    return {
        "representation_checkpoint": str(root),
        "representation_checkpoint_kind": kind,
        "representation_checkpoint_files": {
            name: file_sha256(root / name) for name in relative_paths
        },
    }


def _validate_rl_eval_checkpoint_contract(args: Any) -> tuple[Path, dict[str, Any]]:
    """Validate the self-contained RL artifact before allocating model weights."""

    root = _rl_eval_checkpoint_root(args)
    if root is None:
        raise ValueError("RL eval checkpoint root is required")
    if Path(args.model).resolve() != root:
        raise ValueError("RL eval Qwen and auxiliary components must share one checkpoint root")
    required = (
        root / "config.json",
        root / "rl_state.pt",
        root / "state_proj.pt",
        root / "wm_predictor" / "config.json",
        root / "wm_predictor" / "predictor.pt",
        root / "value_head" / "value_head.pt",
        root / "outcome_head.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete RL eval checkpoint; missing: {missing}")
    if (root / "planner_policy_head").exists():
        raise ValueError("Stage3 RL evaluation does not support PlannerPolicyHead checkpoints")

    state = torch.load(root / "rl_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("RL eval rl_state.pt must contain a dictionary")
    outcome_config = state.get("outcome_config")
    if not isinstance(outcome_config, dict) or not outcome_config.get("enabled", False):
        raise ValueError("RL eval checkpoint must declare an enabled OutcomeHead")
    if not getattr(args, "outcome_head", False):
        raise ValueError("RL eval checkpoint with OutcomeHead requires --outcome-head")
    return root, state


def _load_rl_eval_grid_world_model(
    args: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    pair_parallel: bool,
) -> tuple[GridWorldModel, torch.device]:
    """Build and strictly restore Stage3-compatible components from one RL root."""

    root, _state = _validate_rl_eval_checkpoint_contract(args)
    predictor_metadata = json.loads(
        (root / "wm_predictor" / "config.json").read_text(encoding="utf-8")
    )
    is_residual = predictor_metadata.get("schema") in {
        "nimloth_residual_temporal_spatial_grid_v1",
        "nimloth_residual_temporal_spatial_grid_v2",
    }
    predictor_type = (
        ResidualTemporalSpatialGridPredictor
        if is_residual
        else TemporalSpatialGridPredictor
    )
    predictor = predictor_type.load_checkpoint(
        root / "wm_predictor", map_location="cpu"
    )
    expected_kind = "residual" if is_residual else "direct"
    expected = {
        "grid_predictor_kind": (
            getattr(args, "grid_predictor_kind", "direct"),
            expected_kind,
        ),
        "latent_token_count": (
            int(args.latent_token_count),
            int(predictor.config.grid_tokens),
        ),
        "grid_size": (
            int(args.grid_size) ** 2 + int(getattr(args, "grid_global_tokens", 0)),
            int(predictor.config.grid_tokens),
        ),
        "emb_dim": (int(args.emb_dim), int(predictor.config.emb_dim)),
        "history_size": (int(args.history_size), int(predictor.config.history_size)),
        "grid_wm_depth": (int(args.grid_wm_depth), int(predictor.config.depth)),
        "grid_wm_heads": (int(args.grid_wm_heads), int(predictor.config.heads)),
        "grid_wm_dim_head": (
            int(args.grid_wm_dim_head),
            int(predictor.config.dim_head),
        ),
        "grid_wm_mlp_dim": (
            int(args.grid_wm_mlp_dim),
            int(predictor.config.mlp_dim),
        ),
        "grid_wm_dropout": (
            float(args.grid_wm_dropout),
            float(predictor.config.dropout),
        ),
    }
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"RL eval checkpoint/config mismatch: {mismatches}")

    world_model_device = (
        model_output_device(model, default=device) if pair_parallel else device
    )

    projector_state = torch.load(
        root / "state_proj.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(projector_state, dict):
        raise TypeError("RL eval state_proj.pt must contain a state dictionary")
    projector_first = projector_state.get("net.0.weight")
    if projector_first is None or projector_first.ndim != 2:
        raise ValueError("RL eval state projector is missing net.0.weight")
    if not projector_first.is_floating_point():
        raise ValueError("RL eval state projector weights must be floating point")
    qwen_hidden_dim = int(model.config.hidden_size)
    if int(projector_first.shape[1]) != qwen_hidden_dim:
        raise ValueError(
            "RL eval state projector/Qwen hidden dimension mismatch: "
            f"projector={projector_first.shape[1]}, qwen={qwen_hidden_dim}"
        )
    state_proj = SharedSlotProjector(
        input_dim=qwen_hidden_dim,
        output_dim=predictor.config.emb_dim,
        hidden_dim=int(projector_first.shape[0]),
        grid_tokens=predictor.config.grid_tokens,
    ).to(dtype=projector_first.dtype)
    state_proj.load_state_dict(projector_state, strict=True)
    value_head = ValueHead.load_checkpoint(
        root / "value_head",
        emb_dim=predictor.config.emb_dim,
        map_location="cpu",
    )
    outcome_payload = torch.load(
        root / "outcome_head.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(outcome_payload, dict):
        raise TypeError("RL eval outcome_head.pt must contain a dictionary")
    if outcome_payload.get("schema") != ActionOutcomeHead.schema:
        raise ValueError("unsupported RL eval OutcomeHead checkpoint schema")
    if int(outcome_payload.get("emb_dim", -1)) != predictor.config.emb_dim:
        raise ValueError("RL eval OutcomeHead dimension mismatch")
    outcome_state = outcome_payload.get("state_dict")
    if not isinstance(outcome_state, dict):
        raise TypeError("RL eval OutcomeHead checkpoint is missing state_dict")
    outcome_head = ActionOutcomeHead(predictor.config.emb_dim)
    outcome_head.load_state_dict(outcome_state, strict=True)
    world_model = GridWorldModel(
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
        outcome_head=outcome_head,
    ).to(world_model_device)
    return world_model, world_model_device


def _validate_dino_grid_contract(args: Any) -> dict[str, Any]:
    """Validate the configured grid against the Stage 2 state interface."""

    required = {
        "emb_dim": (args.emb_dim, DINOV2_LARGE_IDENTITY.hidden_size),
        "latent_query_mode": (args.latent_query_mode, "inject"),
    }
    mismatches = {
        name: values
        for name, values in required.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"authoritative DINO-grid SFT2 invariants mismatch: {mismatches}"
        )
    if not math.isfinite(args.lambda_sigreg) or args.lambda_sigreg < 0:
        raise ValueError("lambda_sigreg must be finite and nonnegative")
    expected_tokens = int(args.grid_size) ** 2 + int(args.grid_global_tokens)
    if int(args.latent_token_count) != expected_tokens:
        raise ValueError(
            "DINO-grid token/grid mismatch: "
            f"latent_token_count={args.latent_token_count}, "
            f"grid_size={args.grid_size}, expected={expected_tokens}"
        )
    if args.dino_grid_cache is None:
        raise ValueError("DINO-grid SFT2 requires --dino-grid-cache")

    if _rl_eval_checkpoint_root(args) is not None:
        root, state = _validate_rl_eval_checkpoint_contract(args)
        predictor_metadata = json.loads(
            (root / "wm_predictor" / "config.json").read_text(encoding="utf-8")
        )
        return {"rl_state": state, "wm_predictor": predictor_metadata}

    config_path = Path(args.model) / "grid_state_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"DINO-grid Stage 3 requires Stage 2 state metadata: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    objective = config.get("objective")
    checkpoint_grid_size = (
        objective.get("grid_size") if isinstance(objective, dict) else None
    )
    expected = {
        "grid_tokens": int(args.latent_token_count),
        "state_dim": int(args.emb_dim),
        "shared_slot_projector": True,
        "ordering": (
            "row_major_spatial_then_global" if args.grid_global_tokens else "row_major"
        ),
    }
    checkpoint_mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if checkpoint_grid_size != int(args.grid_size):
        checkpoint_mismatches["objective.grid_size"] = (
            checkpoint_grid_size,
            int(args.grid_size),
        )
    if int(config.get("global_tokens", 0)) != int(args.grid_global_tokens):
        checkpoint_mismatches["global_tokens"] = (
            config.get("global_tokens"),
            int(args.grid_global_tokens),
        )
    if args.grid_global_tokens and (
        not config.get("evaluation_only") or config.get("formal_stage2") is not False
    ):
        checkpoint_mismatches["evaluation_only"] = (
            config.get("evaluation_only"),
            True,
        )
    if config.get("dino_identity") != asdict(DINOV2_LARGE_IDENTITY):
        checkpoint_mismatches["dino_identity"] = (
            config.get("dino_identity"),
            asdict(DINOV2_LARGE_IDENTITY),
        )
    if checkpoint_mismatches:
        raise ValueError(
            "Stage 2 DINO-grid state interface mismatch: "
            f"{checkpoint_mismatches}"
        )
    return config


def _validate_stage2_stage3_dino_cache_contract(
    args: Any,
    stage2_grid_config: dict[str, Any],
) -> dict[str, Any]:
    """Anchor Stage2's cache, then compare feature semantics across corpora."""

    stage2_fingerprint = stage2_grid_config.get("dino_cache_fingerprint")
    if not isinstance(stage2_fingerprint, str) or not stage2_fingerprint:
        raise ValueError("Stage2 checkpoint is missing its DINO cache fingerprint")
    stage3 = inspect_standalone_dino_cache(
        args.dino_grid_cache,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=args.grid_size,
    )
    aligned_root = getattr(args, "stage2_aligned_dino_cache", None)
    if aligned_root is None:
        if stage3.cache_fingerprint != stage2_fingerprint:
            raise ValueError(
                "Stage3 uses a different DINO corpus from Stage2; "
                "--stage2-aligned-dino-cache is required to prove feature-space compatibility"
            )
        aligned = stage3
        aligned_root = args.dino_grid_cache
    else:
        aligned = inspect_standalone_dino_cache(
            aligned_root,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=args.grid_size,
        )
        if aligned.cache_fingerprint != stage2_fingerprint:
            raise ValueError(
                "Stage2 aligned DINO cache fingerprint does not match the checkpoint: "
                f"cache={aligned.cache_fingerprint}, checkpoint={stage2_fingerprint}"
            )
    if aligned.feature_space != stage3.feature_space:
        keys = sorted(set(aligned.feature_space) | set(stage3.feature_space))
        mismatches = {
            key: (aligned.feature_space.get(key), stage3.feature_space.get(key))
            for key in keys
            if aligned.feature_space.get(key) != stage3.feature_space.get(key)
        }
        raise ValueError(
            "Stage2/Stage3 DINO feature-space identity mismatch: "
            f"{mismatches}"
        )
    return {
        "stage2_aligned_cache_root": str(Path(aligned_root).resolve()),
        "stage2_aligned_cache_fingerprint": aligned.cache_fingerprint,
        "stage3_supervision_cache_root": str(Path(args.dino_grid_cache).resolve()),
        "stage3_supervision_cache_fingerprint": stage3.cache_fingerprint,
        "feature_space_fingerprint": stage3.feature_space_fingerprint,
        "feature_space": stage3.feature_space,
    }


def _build_world_model(
    args: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    pair_parallel: bool,
    resume_ckpt_dir: Path | None,
    train_wm_predictor: bool,
) -> tuple[WorldModel, torch.device]:
    """按 objective 构造并恢复 world-model 子模块。"""

    if _rl_eval_checkpoint_root(args) is not None:
        if resume_ckpt_dir is not None or getattr(args, "resume", False):
            raise ValueError("RL eval checkpoint cannot be combined with Stage3 resume")
        return _load_rl_eval_grid_world_model(
            args,
            model=model,
            device=device,
            pair_parallel=pair_parallel,
        )

    world_model_device = device
    if pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get(
            "model.language_model.norm"
        )
        if mapped is not None:
            world_model_device = torch.device(f"cuda:{mapped}")

    model_dtype = next(model.parameters()).dtype
    if args.objective == "dino_grid":
        # SFT1 已用 DINO grid 监督这个 projector；SFT2 从该权重继续训练，
        # 并让 WM 直接在同一个 DINO-aligned state 空间中预测。
        grid_dtype = torch.float32
        state_proj = load_sft1_slot_projector(
            args.model,
            qwen_hidden_dim=int(model.config.hidden_size),
            state_dim=args.emb_dim,
            grid_tokens=args.latent_token_count,
            map_location=world_model_device,
            dtype=grid_dtype,
            state_layout=(
                GridStateLayout(
                    spatial_grid_size=args.grid_size,
                    global_tokens=1,
                    global_role="dino_cls",
                )
                if args.grid_global_tokens
                else None
            ),
        ).to(world_model_device)
        predictor_kind = getattr(args, "grid_predictor_kind", "direct")
        predictor_types = {
            "direct": TemporalSpatialGridPredictor,
            "residual": ResidualTemporalSpatialGridPredictor,
        }
        if predictor_kind not in predictor_types:
            raise ValueError(f"unsupported grid predictor kind: {predictor_kind}")
        wm_predictor = predictor_types[predictor_kind](
            GridPredictorConfig(
                grid_tokens=args.latent_token_count,
                spatial_grid_size=(
                    args.grid_size
                    if args.grid_position_encoding == "fixed_2d_sincos_v1"
                    else None
                ),
                global_tokens=args.grid_global_tokens,
                position_encoding=args.grid_position_encoding,
                emb_dim=args.emb_dim,
                history_size=args.history_size,
                depth=args.grid_wm_depth,
                heads=args.grid_wm_heads,
                dim_head=args.grid_wm_dim_head,
                mlp_dim=args.grid_wm_mlp_dim,
                dropout=args.grid_wm_dropout,
            )
        ).to(device=world_model_device, dtype=grid_dtype)
        outcome_head = None
        if getattr(args, "outcome_head", False):
            outcome_head = ActionOutcomeHead(args.emb_dim).to(world_model_device)
            outcome_head.requires_grad_(getattr(args, "lambda_outcome", 0.0) > 0)
        world_model = GridWorldModel(
            outcome_head=outcome_head,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=ValueHead(args.emb_dim).to(
                device=world_model_device,
                dtype=grid_dtype,
            ),
        )
    else:
        if args.wm_predictor_checkpoint is not None:
            wm_predictor = LatentWMPredictor.load_checkpoint(
                args.wm_predictor_checkpoint,
                map_location=world_model_device,
            ).to(world_model_device)
            require_sft2_wm_history(
                wm_predictor,
                history_size=args.history_size,
                source=args.wm_predictor_checkpoint,
            )
        else:
            wm_predictor = LatentWMPredictor.create(
                LeWMConfig(
                    emb_dim=args.emb_dim,
                    history_size=args.history_size,
                )
            ).to(world_model_device)
        world_model = WorldModel(
            state_proj=StateProjector(
                model.config.hidden_size,
                wm_predictor.emb_dim,
                latent_token_count=args.latent_token_count,
            ).to(device=world_model_device, dtype=model_dtype),
            wm_predictor=wm_predictor,
            value_head=ValueHead(wm_predictor.emb_dim).to(
                device=world_model_device,
                dtype=model_dtype,
            ),
        )

    if not train_wm_predictor:
        world_model.wm_predictor.requires_grad_(False)
    resume_state = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    if args.resume and resume_state is not None and resume_state.exists():
        load_world_model_checkpoint(
            resume_ckpt_dir,
            world_model,
            world_model_device,
            latent_query_mode=args.latent_query_mode,
            query_tune=args.query_tune,
        )
    return world_model, world_model_device


def _wrap_sft2_agent(
    loaded,
    world_model: WorldModel,
    *,
    device: torch.device,
    world_model_device: torch.device,
    world_size: int,
    train_wm_predictor: bool,
    distributed_strategy: str = "ddp",
    fsdp_wrap_granularity: str = "linear",
) -> tuple[Agent, bool]:
    """按现有多卡语义包装模型，再组成唯一的神经网络 Agent。"""

    model = loaded.backbone.model
    state_proj = world_model.state_proj
    wm_predictor = world_model.wm_predictor
    value_head = world_model.value_head
    outcome_head = world_model.outcome_head
    static_graph = world_size > 1
    if distributed_strategy == "fsdp" and (loaded.pair_parallel or world_size < 2):
        raise ValueError("FSDP requires multi-rank single-device Qwen ranks")
    if world_size > 1:
        if distributed_strategy == "fsdp":
            from nimloth.training.sft.stage3.fsdp import wrap_qwen_fsdp
            model = wrap_qwen_fsdp(model, device, granularity=fsdp_wrap_granularity)
        elif loaded.pair_parallel:
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
        world_model_device_index = int(str(world_model_device).split(":")[-1])
        state_proj = DDP(
            state_proj,
            device_ids=[world_model_device_index],
            output_device=world_model_device_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        value_head = DDP(
            value_head,
            device_ids=[world_model_device_index],
            output_device=world_model_device_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        if outcome_head is not None and any(p.requires_grad for p in outcome_head.parameters()):
            outcome_head = DDP(outcome_head, device_ids=[world_model_device_index],
                               output_device=world_model_device_index, static_graph=static_graph)
        if train_wm_predictor:
            wm_predictor = DDP(
                wm_predictor,
                device_ids=[world_model_device_index],
                output_device=world_model_device_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
    if isinstance(world_model, GridWorldModel):
        wrapped_world_model: WorldModel = GridWorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
            outcome_head=outcome_head,
        )
    else:
        wrapped_world_model = WorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
            outcome_head=outcome_head,
        )

    return (
        Agent(
            backbone=loaded.backbone.with_model(model),
            wm=wrapped_world_model,
        ),
        static_graph,
    )


def _build_optimizer(
    args: Any,
    *,
    agent: Agent,
    query_adapter: Any,
    train_wm_predictor: bool,
) -> torch.optim.Optimizer:
    """按模块名称建立可审计的 SFT2 参数组。"""

    query_parameter = query_adapter.delta if query_adapter is not None else None
    selected_groups = None
    selected_ids: set[int] = set()
    if getattr(args, "query_tune", "freeze") == "selected_rows":
        from nimloth.backbone.selected_token_rows import selected_row_parameters
        if query_adapter is not None:
            raise ValueError("selected rows cannot coexist with Query delta adapter")
        selected_groups = selected_row_parameters(agent.backbone.model)
        selected_ids = {id(p) for group in selected_groups.values() for p in group}
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in agent.backbone.model.parameters()
                if parameter.requires_grad and parameter is not query_parameter
                and id(parameter) not in selected_ids
            ],
            "lr": args.lr_qwen_start,
            "name": "qwen",
        },
        {
            "params": [
                parameter
                for parameter in agent.wm.state_proj.parameters()
                if parameter.requires_grad
            ],
            "lr": args.state_proj_lr,
            "name": "state_proj",
        },
        {
            "params": agent.wm.value_head.parameters(),
            "lr": args.value_head_lr,
            "name": "value_head",
        },
    ]
    if selected_groups is not None:
        for name, rate in (("query", args.query_lr), ("protocol", args.protocol_lr)):
            parameter_groups.append({"params": selected_groups[name], "lr": rate,
                                     "weight_decay": 0.0, "name": f"selected_{name}_rows"})
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
        predictor = agent.wm.wm_predictor
        predictor_parameters = (
            predictor.module.parameters()
            if hasattr(predictor, "module")
            else predictor.parameters()
        )
        parameter_groups.append(
            {
                "params": list(predictor_parameters),
                "lr": args.wm_predictor_lr,
                "name": "wm_predictor",
            }
        )
    if agent.wm.outcome_head is not None:
        params = [p for p in agent.wm.outcome_head.parameters() if p.requires_grad]
        if params:
            parameter_groups.append({"params": params, "lr": args.outcome_head_lr, "name": "outcome_head"})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def _train_sft2_impl(args=None) -> int:
    if args is None:
        args = parse_sft2_args()
    args.latent_token_count = int(getattr(args, "latent_token_count", 1))
    args.latent_query_mode = resolve_latent_query_mode(
        getattr(args, "latent_query_mode", None),
        default="inject",
    )
    args.mask_latent_query_labels = query_labels_are_masked(args.latent_query_mode)
    args.query_tune = str(getattr(args, "query_tune", "freeze"))
    args.query_lr = float(getattr(args, "query_lr", 5e-5))
    args.objective = str(getattr(args, "objective", "latent"))
    if args.objective not in {"latent", "dino_grid"}:
        raise ValueError(f"unsupported SFT2 objective: {args.objective!r}")
    if args.query_tune not in {"freeze", "adapter", "selected_rows"}:
        raise ValueError(f"query_tune must be freeze, adapter or selected_rows, got {args.query_tune!r}")
    if args.latent_token_count < 1:
        raise ValueError(f"--latent-token-count must be >= 1, got {args.latent_token_count}")
    args.history_size = int(getattr(args, "history_size", 4))
    if args.history_size < 1:
        raise ValueError(f"--history-size must be >= 1, got {args.history_size}")
    args.prediction_horizon = int(getattr(args, "prediction_horizon", 1))
    if args.prediction_horizon < 1:
        raise ValueError(
            f"--prediction-horizon must be >= 1, got {args.prediction_horizon}"
        )
    if args.prediction_horizon > 1 and args.history_size != 1:
        raise ValueError(
            "multi-step SFT2 training requires --history-size 1, "
            f"got H={args.history_size}, T={args.prediction_horizon}"
        )
    stage2_grid_config = None
    args.dino_cache_audit = None
    if args.objective == "dino_grid":
        stage2_grid_config = _validate_dino_grid_contract(args)
        if args.grid_global_tokens and _rl_eval_checkpoint_root(args) is None:
            args.dino_cache_audit = _validate_stage2_stage3_dino_cache_contract(
                args,
                stage2_grid_config,
            )

    llm_tune, vision_tune = resolve_tune_modes(args)
    if args.query_tune == "adapter" and uses_lora(args):
        raise ValueError("query_tune=adapter is not supported with LoRA tuning")
    if args.query_tune == "selected_rows":
        if llm_tune != "full" or uses_lora(args):
            raise ValueError("selected_rows requires full dense language tuning")
        if args.query_lr <= 0 or args.protocol_lr <= 0:
            raise ValueError("selected-row learning rates must be positive")
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)
    train_wm_predictor = args.train_wm_predictor and not args.freeze_wm_predictor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, _local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("WANDB_RUN_PREFIX", "")
    wandb_run = init_wandb_run(
        rank=rank,
        output_dir=args.output_dir,
        enabled=not args.no_wandb,
        default_project="nimloth",
        run_name=args.wandb_run_name or f"{prefix}sft2-{args.objective}-value",
        config=vars(args),
        metric_definitions=(
            ("global_step", None),
            ("train/*", "global_step"),
            ("epoch", None),
            ("val/*", "epoch"),
        ),
    )

    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(args.output_dir, args.resume_from)
    resume_state_path = (
        resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir is not None else None
    )
    rl_eval_checkpoint_dir = _rl_eval_checkpoint_root(args)
    if is_main():
        print(
            json.dumps(
                {
                    "llm_tune": llm_tune,
                    "vision_tune": vision_tune,
                    "vision_ema": vision_ema_enabled,
                    "vision_ema_decay": args.vision_ema_decay,
                    "train_wm_predictor": train_wm_predictor,
                    "objective": args.objective,
                    "resume": args.resume,
                    "resume_from": str(resume_ckpt_dir) if resume_ckpt_dir is not None else None,
                    "rl_eval_checkpoint": (
                        str(rl_eval_checkpoint_dir)
                        if rl_eval_checkpoint_dir is not None
                        else None
                    ),
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                    "batch_mode": args.batch_mode,
                    "history_size": args.history_size,
                    "prediction_horizon": args.prediction_horizon,
                    "training_unit": "complete_trajectory_v1",
                    "batch_unit": "trajectory",
                    "latent_token_count": args.latent_token_count,
                    "latent_query_mode": args.latent_query_mode,
                    "query_tune": args.query_tune,
                    "mask_latent_query_labels": args.mask_latent_query_labels,
                    "preprocess_cache_image_dtype": args.preprocess_cache_image_dtype,
                    "preprocess_cache_reuse_image_root": (
                        str(args.preprocess_cache_reuse_image_root)
                        if args.preprocess_cache_reuse_image_root is not None
                        else None
                    ),
                    "preprocess_cache_reuse_processor_source": (
                        str(args.preprocess_cache_reuse_processor_source)
                        if args.preprocess_cache_reuse_processor_source is not None
                        else None
                    ),
                    "require_prebuilt_cache": args.require_prebuilt_cache,
                    "dino_grid_cache": (
                        str(args.dino_grid_cache)
                        if args.dino_grid_cache is not None
                        else None
                    ),
                    "dino_cache_audit": args.dino_cache_audit,
                }
            )
        )

    loaded = load_backbone(
        args,
        device=device,
        latent_token_count=args.latent_token_count,
        resume_dir=resume_ckpt_dir,
        resume_state_path=resume_state_path,
    )
    world_model, world_model_device = _build_world_model(
        args,
        model=loaded.backbone.model,
        device=device,
        pair_parallel=loaded.pair_parallel,
        resume_ckpt_dir=resume_ckpt_dir,
        train_wm_predictor=train_wm_predictor,
    )
    agent, ddp_static_graph = _wrap_sft2_agent(
        loaded,
        world_model,
        device=device,
        world_model_device=world_model_device,
        world_size=world,
        train_wm_predictor=train_wm_predictor,
        distributed_strategy=getattr(args, "distributed_strategy", "ddp"),
        fsdp_wrap_granularity=getattr(args, "fsdp_wrap_granularity", "linear"),
    )
    if getattr(args, "distributed_strategy", "ddp") == "fsdp" and vision_ema_enabled:
        from nimloth.training.sft.stage3.vision_ema_fsdp import build_fsdp_vision_ema
        vision_ema = build_fsdp_vision_ema(
            decay=args.vision_ema_decay, model=agent.backbone.model,
            resume_path=(
                (resume_ckpt_dir or rl_eval_checkpoint_dir) / "vision_ema.pt"
                if (resume_ckpt_dir or rl_eval_checkpoint_dir) is not None
                else None
            ),
        )
    else:
        vision_ema = build_vision_ema(
            enabled=vision_ema_enabled,
            decay=args.vision_ema_decay,
            llm=agent.backbone.model,
            resume_path=(
                (resume_ckpt_dir or rl_eval_checkpoint_dir) / "vision_ema.pt"
                if (resume_ckpt_dir or rl_eval_checkpoint_dir) is not None
                else None
            ),
            device=device,
        )
    input_builder = build_input_builder(
        loaded,
        max_length=args.max_length,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
    )
    base_batch_builder = Stage3BatchAssembler(
        input_builder=input_builder,
        device=world_model_device,
        prediction_horizon=args.prediction_horizon,
    )
    if args.objective == "dino_grid":
        dino_targets = CachedDINOGridTargets.from_cache_root(
            args.dino_grid_cache,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=args.grid_size,
        )
        if dino_targets.include_cls != bool(args.grid_global_tokens):
            raise ValueError("Stage3 state layout and DINO cache global-token schema mismatch")
        if (
            args.dino_cache_audit is not None
            and args.dino_cache_audit["stage3_supervision_cache_fingerprint"]
            != dino_targets.cache_fingerprint
        ):
            raise ValueError(
                "Stage3 DINO cache changed after feature-space preflight"
            )
        args.dino_cache_fingerprint = dino_targets.cache_fingerprint
        batch_builder = DINOGridBatchAssembler(
            base_batch_builder,
            dino_targets,
        )
    else:
        args.dino_cache_fingerprint = None
        batch_builder = base_batch_builder
    model_runtime = SFT2ModelRuntime(
        agent=agent,
        backbone_ema=vision_ema,
    )
    optimizer = _build_optimizer(
        args,
        agent=agent,
        query_adapter=loaded.query_adapter,
        train_wm_predictor=train_wm_predictor,
    )
    use_ddp_no_sync = (
        world > 1
        and not loaded.pair_parallel
        and not ddp_static_graph
        and getattr(args, "distributed_strategy", "ddp") != "fsdp"
    )
    if (
        is_main()
        and world > 1
        and args.grad_accum > 1
        and not use_ddp_no_sync
    ):
        print(
            json.dumps(
                {
                    "ddp_gradient_accumulation": "sync_each_microbatch",
                    "reason": "torch_2_8_static_graph_no_sync_regression",
                }
            )
        )
    data = build_data_bundle(
        args,
        batch_builder,
        rank=rank,
        world_size=world,
    )
    train_loader = data.train_loader
    val_loader = data.val_loader
    train_batch_sampler = data.train_batch_sampler
    local_batch_histogram = Counter(train_batch_sampler.trajectories_per_batch)
    batch_histograms: list[dict[int, int] | None] = [None] * world
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(
            batch_histograms,
            dict(local_batch_histogram),
        )
    else:
        batch_histograms[0] = dict(local_batch_histogram)
    global_batch_histogram: Counter[int] = Counter()
    for histogram in batch_histograms:
        if histogram is not None:
            global_batch_histogram.update(histogram)
    owned_trajectories = sum(
        batch_size * count
        for batch_size, count in global_batch_histogram.items()
    )
    if owned_trajectories != train_batch_sampler.trajectory_count:
        raise RuntimeError(
            "complete trajectory sampler ownership mismatch: "
            f"expected={train_batch_sampler.trajectory_count}, actual={owned_trajectories}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "stage3_sample_ownership": "trajectory_windows_once_v1",
                    "train_trajectories": train_batch_sampler.trajectory_count,
                    "train_current_steps": train_batch_sampler.window_count,
                    "actual_trajectories_per_microbatch": dict(
                        sorted(global_batch_histogram.items())
                    ),
                }
            )
        )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = args.schedule_total_steps or steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))

    def after_optimizer_step() -> None:
        if vision_ema is not None:
            vision_ema.update(agent.backbone.model)

    from nimloth.training.sft.stage3.fsdp import clip_mixed_grad_norm
    optimization_runtime = SFT2OptimizationRuntime(
        optimization=OptimizationRuntime(
            optimizer=optimizer,
            synchronized_modules=agent.synchronized_modules,
            enable_no_sync=use_ddp_no_sync,
            after_step=after_optimizer_step,
            gradient_clipper=(
                (lambda limit: clip_mixed_grad_norm(agent.backbone.model, agent.wm.trainable_modules, limit))
                if getattr(args, "distributed_strategy", "ddp") == "fsdp" else None
            ),
        ),
        qwen_warmup_steps=qwen_warmup_steps,
        total_steps=total_steps,
        qwen_start_lr=args.lr_qwen_start,
        qwen_peak_lr=args.lr_qwen_peak,
    )
    checkpoint_invariants = {
        "schedule_total_steps": total_steps,
        "objective": args.objective,
        "seed": int(args.seed),
        "world_size": int(world),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "latent_query_mode": args.latent_query_mode,
        "query_tune": args.query_tune,
        "history_size": int(args.history_size),
        "prediction_horizon": int(args.prediction_horizon),
        "state_layout": (
            agent.wm.state_layout.metadata()
            if getattr(agent.wm, "state_layout", None) is not None
            else None
        ),
        "grid_position_encoding": str(args.grid_position_encoding),
        "evaluation_only": bool(args.grid_global_tokens),
        "formal_stage3": not bool(args.grid_global_tokens),
        "training_unit": "complete_trajectory_v1",
        "batch_unit": "trajectory",
        "sigreg_batch_scope": "global_unique_trajectory_transitions_v1",
        "lambda_sigreg": float(args.lambda_sigreg),
        "wm_value_backbone_grad": args.wm_value_backbone_grad,
        "sample_ownership_version": "trajectory_windows_once_v1",
        "value_objective": SFT2_VALUE_OBJECTIVE,
        "lm_supervision": "successful_trajectory_window_mean_v1",
        "loss_normalization": "global_optimizer_group_counts_v1",
        "train_micro_batches": len(train_loader),
        "rng_schedule_version": "trajectory_micro_rank_v1",
        "training_mode_contract": "online_train_teacher_eval_v1",
        "evaluation_state_contract": "online_policy_eval_target_visual_ema_v1",
    }
    # Preserve historical direct identities; residual resumes require explicit identity.
    if getattr(args, "grid_predictor_kind", "direct") != "direct":
        checkpoint_invariants["grid_predictor_kind"] = args.grid_predictor_kind
    if getattr(args, "activation_offload", False):
        checkpoint_invariants["activation_offload"] = True
    if getattr(args, "distributed_strategy", "ddp") == "fsdp":
        from nimloth.training.sft.stage3.fsdp import STRATEGY_ID
        checkpoint_invariants["distributed_strategy"] = STRATEGY_ID
    if getattr(args, "outcome_head", False):
        checkpoint_invariants.update({
            "outcome_schema": ActionOutcomeHead.schema,
            "lambda_outcome": args.lambda_outcome,
            "outcome_head_lr": args.outcome_head_lr,
        })
    if args.query_tune == "selected_rows":
        checkpoint_invariants.update({
            "query_lr": float(args.query_lr),
            "protocol_lr": float(args.protocol_lr),
            "lr_qwen_start": float(args.lr_qwen_start),
            "lr_qwen_peak": float(args.lr_qwen_peak),
            "backbone_master_dtype": "float32",
        })
    if args.objective == "dino_grid":
        checkpoint_invariants.update(
            {
                "grid_tokens": int(args.latent_token_count),
                "grid_ordering": (
                    "row_major_spatial_then_global"
                    if args.grid_global_tokens
                    else "row_major"
                ),
                "dino_grid_size": int(args.grid_size),
                "dino_identity": asdict(DINOV2_LARGE_IDENTITY),
                "dino_cache_fingerprint": args.dino_cache_fingerprint,
                "dino_cache_audit": args.dino_cache_audit,
                "dino_weight": float(args.lambda_dino),
                "grid_state_format": "trainable_sft1_projector_v2",
                "dino_supervision": "unique_observed_online_state_mse_v1",
                "dino_normalization": "global_optimizer_group_observed_states_v1",
            }
        )
    if args.grid_global_tokens:
        checkpoint_invariants["loss_weights"] = {
            "wm_start": float(args.lambda_wm_start),
            "wm_end": float(args.lambda_wm_end),
            "lm_ce": float(args.lambda_ce),
            "dino_spatial": float(args.lambda_dino),
            "dino_cls": float(args.lambda_dino),
            "value": float(args.lambda_value),
            "outcome": float(args.lambda_outcome),
            "sigreg": float(args.lambda_sigreg),
        }
    checkpoint_manager = SFT2CheckpointManager(
        output_dir=args.output_dir,
        agent=agent,
        processor=loaded.processor,
        vision_ema=vision_ema,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
        lora=uses_lora(args),
        base_model_path=Path(loaded.base_model_path),
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        latent_query_mode=args.latent_query_mode,
        query_tune=args.query_tune,
    )
    checkpoint_runtime = SFT2CheckpointRuntime(
        manager=checkpoint_manager,
        rank=rank,
        device=device,
        interval_steps=int(args.checkpoint_interval_steps or 0),
        interval_minutes=float(args.checkpoint_interval_minutes),
        keep_last=int(args.checkpoint_keep_last or 0),
        checkpoint_latest_only=bool(getattr(args, "checkpoint_latest_only", False)),
        deduplicate_epoch_checkpoints=bool(getattr(args, "deduplicate_epoch_checkpoints", False)),
    )

    log_writer = CSVRecordWriter(
        args.output_dir / "train_step_log.csv",
        STAGE3_LOG_COLUMNS,
    )
    if is_main():
        log_writer.ensure_header()
    reporter = SFT2Reporter(
        log_writer=log_writer,
        wandb_run=wandb_run,
        llm_tune=llm_tune,
        vision_tune=vision_tune,
    )

    algorithm_kwargs = {
        "history_size": args.history_size,
        "sigreg": (
            SequenceSIGReg(
                knots=args.sigreg_knots,
                num_proj=args.sigreg_num_proj,
            ).to(device=world_model_device)
            if args.lambda_sigreg > 0.0
            else None
        ),
        "sigreg_weight": args.lambda_sigreg,
        "wm_value_backbone_grad": args.wm_value_backbone_grad,
        "value_weight": args.lambda_value,
        "ce_weight": args.lambda_ce,
        "wm_weight_start": args.lambda_wm_start,
        "wm_weight_end": args.lambda_wm_end,
        "dino_grid_weight": (args.lambda_dino if args.objective == "dino_grid" else 0.0),
        "prediction_horizon": args.prediction_horizon,
    }
    algorithm_kwargs["outcome_weight"] = getattr(args, "lambda_outcome", 0.0)
    algorithm = SFT2Algorithm(**algorithm_kwargs)

    loop_state = load_sft2_loop_state(
        legacy_schedule_total_steps=args.schedule_total_steps or None,
        resume=args.resume,
        agent=agent,
        resume_state_path=resume_state_path,
        resume_checkpoint_dir=resume_ckpt_dir,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
    )
    outcome_export_identity = None
    if getattr(args, "outcome_eval_dir", None) is not None:
        import subprocess

        from nimloth.eval.stage3_outcome import file_sha256
        if args.max_val_batches > 0:
            raise ValueError("complete outcome exports require max_val_batches=-1")
        outcome_export_identity = {
            "train_sha256": file_sha256(args.train_jsonl), "eval_sha256": file_sha256(args.val_jsonl),
            "initial_checkpoint": str(args.model), "initial_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "initial_training_state_sha256": file_sha256(Path(args.model) / "training_state.pt"),
            "seed": args.seed, "run_output": str(args.output_dir),
            "evaluation_state_contract": "online_policy_eval_target_visual_ema_v1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
    frozen_wm_cache_identity = None
    if getattr(args, "frozen_wm_cache_dir", None) is not None:
        import subprocess

        from nimloth.eval.stage3_outcome import file_sha256
        from nimloth.rollout.fresh import policy_artifact_fingerprint

        frozen_split = args.frozen_wm_cache_split
        if frozen_split == "train":
            split_jsonl = args.train_jsonl
            split_sampler = data.train_batch_sampler
        elif frozen_split == "eval":
            split_jsonl = args.val_jsonl
            split_sampler = data.val_batch_sampler
        else:
            raise ValueError("frozen-WM export split must be train or eval")
        frozen_wm_cache_identity = {
            "split": frozen_split,
            "split_jsonl": str(split_jsonl),
            "split_sha256": file_sha256(split_jsonl),
            "trajectory_count": int(split_sampler.trajectory_count),
            "window_count": int(split_sampler.window_count),
            "stage2_checkpoint": str(args.model),
            "stage2_policy_fingerprint": policy_artifact_fingerprint(Path(args.model)),
            "stage2_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "stage2_commit_marker_sha256": file_sha256(Path(args.model) / "COMMITTED"),
            "stage2_grid_config_sha256": file_sha256(
                Path(args.model) / "grid_state_config.json"
            ),
            "stage2_projector_sha256": file_sha256(Path(args.model) / "slot_projector.pt"),
            "dino_cache_fingerprint": args.dino_cache_fingerprint,
            "prediction_horizon": args.prediction_horizon,
            "grid_tokens": args.latent_token_count,
            "state_dim": args.emb_dim,
            "action_dim": 8,
            "state_layout": (
                agent.wm.state_layout.metadata()
                if getattr(agent.wm, "state_layout", None) is not None
                else None
            ),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        frozen_wm_cache_identity.update(
            _frozen_wm_representation_identity(args, resume_ckpt_dir)
        )
    feature_export_step = int(loop_state.global_step)
    diagnostic_identity = {
        "run_output": str(Path(args.output_dir).resolve()),
        "initialization": str(Path(args.model).resolve()),
        "validation": str(Path(args.val_jsonl).resolve()),
        "invariants": checkpoint_invariants,
    }
    if rl_eval_checkpoint_dir is not None:
        from nimloth.eval.stage3_outcome import file_sha256

        rl_eval_state = torch.load(
            rl_eval_checkpoint_dir / "rl_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        feature_export_step = int(rl_eval_state["global_step"])
        diagnostic_identity["rl_eval"] = {
            "checkpoint": str(rl_eval_checkpoint_dir),
            "rl_state_sha256": file_sha256(rl_eval_checkpoint_dir / "rl_state.pt"),
            "global_step": feature_export_step,
        }
    training_loop = SFT2TrainingLoop(
        config=SFT2LoopConfig.from_namespace(args),
        outcome_eval_dir=getattr(args, "outcome_eval_dir", None),
        outcome_export_identity=outcome_export_identity,
        feature_export_dir=getattr(args, "feature_export_dir", None),
        feature_export_identity=diagnostic_identity,
        feature_export_step=feature_export_step,
        diagnostic_steps=tuple(getattr(args, "diagnostic_steps", ())),
        diagnostic_dir=getattr(args, "diagnostic_dir", None),
        diagnostic_identity=diagnostic_identity,
        frozen_wm_cache_dir=getattr(args, "frozen_wm_cache_dir", None),
        frozen_wm_cache_split=getattr(args, "frozen_wm_cache_split", None),
        frozen_wm_cache_identity=frozen_wm_cache_identity,
        rank=rank,
        train_loader=train_loader,
        val_loader=val_loader,
        train_batch_sampler=train_batch_sampler,
        algorithm=algorithm,
        model_runtime=model_runtime,
        optimization_runtime=optimization_runtime,
        batch_builder=batch_builder,
        checkpoint_runtime=checkpoint_runtime,
        reporter=reporter,
        state=loop_state,
        total_steps=total_steps,
    )
    if getattr(args, "eval_only", False):
        training_loop.evaluate_only()
    else:
        training_loop.run()
    if wandb_run is not None:
        wandb_run.finish()
    return 0


def train_sft2(args=None) -> int:
    # Let the training frame go out of scope before shutting down NCCL. The
    # runtime/optimizer callbacks may form cycles retaining FSDP parameters,
    # gradients and Adam moments even after the loop itself has returned.
    result = _train_sft2_impl(args)
    gc.collect()
    if torch.cuda.is_available():
        # Surface asynchronous CUDA failures; never turn failed training or
        # teardown into a successful exit. Release cached blocks only after
        # outstanding work is complete, leaving headroom for NCCL shutdown.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    cleanup_dist()
    return result


def main() -> int:
    return train_sft2()
