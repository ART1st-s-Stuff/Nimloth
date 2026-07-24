"""RL 训练入口：校验运行模式、装配依赖并启动训练循环。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nimloth.agent import Agent, PlanningPolicy
from nimloth.backbone import (
    backbone_hidden_size,
    build_action_log_prob_replay,
    build_agent_policy,
    build_input_builder,
    build_vision_ema,
    load_backbone,
    resolve_tune_modes,
    resolve_vision_ema,
)
from nimloth.config.rl import RLConfig
from nimloth.rollout import FreshJSONLRolloutCollector, RolloutCollector
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.checkpoint import load_rl_wm_checkpoint
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.loop import RLLoopState, RLTrainingLoop
from nimloth.training.rl.reporting import RLReporter
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.training.rl.rollout_runtime import (
    bind_online_collectors,
    online_policy_required,
    validate_collector_configuration,
    validate_fresh_rollout_policy,
    validate_online_policy_configuration,
    validate_planning_initialization,
)
from nimloth.util.distributed import (
    broadcast_module_state,
    cleanup_dist,
    is_main,
    setup_dist,
)
from nimloth.util.optim import OptimizationRuntime
from nimloth.wm import (
    LeWMConfig,
    LatentWMPredictor,
    StateProjector,
    SequenceSIGReg,
    ValueHead,
    WorldModel,
)
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)


@dataclass(frozen=True)
class RLResumeState:
    start_iteration: int = 1
    global_step: int = 0
    best_eval_metric: float = float("-inf")
    loaded: bool = False


def _is_grid_predictor_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return "grid_tokens" in raw


def _grid_mlp_hidden_dim(
    state: dict[str, torch.Tensor],
    *,
    first_weight: str,
    emb_dim: int,
) -> int:
    weight = state.get(first_weight)
    if weight is None or weight.ndim != 2 or weight.shape[1] != emb_dim:
        raise ValueError(
            "cannot infer grid MLP hidden_dim from checkpoint tensor "
            f"{first_weight!r}"
        )
    return int(weight.shape[0])


def _build_grid_world_model(
    args: argparse.Namespace,
    config: RLConfig,
    *,
    llm: torch.nn.Module,
    device: torch.device,
) -> GridWorldModel:
    """从完整 SFT2 DINO-grid checkpoint 构造 RL world model。

    RL 不计算 DINO loss，也不更新 DINO decoder 或 EMA target encoder；二者仍
    严格加载和保存，以保证 checkpoint 可恢复且 grid WM target 语义不变。
    """

    if args.state_proj_checkpoint is None or args.value_head_checkpoint is None:
        raise ValueError(
            "grid RL requires --state-proj-checkpoint and --value-head-checkpoint"
        )
    wm_checkpoint = Path(args.wm_checkpoint)
    checkpoint_root = wm_checkpoint.parent
    predictor = TemporalSpatialGridPredictor.load_checkpoint(
        wm_checkpoint,
        map_location="cpu",
    )
    if predictor.config.history_size != config.predictor.history_size:
        raise ValueError(
            "RL grid WM checkpoint history_size does not match config: "
            f"checkpoint={predictor.config.history_size}, "
            f"config={config.predictor.history_size}"
        )
    if predictor.config.emb_dim != config.predictor.emb_dim:
        raise ValueError(
            "RL grid WM checkpoint emb_dim does not match config: "
            f"checkpoint={predictor.config.emb_dim}, "
            f"config={config.predictor.emb_dim}"
        )

    state_proj_state = torch.load(
        args.state_proj_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    encoder_hidden_dim = _grid_mlp_hidden_dim(
        state_proj_state,
        first_weight="online_encoder.net.net.0.weight",
        emb_dim=predictor.config.emb_dim,
    )
    slot_first = state_proj_state.get("slot_projector.net.0.weight")
    slot_last = state_proj_state.get("slot_projector.net.3.weight")
    qwen_hidden_dim = backbone_hidden_size(llm.config)
    if (
        slot_first is None
        or slot_last is None
        or slot_first.ndim != 2
        or slot_last.ndim != 2
        or slot_first.shape[1] != qwen_hidden_dim
        or slot_last.shape[0] != predictor.config.emb_dim
        or slot_last.shape[1] != slot_first.shape[0]
    ):
        raise ValueError(
            "SFT2 grid state projector is incompatible with the Qwen/predictor "
            "dimensions"
        )
    slot_projector = SharedSlotProjector(
        input_dim=qwen_hidden_dim,
        output_dim=predictor.config.emb_dim,
        hidden_dim=int(slot_first.shape[0]),
        grid_tokens=predictor.config.grid_tokens,
    ).to(dtype=slot_first.dtype)
    state_proj = GridStateProjector(
        slot_projector,
        LeWMGridEncoder(
            emb_dim=predictor.config.emb_dim,
            hidden_dim=encoder_hidden_dim,
        ),
    )
    state_proj.load_state_dict(state_proj_state)

    metadata_path = checkpoint_root / "dino_grid_config.json"
    decoder_path = checkpoint_root / "dino_grid_decoder.pt"
    if not metadata_path.is_file() or not decoder_path.is_file():
        raise FileNotFoundError(
            f"incomplete DINO-grid checkpoint extras under {checkpoint_root}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    decoder_state = torch.load(
        decoder_path,
        map_location="cpu",
        weights_only=True,
    )
    decoder_hidden_dim = _grid_mlp_hidden_dim(
        decoder_state,
        first_weight="net.net.0.weight",
        emb_dim=predictor.config.emb_dim,
    )
    target_encoder = EMATargetGridEncoder(
        state_proj.online_encoder,
        decay=float(metadata["ema_decay"]),
    )
    value_head = ValueHead.load_checkpoint(
        args.value_head_checkpoint,
        emb_dim=predictor.config.emb_dim,
        map_location="cpu",
    )
    world_model = GridWorldModel(
        state_proj=state_proj,
        target_encoder=target_encoder,
        wm_predictor=predictor,
        dino_decoder=LeWMGridDecoder(
            emb_dim=predictor.config.emb_dim,
            hidden_dim=decoder_hidden_dim,
        ),
        value_head=value_head,
        train_dino_decoder=False,
        update_target_encoder=False,
    )
    world_model.load_checkpoint_extras(
        checkpoint_root,
        map_location=torch.device("cpu"),
    )
    world_model.target_encoder.requires_grad_(False).eval()
    world_model.dino_decoder.requires_grad_(False).eval()
    if config.freeze.state_proj:
        world_model.state_proj.requires_grad_(False).eval()
    return world_model.to(device)


def _build_world_model(
    args: argparse.Namespace,
    config: RLConfig,
    *,
    llm: torch.nn.Module,
    device: torch.device,
) -> WorldModel:
    """构造 RL 使用的 WorldModel，并加载显式指定的子模块 checkpoint。"""

    if args.wm_checkpoint is not None and _is_grid_predictor_checkpoint(
        Path(args.wm_checkpoint)
    ):
        return _build_grid_world_model(
            args,
            config,
            llm=llm,
            device=device,
        )

    if args.wm_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint)
        if wm_predictor.config.history_size != config.predictor.history_size:
            raise ValueError(
                "RL WM checkpoint history_size does not match config: "
                f"checkpoint={wm_predictor.config.history_size}, "
                f"config={config.predictor.history_size}"
            )
    else:
        wm_predictor = LatentWMPredictor.create(
            LeWMConfig(
                emb_dim=config.predictor.emb_dim,
                history_size=config.predictor.history_size,
            )
        )
    state_proj = StateProjector(
        qwen_hidden_dim=backbone_hidden_size(llm.config),
        lewm_emb_dim=wm_predictor.emb_dim,
    )
    value_head = ValueHead(emb_dim=wm_predictor.emb_dim)

    if args.state_proj_checkpoint is not None:
        state_proj.load_state_dict(
            torch.load(
                args.state_proj_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        )
    if args.value_head_checkpoint is not None:
        loaded_head = ValueHead.load_checkpoint(
            args.value_head_checkpoint,
            emb_dim=wm_predictor.emb_dim,
        )
        value_head.load_state_dict(loaded_head.state_dict())

    if config.freeze.state_proj:
        state_proj.eval()
        for parameter in state_proj.parameters():
            parameter.requires_grad = False

    return WorldModel(
        state_proj=state_proj.to(device),
        wm_predictor=wm_predictor.to(device),
        value_head=value_head.to(device),
    )


def _wrap_llm_fsdp(
    llm: torch.nn.Module,
    *,
    world_size: int,
) -> torch.nn.Module:
    if world_size <= 1:
        return llm
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )

    # FULL_SHARD 的局部 embedding 不保证包含 padding row。
    embedding = llm.get_input_embeddings()
    if getattr(embedding, "padding_idx", None) is not None:
        embedding.padding_idx = None
    wrapped = FSDP(
        llm,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )
    if is_main():
        print(json.dumps({"fsdp": "wrapped", "world_size": world_size}))
    return wrapped


def _build_optimizer(
    model: torch.nn.Module,
    world_model: WorldModel,
    config: RLConfig,
) -> torch.optim.Optimizer:
    """按 Agent 子模块建立 RL 参数组。"""

    parameter_groups: list[dict[str, Any]] = []
    qwen_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if qwen_parameters:
        parameter_groups.append(
            {
                "params": qwen_parameters,
                "lr": config.gradient.backbone_lr,
                "name": "qwen",
            }
        )
    for name, module, learning_rate in (
        ("state_proj", world_model.state_proj, config.predictor.lr),
        ("value_head", world_model.value_head, config.value_head.lr),
        ("wm_predictor", world_model.wm_predictor, config.predictor.lr),
    ):
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if parameters:
            parameter_groups.append(
                {"params": parameters, "lr": learning_rate, "name": name}
            )
    if not parameter_groups:
        raise ValueError("RL configuration leaves no trainable parameters")
    return torch.optim.AdamW(parameter_groups, weight_decay=1e-4)


def _load_resume_state(
    *,
    checkpoint_dir: Path | None,
    world_model: WorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
    world_size: int,
    expected_checkpoint_metric: str,
) -> RLResumeState:
    """恢复 WM、optimizer 和 iteration 位置。"""

    if checkpoint_dir is None:
        return RLResumeState()
    state = load_rl_wm_checkpoint(checkpoint_dir, world_model, device)
    if not state:
        return RLResumeState()
    if world_size > 1:
        saved_world = int(state.get("optimizer_world_size", 0))
        if saved_world != world_size:
            raise RuntimeError(
                f"FSDP optimizer checkpoint world_size={saved_world}, current={world_size}"
            )
        optimizer_path = checkpoint_dir / f"optimizer_rank_{rank:05d}.pt"
        if not optimizer_path.is_file():
            raise FileNotFoundError(
                f"missing rank optimizer checkpoint: {optimizer_path}"
            )
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
    elif state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    checkpoint_metric = state.get("checkpoint_metric")
    if checkpoint_metric is not None and checkpoint_metric != expected_checkpoint_metric:
        raise ValueError(
            "resume checkpoint metric mismatch: "
            f"saved={checkpoint_metric!r}, configured={expected_checkpoint_metric!r}"
        )
    return RLResumeState(
        start_iteration=int(state.get("iteration", 0)) + 1,
        global_step=int(state.get("global_step", 0)),
        best_eval_metric=float(state.get("best_eval_metric", float("-inf"))),
        loaded=True,
    )


def train_rl(
    *,
    args: argparse.Namespace,
    config: RLConfig,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    output_dir: Path,
) -> int:
    """装配 RL runtime；核心 batch 算法见 ``RLAlgorithm``。"""

    llm_tune, vision_tune = resolve_tune_modes(args)
    actor_enabled = config.actor.enabled
    planning_enabled = config.agent.planning.enabled
    validate_online_policy_configuration(
        actor_enabled=actor_enabled,
        planning_enabled=planning_enabled,
    )
    backbone_trainable = llm_tune != "freeze" or vision_tune != "freeze"
    if actor_enabled and not backbone_trainable:
        raise ValueError(
            "actor.enabled requires a trainable --llm-tune or --vision-tune mode"
        )
    validate_collector_configuration(
        actor_enabled=actor_enabled,
        train_collector=train_collector,
        eval_collector=eval_collector,
        validation_enabled=config.validation.enabled,
    )
    if (
        actor_enabled
        and config.actor.credit_assignment == "turn"
        and not isinstance(train_collector, FreshJSONLRolloutCollector)
    ):
        raise ValueError(
            "actor.credit_assignment=turn requires a fresh vLLM JSONL rollout"
        )
    needs_online_policy = online_policy_required(
        train_collector,
        eval_collector,
    )
    if not args.resume:
        # 非 resume 运行可以在加载大模型前完成 planning artifact 校验。
        validate_planning_initialization(
            planning_enabled=planning_enabled,
            online_policy_needed=needs_online_policy,
            resume_loaded=False,
            wm_checkpoint=args.wm_checkpoint,
            state_proj_checkpoint=args.state_proj_checkpoint,
            value_head_checkpoint=args.value_head_checkpoint,
        )

    rank, world, _, device = setup_dist()
    validate_fresh_rollout_policy(train_collector)
    if actor_enabled and world > 1 and not isinstance(
        train_collector,
        FreshJSONLRolloutCollector,
    ):
        raise RuntimeError(
            "multi-rank PPO actor is disabled until rollout freshness and FSDP "
            "forward/EMA semantics have dedicated integration coverage"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.training.seed)
    reporter = RLReporter(
        rank=rank,
        output_dir=output_dir,
        run_name=args.experiment_name,
        config=config,
    )

    try:
        from transformers import AutoConfig
        from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol

        latent_token_count = validate_agent_policy_protocol(
            AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        )
        resume_dir = output_dir / "latest"
        loaded = load_backbone(
            args,
            device=device,
            latent_token_count=latent_token_count,
            resume_dir=resume_dir,
            resume_state_path=resume_dir / "rl_state.pt",
        )
        model = loaded.backbone.model
        world_model = _build_world_model(args, config, llm=model, device=device)
        if world > 1:
            broadcast_module_state(world_model.state_proj)
            broadcast_module_state(world_model.wm_predictor)
            broadcast_module_state(world_model.value_head)
            if isinstance(world_model, GridWorldModel):
                broadcast_module_state(world_model.target_encoder)
                broadcast_module_state(world_model.dino_decoder)

        vision_ema_enabled = resolve_vision_ema(args, vision_tune)
        if vision_ema_enabled and world > 1:
            raise RuntimeError(
                "Vision EMA 尚未验证 FSDP shard 语义；多卡时请显式关闭 EMA"
            )
        vision_ema = build_vision_ema(
            enabled=vision_ema_enabled,
            decay=args.vision_ema_decay,
            llm=model,
            resume_path=(output_dir / "latest" / "vision_ema.pt") if args.resume else None,
            device=device,
        )
        model = _wrap_llm_fsdp(model, world_size=world)
        optimizer = _build_optimizer(model, world_model, config)
        resume = _load_resume_state(
            checkpoint_dir=loaded.resume_aux_dir,
            world_model=world_model,
            optimizer=optimizer,
            device=device,
            rank=rank,
            world_size=world,
            expected_checkpoint_metric=config.validation.checkpoint_metric,
        )
        agent = Agent(
            backbone=loaded.backbone.with_model(model),
            wm=world_model,
        )
        input_builder = build_input_builder(
            loaded,
            max_length=999_999,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=True,
        )
        # resume 只有在 checkpoint state 确实恢复后才算有效，不能只相信 CLI flag。
        validate_planning_initialization(
            planning_enabled=planning_enabled,
            online_policy_needed=needs_online_policy,
            resume_loaded=resume.loaded,
            wm_checkpoint=args.wm_checkpoint,
            state_proj_checkpoint=args.state_proj_checkpoint,
            value_head_checkpoint=args.value_head_checkpoint,
        )
        if needs_online_policy:
            if planning_enabled:
                policy = PlanningPolicy(
                    agent=agent,
                    input_builder=input_builder,
                    horizon=config.agent.planning.horizon,
                    beam_width=config.agent.planning.beam_width,
                    temperature=config.rollout.temperature,
                    top_p=config.rollout.top_p,
                )
            else:
                policy = build_agent_policy(
                    loaded,
                    model=model,
                    device=device,
                    temperature=config.rollout.temperature,
                    top_p=config.rollout.top_p,
                )
            bind_online_collectors(
                train_collector=train_collector,
                eval_collector=eval_collector,
                policy=policy,
                latent_token_count=latent_token_count,
                world_size=world,
            )
            if is_main():
                print(
                    json.dumps(
                        {
                            "agent_policy": (
                                "wm_planning" if planning_enabled else "qwen_direct"
                            ),
                            "planning_horizon": (
                                config.agent.planning.horizon
                                if planning_enabled
                                else None
                            ),
                            "planning_beam_width": (
                                config.agent.planning.beam_width
                                if planning_enabled
                                else None
                            ),
                        }
                    )
                )
        checkpoint_manager = RLCheckpointManager(
            config=config,
            agent=agent,
            processor=loaded.processor,
            vision_ema=vision_ema,
            optimizer=optimizer,
            base_model_path=str(loaded.base_model_path),
            llm_tune=llm_tune,
            vision_tune=vision_tune,
        )
        algorithm = RLAlgorithm(
            history_size=config.predictor.history_size,
            sigreg=(
                SequenceSIGReg(
                    knots=config.predictor.sigreg_knots,
                    num_proj=config.predictor.sigreg_num_proj,
                ).to(device)
                if config.predictor.lambda_sigreg > 0.0
                else None
            ),
            sigreg_weight=config.predictor.lambda_sigreg,
            value_rank_margin=config.value_head.rank_margin,
            value_rank_weight=config.value_head.lambda_rank,
            ppo_clip_ratio=config.actor.clip_ratio,
            entropy_weight=config.actor.entropy_coeff,
            credit_assignment=config.actor.credit_assignment,
        )
        model_runtime = RLModelRuntime(
            agent=agent,
            input_builder=input_builder,
            representation_to_backbone=(
                config.gradient.representation_to_backbone
            ),
            policy_replay=(
                build_action_log_prob_replay(
                    loaded,
                    model=model,
                    device=device,
                )
                if actor_enabled
                else None
            ),
        )
        optimization_runtime = OptimizationRuntime(
            optimizer=optimizer,
            synchronized_modules=agent.synchronized_modules,
            after_step=(
                lambda: vision_ema.update(agent.backbone.model)
                if vision_ema is not None
                else None
            ),
        )
        loop = RLTrainingLoop(
            config=config,
            algorithm=algorithm,
            model_runtime=model_runtime,
            optimization_runtime=optimization_runtime,
            device=device,
            train_collector=train_collector,
            eval_collector=eval_collector,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_manager,
            reporter=reporter,
            start_iteration=resume.start_iteration,
            state=RLLoopState(
                global_step=resume.global_step,
                best_eval_metric=resume.best_eval_metric,
            ),
        )
        loop.run()
        return 0
    finally:
        reporter.finish()
        cleanup_dist()
