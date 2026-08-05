"""RL 训练入口：校验运行模式、装配依赖并启动训练循环。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from nimloth.agent import Agent
from nimloth.backbone import (
    Backbone,
    DINOV2_LARGE_IDENTITY,
    DistributedBackbone,
    FrozenDINOGridTargets,
    backbone_hidden_size,
    build_action_log_prob_replay,
    build_agent_policy,
    build_input_builder,
    build_vision_ema,
    load_backbone,
    model_output_device,
    resolve_tune_modes,
    resolve_vision_ema,
)
from nimloth.config.rl import RLConfig
from nimloth.rollout import FreshJSONLRolloutCollector, RolloutCollector
from nimloth.training.rl.algorithm import PLANNER_TRAINING_OBJECTIVE, RLAlgorithm
from nimloth.training.rl.checkpoint import load_rl_wm_checkpoint
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.loop import RLLoopState, RLTrainingLoop
from nimloth.training.rl.planning_loader import validate_planning_value_semantics
from nimloth.training.rl.reporting import RLReporter
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.training.rl.token_value import TokenValueHead
from nimloth.training.rl.rollout_runtime import (
    bind_online_collectors,
    online_policy_required,
    validate_collector_configuration,
    validate_fresh_rollout_policy,
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
    GridWorldModel,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)


@dataclass(frozen=True)
class RLResumeState:
    start_iteration: int = 1
    global_step: int = 0
    best_eval_metric: float = float("-inf")
    loaded: bool = False


@dataclass(frozen=True)
class RLDistributedModules:
    """分布式包装后的 Agent 参数边界与 checkpoint 布局。"""

    backbone: Backbone
    world_model: WorldModel
    token_value_head: torch.nn.Module | None
    optimizer_state_sharded: bool
    strategy: str

    @property
    def model(self) -> torch.nn.Module:
        """返回 optimizer、EMA 与 checkpoint 共享的底层语言模型。"""

        return self.backbone.model


def _is_grid_predictor_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return "grid_tokens" in raw


def _build_grid_world_model(
    args: argparse.Namespace,
    config: RLConfig,
    *,
    llm: torch.nn.Module,
    device: torch.device,
) -> GridWorldModel:
    """从 SFT2 checkpoint 构造使用同一 projector state 的 RL world model。"""

    if args.state_proj_checkpoint is None or args.value_head_checkpoint is None:
        raise ValueError(
            "grid RL requires --state-proj-checkpoint and --value-head-checkpoint"
        )
    wm_checkpoint = Path(args.wm_checkpoint)
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
    projector_first = state_proj_state["net.0.weight"]
    qwen_hidden_dim = backbone_hidden_size(llm.config)
    state_proj = SharedSlotProjector(
        input_dim=qwen_hidden_dim,
        output_dim=predictor.config.emb_dim,
        hidden_dim=int(projector_first.shape[0]),
        grid_tokens=predictor.config.grid_tokens,
    ).to(dtype=projector_first.dtype)
    state_proj.load_state_dict(state_proj_state)
    value_head = ValueHead.load_checkpoint(
        args.value_head_checkpoint,
        emb_dim=predictor.config.emb_dim,
        map_location="cpu",
    )
    world_model = GridWorldModel(
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
    )
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

    explicit_planning_modules = (
        args.wm_checkpoint,
        args.state_proj_checkpoint,
        args.value_head_checkpoint,
    )
    if config.agent.planning.enabled and all(
        path is not None for path in explicit_planning_modules
    ):
        validate_planning_value_semantics(
            wm_checkpoint=args.wm_checkpoint,
            state_proj_checkpoint=args.state_proj_checkpoint,
            value_head_checkpoint=args.value_head_checkpoint,
        )
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


def _wrap_trainable_ddp(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """只包装真正可训练的单设备辅助模块。"""

    if not any(parameter.requires_grad for parameter in module.parameters()):
        return module
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_index = int(str(device).split(":")[-1])
    return DDP(
        module,
        device_ids=[device_index],
        output_device=device_index,
        find_unused_parameters=False,
        static_graph=True,
    )


def _wrap_model_parallel_backbone_ddp(
    backbone: Backbone,
    *,
    world_size: int,
) -> Backbone:
    """用 DDP 同步直接返回 critic hidden 的 Backbone forward。"""

    if world_size <= 1:
        return backbone
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DistributedBackbone(
        DDP(
            backbone,
            device_ids=None,
            output_device=None,
            # PyTorch 2.8 static_graph calls prepare_for_backward([]) and relies
            # on its first-iteration sink.  Qwen critic forwards intentionally
            # leave lm_head unused and return a hook-derived hidden tensor, so
            # force the reducer to traverse that returned graph every epoch.
            find_unused_parameters=True,
            static_graph=False,
        )
    )


def _wrap_model_parallel_llm_ddp(
    llm: torch.nn.Module,
    *,
    world_size: int,
) -> torch.nn.Module:
    """用 DDP 同步直接返回 actor logits 的底层语言模型。"""

    if world_size <= 1:
        return llm
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(
        llm,
        device_ids=None,
        output_device=None,
        find_unused_parameters=False,
        static_graph=True,
    )


def _wrap_world_model_ddp(
    world_model: WorldModel,
    *,
    device: torch.device,
    world_size: int,
) -> WorldModel:
    if world_size <= 1:
        return world_model
    state_proj = _wrap_trainable_ddp(world_model.state_proj, device=device)
    wm_predictor = _wrap_trainable_ddp(world_model.wm_predictor, device=device)
    value_head = _wrap_trainable_ddp(world_model.value_head, device=device)
    if isinstance(world_model, GridWorldModel):
        return GridWorldModel(
            state_proj=state_proj,  # type: ignore[arg-type]
            wm_predictor=wm_predictor,  # type: ignore[arg-type]
            value_head=value_head,  # type: ignore[arg-type]
        )
    return WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )


def _wrap_distributed_modules(
    backbone: Backbone,
    world_model: WorldModel,
    token_value_head: torch.nn.Module | None,
    *,
    world_size: int,
    model_parallel: bool,
    synchronize_backbone_hidden: bool,
    training_device: torch.device,
) -> RLDistributedModules:
    if model_parallel:
        if synchronize_backbone_hidden:
            # Planner critic 直接消费 Backbone hidden，DDP 必须包住
            # 这个forward边界；只包HF model会让hook hidden绕过reducer。
            backbone = _wrap_model_parallel_backbone_ddp(
                backbone,
                world_size=world_size,
            )
        else:
            # Direct-Qwen actor 的 loss 消费底层 model logits，保留原有
            # DDP 边界，防止修复 planner 时绕过 actor reducer。
            llm = _wrap_model_parallel_llm_ddp(
                backbone.model,
                world_size=world_size,
            )
            backbone = backbone.with_model(llm)
        world_model = _wrap_world_model_ddp(
            world_model,
            device=training_device,
            world_size=world_size,
        )
        if token_value_head is not None:
            token_value_head = _wrap_trainable_ddp(
                token_value_head,
                device=training_device,
            )
        strategy = "model_parallel_ddp" if world_size > 1 else "model_parallel"
        optimizer_state_sharded = False
    else:
        llm = backbone.model
        llm = _wrap_llm_fsdp(llm, world_size=world_size)
        backbone = backbone.with_model(llm)
        strategy = "fsdp" if world_size > 1 else "single_gpu"
        optimizer_state_sharded = world_size > 1
        world_model = _wrap_world_model_ddp(
            world_model,
            device=training_device,
            world_size=world_size,
        )
        if token_value_head is not None:
            token_value_head = _wrap_trainable_ddp(
                token_value_head,
                device=training_device,
            )
    if is_main():
        print(
            json.dumps(
                {
                    "rl_distributed_strategy": strategy,
                    "world_size": world_size,
                    "training_device": str(training_device),
                }
            )
        )
    return RLDistributedModules(
        backbone=backbone,
        world_model=world_model,
        token_value_head=token_value_head,
        optimizer_state_sharded=optimizer_state_sharded,
        strategy=strategy,
    )


def _build_optimizer(
    model: torch.nn.Module,
    world_model: WorldModel,
    token_value_head: torch.nn.Module | None,
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
                "weight_decay": config.gradient.backbone_weight_decay,
                "name": "qwen",
            }
        )
    for name, module, learning_rate in (
        ("state_proj", world_model.state_proj, config.predictor.lr),
        ("value_head", world_model.value_head, config.value_head.lr),
        ("wm_predictor", world_model.wm_predictor, config.predictor.lr),
    ):
        if name == "wm_predictor" and not config.predictor.train_wm:
            continue
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if parameters:
            parameter_groups.append(
                {"params": parameters, "lr": learning_rate, "name": name}
            )
    if token_value_head is not None:
        assert config.token_credit.value_lr is not None
        token_value_parameters = [
            parameter
            for parameter in token_value_head.parameters()
            if parameter.requires_grad
        ]
        if token_value_parameters:
            parameter_groups.append(
                {
                    "params": token_value_parameters,
                    "lr": config.token_credit.value_lr,
                    "name": "token_value_head",
                }
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
    optimizer_state_sharded: bool,
    expected_checkpoint_metric: str,
    expected_credit_assignment: str,
    expected_token_credit_config: dict[str, Any],
    expected_truncated_bootstrap: str | None,
    expected_planner_config: dict[str, Any],
    expected_planner_training_objective: str | None,
    expected_planner_value_config: dict[str, Any] | None,
    expected_reference_kl_config: dict[str, Any],
    expected_train_world_model: bool,
) -> RLResumeState:
    """恢复 WM、optimizer 和 iteration 位置。"""

    if checkpoint_dir is None:
        return RLResumeState()
    state = load_rl_wm_checkpoint(checkpoint_dir, world_model, device)
    if not state:
        return RLResumeState()
    if optimizer_state_sharded:
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
    else:
        saved_optimizer = state.get("optimizer")
        if saved_optimizer is None:
            raise RuntimeError(
                "replicated optimizer checkpoint is missing rl_state.optimizer"
            )
        optimizer.load_state_dict(saved_optimizer)
    checkpoint_metric = state.get("checkpoint_metric")
    if checkpoint_metric is not None and checkpoint_metric != expected_checkpoint_metric:
        raise ValueError(
            "resume checkpoint metric mismatch: "
            f"saved={checkpoint_metric!r}, configured={expected_checkpoint_metric!r}"
        )
    saved_credit_assignment = state.get("credit_assignment")
    if (
        saved_credit_assignment is not None
        and str(saved_credit_assignment) != expected_credit_assignment
    ):
        raise ValueError(
            "resume credit assignment mismatch: "
            f"saved={saved_credit_assignment!r}, "
            f"configured={expected_credit_assignment!r}"
        )
    if expected_credit_assignment == "token":
        if saved_credit_assignment is None:
            raise ValueError("resume token credit checkpoint lacks credit metadata")
        if state.get("token_credit_config") != expected_token_credit_config:
            raise ValueError("resume token credit config mismatch")
        if state.get("truncated_bootstrap") != expected_truncated_bootstrap:
            raise ValueError("resume truncation bootstrap config mismatch")
    saved_planner_config = state.get("planner_config")
    if (
        expected_planner_config.get("enabled", False)
        and saved_planner_config != expected_planner_config
    ) or (
        saved_planner_config is not None
        and saved_planner_config != expected_planner_config
    ):
        raise ValueError("resume planner config mismatch")
    if state.get("planner_training_objective") != expected_planner_training_objective:
        raise ValueError("resume planner training objective mismatch")
    if state.get("planner_value_config") != expected_planner_value_config:
        raise ValueError("resume planner value config mismatch")
    saved_reference_kl_config = state.get(
        "reference_kl_config",
        {"weight": 0.0, "type": None},
    )
    if saved_reference_kl_config != expected_reference_kl_config:
        raise ValueError("resume reference KL config mismatch")
    if state.get("train_world_model", True) != expected_train_world_model:
        raise ValueError("resume train_world_model config mismatch")
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
    backbone_trainable = llm_tune != "freeze" or vision_tune != "freeze"
    if actor_enabled and not backbone_trainable:
        raise ValueError(
            "Qwen policy training requires a trainable --llm-tune or "
            "--vision-tune mode"
        )
    if planning_enabled and llm_tune == "freeze":
        raise ValueError(
            "planner value training requires trainable Qwen language parameters; "
            "--llm-tune cannot be freeze"
        )
    if planning_enabled and (
        config.gradient.state_source != "recompute"
        or not config.gradient.representation_to_backbone
    ):
        raise ValueError(
            "planner RL requires differentiable full-prefix Qwen recomputation"
        )
    validate_collector_configuration(
        actor_enabled=actor_enabled,
        planning_enabled=planning_enabled,
        train_collector=train_collector,
        eval_collector=eval_collector,
        validation_enabled=config.validation.enabled,
    )
    if (
        actor_enabled
        and config.actor.credit_assignment in {"turn", "token"}
        and not isinstance(train_collector, FreshJSONLRolloutCollector)
    ):
        raise ValueError(
            f"actor.credit_assignment={config.actor.credit_assignment} requires "
            "a fresh vLLM JSONL rollout"
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

    rank, world, _, device = setup_dist(
        gpu_stride=config.distributed.gpus_per_rank,
    )
    if world != config.distributed.world_size:
        raise RuntimeError(
            "launched distributed world does not match config: "
            f"launched={world}, configured={config.distributed.world_size}"
        )
    validate_fresh_rollout_policy(train_collector)
    if actor_enabled and world > 1 and not isinstance(
        train_collector,
        FreshJSONLRolloutCollector,
    ):
        raise RuntimeError(
            "multi-rank actor training is disabled until rollout freshness and FSDP "
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
        resume_dir = (
            Path(args.resume_checkpoint).resolve()
            if args.resume_checkpoint is not None
            else output_dir / "latest"
        )
        loaded = load_backbone(
            args,
            device=device,
            latent_token_count=latent_token_count,
            model_parallel_size=config.distributed.gpus_per_rank,
            resume_dir=resume_dir,
            resume_state_path=resume_dir / "rl_state.pt",
        )
        if isinstance(train_collector, FreshJSONLRolloutCollector):
            train_collector.manifest.validate_processor(loaded.processor)
        model = loaded.backbone.model
        training_device = model_output_device(model, default=device)
        world_model = _build_world_model(
            args,
            config,
            llm=model,
            device=training_device,
        )
        if not config.predictor.train_wm:
            world_model.wm_predictor.requires_grad_(False).eval()
        token_value_head: torch.nn.Module | None = None
        if actor_enabled and config.actor.credit_assignment == "token":
            assert config.token_credit.hidden_dim is not None
            token_value_head = TokenValueHead(
                input_dim=backbone_hidden_size(model.config),
                hidden_dim=config.token_credit.hidden_dim,
            ).to(training_device)
            if loaded.resume_aux_dir is not None:
                resumed_token_head = TokenValueHead.load_checkpoint(
                    loaded.resume_aux_dir / "token_value_head",
                    map_location=training_device,
                )
                if (
                    resumed_token_head.input_dim != token_value_head.input_dim
                    or resumed_token_head.hidden_dim != token_value_head.hidden_dim
                ):
                    raise ValueError(
                        "resume TokenValueHead dimensions do not match RL config/model"
                    )
                token_value_head.load_state_dict(resumed_token_head.state_dict())
        if world > 1:
            broadcast_module_state(world_model.state_proj)
            broadcast_module_state(world_model.wm_predictor)
            broadcast_module_state(world_model.value_head)
            if token_value_head is not None:
                broadcast_module_state(token_value_head)
        vision_ema_enabled = resolve_vision_ema(args, vision_tune)
        if vision_ema_enabled and world > 1:
            raise RuntimeError(
                "Vision EMA 尚未验证 FSDP shard 语义；多卡时请显式关闭 EMA"
            )
        vision_ema = build_vision_ema(
            enabled=vision_ema_enabled,
            decay=args.vision_ema_decay,
            llm=model,
            resume_path=(resume_dir / "vision_ema.pt") if args.resume else None,
            device=device,
        )
        distributed_modules = _wrap_distributed_modules(
            loaded.backbone,
            world_model,
            token_value_head,
            world_size=world,
            model_parallel=loaded.pair_parallel,
            synchronize_backbone_hidden=planning_enabled,
            training_device=training_device,
        )
        model = distributed_modules.model
        world_model = distributed_modules.world_model
        token_value_head = distributed_modules.token_value_head
        optimizer = _build_optimizer(
            model,
            world_model,
            token_value_head,
            config,
        )
        resume = _load_resume_state(
            checkpoint_dir=loaded.resume_aux_dir,
            world_model=world_model,
            optimizer=optimizer,
            device=training_device,
            rank=rank,
            world_size=world,
            optimizer_state_sharded=distributed_modules.optimizer_state_sharded,
            expected_checkpoint_metric=config.validation.checkpoint_metric,
            expected_credit_assignment=(
                "none" if planning_enabled else config.actor.credit_assignment
            ),
            expected_token_credit_config=asdict(config.token_credit),
            expected_truncated_bootstrap=config.rl.truncated_bootstrap,
            expected_planner_config=asdict(config.agent.planning),
            expected_planner_training_objective=(
                PLANNER_TRAINING_OBJECTIVE
                if planning_enabled
                else None
            ),
            expected_planner_value_config=(
                asdict(config.value_head) if planning_enabled else None
            ),
            expected_reference_kl_config={
                "weight": config.actor.reference_kl_loss_weight,
                "type": config.actor.reference_kl_loss_type,
            },
            expected_train_world_model=config.predictor.train_wm,
        )
        agent = Agent(
            backbone=distributed_modules.backbone,
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
                raise RuntimeError(
                    "planner policy requires the independent vLLM rollout backend; "
                    "collect a fresh traced JSONL before distributed training"
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
            token_value_head=token_value_head,
        )
        algorithm = RLAlgorithm(
            history_size=config.predictor.history_size,
            sigreg=(
                SequenceSIGReg(
                    knots=config.predictor.sigreg_knots,
                    num_proj=config.predictor.sigreg_num_proj,
                ).to(training_device)
                if config.predictor.lambda_sigreg > 0.0
                else None
            ),
            sigreg_weight=config.predictor.lambda_sigreg,
            value_rank_margin=config.value_head.rank_margin,
            value_rank_weight=config.value_head.lambda_rank,
            value_ppo_clip_range=config.value_head.ppo_clip_range,
            ppo_clip_ratio=config.actor.clip_ratio,
            entropy_weight=config.actor.entropy_coeff,
            credit_assignment=config.actor.credit_assignment,
            token_gamma=config.token_credit.gamma,
            token_gae_lambda=config.token_credit.gae_lambda,
            token_value_loss_weight=config.token_credit.value_loss_weight,
            reference_kl_loss_weight=config.actor.reference_kl_loss_weight,
            train_world_model=config.predictor.train_wm,
            world_model_weight=config.predictor.lambda_wm,
            dino_grid_weight=config.predictor.lambda_dino,
        )
        dino_grid_targets = None
        if config.predictor.train_wm and config.predictor.lambda_dino > 0.0:
            if not isinstance(agent.wm, GridWorldModel):
                raise ValueError("RL DINO-grid loss requires a grid world model")
            dino_grid_targets = FrozenDINOGridTargets.from_pretrained(
                DINOV2_LARGE_IDENTITY,
                device=device,
                dtype=torch.bfloat16,
            )
        model_runtime = RLModelRuntime(
            agent=agent,
            input_builder=input_builder,
            state_source=config.gradient.state_source,
            representation_to_backbone=(
                config.gradient.representation_to_backbone
            ),
            policy_replay=(
                build_action_log_prob_replay(
                    loaded,
                    model=model,
                    device=device,
                    token_value_head=token_value_head,
                )
                if actor_enabled
                else None
            ),
            max_state_tokens=config.actor.max_state_tokens,
            dino_grid_targets=dino_grid_targets,
        )
        optimization_runtime = OptimizationRuntime(
            optimizer=optimizer,
            synchronized_modules=(
                *agent.synchronized_modules,
                *((token_value_head,) if token_value_head is not None else ()),
            ),
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
            device=training_device,
            train_collector=train_collector,
            eval_collector=eval_collector,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_manager,
            reporter=reporter,
            write_final_checkpoint=not args.defer_final_checkpoint,
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
