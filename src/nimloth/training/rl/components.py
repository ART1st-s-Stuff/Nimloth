"""构造 RL 阶段使用的 Qwen、WM 组件、EMA、分布式包装与 optimizer。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.checkpoint import load_adapter_state
from nimloth.backbone.qwen25vl.loading import (
    load_qwen_processor,
    qwen_hidden_size,
)
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.backbone.qwen25vl.tuning import (
    configure_qwen_tuning,
    resolve_tune_modes,
    uses_lora,
)
from nimloth.backbone.qwen25vl.vision_ema import (
    VisionEncoderEMA,
    resolve_vision_ema,
)
from nimloth.config.rl import RLConfig
from nimloth.model import NimlothModel
from nimloth.training.rl.checkpoint import load_rl_wm_checkpoint
from nimloth.util.distributed import broadcast_module_state, is_main
from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead
from nimloth.wm.model import WorldModel


@dataclass(frozen=True)
class RLResumeState:
    start_iteration: int = 1
    global_step: int = 0
    best_eval_metric: float = float("-inf")


@dataclass(frozen=True)
class RLComponents:
    nimloth_model: NimlothModel
    processor: Any
    token_id_map: dict[str, int]
    vision_ema: VisionEncoderEMA | None
    optimizer: torch.optim.Optimizer
    base_model_path: str
    llm_tune: str
    vision_tune: str
    resume: RLResumeState


def _load_qwen(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, int], str, Path | None]:
    """按 RL checkpoint 语义加载并配置 Qwen。"""

    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=1,
    )
    processor = processor_bundle.processor
    tokenizer_vocab = len(processor.tokenizer)
    resume_dir = output_dir / "latest"
    resume_state_path = resume_dir / "rl_state.pt"
    resume_adapter = resume_dir / "adapter_config.json"
    base_model_path = str(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.resize_token_embeddings(tokenizer_vocab)

    resume_aux_dir: Path | None = None
    if args.resume and resume_state_path.is_file() and resume_adapter.is_file():
        if not uses_lora(args):
            raise ValueError(
                "--resume with LoRA adapter requires a LoRA tune mode"
            )
        saved = torch.load(
            resume_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if saved.get("base_model_path"):
            base_model_path = str(saved["base_model_path"])
        model = configure_qwen_tuning(model, args)
        report = load_adapter_state(model, resume_dir)
        resume_aux_dir = resume_dir
        if is_main():
            print(
                json.dumps(
                    {
                        "resume_lora_adapter": str(resume_dir),
                        "base_model_path": base_model_path,
                        "missing_keys": report.missing_keys,
                        "unexpected_keys": report.unexpected_keys,
                        "vision_full_state_loaded": (
                            report.vision_full_state_loaded
                        ),
                    }
                )
            )
    elif (
        args.resume
        and resume_state_path.is_file()
        and (resume_dir / "config.json").is_file()
    ):
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with LoRA tuning")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_dir,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.resize_token_embeddings(tokenizer_vocab)
        model = configure_qwen_tuning(model, args)
        resume_aux_dir = resume_dir
        if is_main():
            print(json.dumps({"resume_full": str(resume_dir)}))
    else:
        model = configure_qwen_tuning(model, args)

    validate_agent_policy_protocol(model.config)
    model.to(device)
    return (
        model,
        processor,
        processor_bundle.token_id_map,
        base_model_path,
        resume_aux_dir,
    )


def _build_wm_components(
    args: argparse.Namespace,
    config: RLConfig,
    *,
    qwen_model: torch.nn.Module,
    device: torch.device,
) -> WorldModel:
    """按真实 Qwen hidden size 构造并应用显式 warm-start。"""

    if args.wm_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint)
    else:
        wm_predictor = LatentWMPredictor.create(
            LeWMConfig(
                emb_dim=config.predictor.emb_dim,
                history_size=config.predictor.history_size,
            )
        )
    state_proj = StateProjector(
        qwen_hidden_dim=qwen_hidden_size(qwen_model.config),
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

    state_proj.to(device)
    wm_predictor.to(device)
    value_head.to(device)
    return WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )


def _wrap_qwen_fsdp(
    model: torch.nn.Module,
    *,
    world_size: int,
) -> torch.nn.Module:
    if world_size <= 1:
        return model
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )

    # FULL_SHARD 后并非每个 rank 都拥有 padding row，保留 padding_idx 会触发
    # embedding 内部的局部 shape 断言；Qwen 不依赖该 row 在 forward 时清零。
    embedding = model.get_input_embeddings()
    if getattr(embedding, "padding_idx", None) is not None:
        embedding.padding_idx = None
    wrapped = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )
    if is_main():
        print(json.dumps({"fsdp": "wrapped", "world_size": world_size}))
    return wrapped


def _load_resume_state(
    *,
    checkpoint_dir: Path | None,
    wm: WorldModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
    world_size: int,
    expected_checkpoint_metric: str,
) -> RLResumeState:
    if checkpoint_dir is None:
        return RLResumeState()
    state = load_rl_wm_checkpoint(
        checkpoint_dir,
        wm,
        device,
    )
    if not state:
        return RLResumeState()
    if world_size > 1:
        saved_world = int(state.get("optimizer_world_size", 0))
        if saved_world != world_size:
            raise RuntimeError(
                f"FSDP optimizer checkpoint world_size={saved_world}, "
                f"current={world_size}"
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
    )


def build_rl_components(
    args: argparse.Namespace,
    config: RLConfig,
    *,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> RLComponents:
    """创建完整 RL component graph，并恢复可选训练状态。"""

    llm_tune, vision_tune = resolve_tune_modes(args)
    model, processor, token_id_map, base_model_path, resume_dir = _load_qwen(
        args,
        output_dir=output_dir,
        device=device,
    )
    wm = _build_wm_components(
        args,
        config,
        qwen_model=model,
        device=device,
    )
    if world_size > 1:
        broadcast_module_state(wm.state_proj)
        broadcast_module_state(wm.wm_predictor)
        broadcast_module_state(wm.value_head)

    vision_ema: VisionEncoderEMA | None = None
    if resolve_vision_ema(args, vision_tune):
        if world_size > 1:
            raise RuntimeError(
                "Vision EMA 尚未验证 FSDP shard 语义；多卡时请显式关闭 EMA"
            )
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = output_dir / "latest" / "vision_ema.pt"
        if args.resume and ema_path.is_file():
            vision_ema = VisionEncoderEMA.load_checkpoint(
                ema_path,
                map_location=device,
            )

    model = _wrap_qwen_fsdp(model, world_size=world_size)
    parameter_groups: list[dict[str, Any]] = []
    qwen_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if qwen_parameters:
        parameter_groups.append(
            {"params": qwen_parameters, "lr": config.actor.lr, "name": "qwen"}
        )
    for name, module, learning_rate in (
        ("state_proj", wm.state_proj, config.predictor.lr),
        ("value_head", wm.value_head, config.value_head.lr),
        ("wm_predictor", wm.wm_predictor, config.predictor.lr),
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
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    resume = _load_resume_state(
        checkpoint_dir=resume_dir,
        wm=wm,
        optimizer=optimizer,
        device=device,
        rank=rank,
        world_size=world_size,
        expected_checkpoint_metric=config.validation.checkpoint_metric,
    )
    nimloth_model = NimlothModel(
        llm=model,
        wm=wm,
    )
    return RLComponents(
        nimloth_model=nimloth_model,
        processor=processor,
        token_id_map=token_id_map,
        vision_ema=vision_ema,
        optimizer=optimizer,
        base_model_path=base_model_path,
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        resume=resume,
    )
