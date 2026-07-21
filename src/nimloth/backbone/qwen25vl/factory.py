"""Qwen2.5-VL 模型、processor、tuning 和 backend 的构造。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.base import LoadedBackbone, RLBackboneAdapters
from nimloth.backbone.qwen25vl.checkpoint import load_adapter_state
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.backbone.qwen25vl.policy import (
    QwenActionLogProbReplay,
    QwenAgentPolicy,
    validate_agent_policy_protocol,
)
from nimloth.backbone.qwen25vl.rollout import QwenRolloutEncoder
from nimloth.backbone.qwen25vl.transition import Qwen25VLBatchBuilder
from nimloth.backbone.qwen25vl.tuning import (
    configure_qwen_tuning,
    resolve_tune_modes,
    uses_lora,
)
from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA
from nimloth.latent import (
    initialize_extra_latent_token_embeddings,
    install_query_embedding_adapter,
    latent_state_tokens,
)
from nimloth.util.distributed import is_main


def _load_kwargs(device: torch.device) -> tuple[bool, dict[str, Any]]:
    gpu_stride = int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
    enabled = gpu_stride > 1 and torch.cuda.is_available()
    if not enabled:
        return False, {}
    primary_idx = int(str(device).split(":")[-1])
    pair = [primary_idx + offset for offset in range(gpu_stride)]
    if is_main():
        print(json.dumps({"qwen_pair_parallel": True, "gpu_stride": gpu_stride, "rank0_pair": pair}))
    return True, {
        "device_map": "auto",
        "max_memory": {index: "74GiB" for index in pair} | {"cpu": "64GiB"},
        "low_cpu_mem_usage": True,
    }


def _configure_shape(model, processor, args, token_id_map, added_count: int) -> None:
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.resize_token_embeddings(len(processor.tokenizer))
    if added_count > 0:
        initialize_extra_latent_token_embeddings(
            model,
            token_id_map,
            latent_token_count=args.latent_token_count,
        )
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.vocab_size = len(processor.tokenizer)


def load_sft2_backbone(
    args,
    resume_ckpt_dir: Path | None,
    *,
    device: torch.device,
) -> LoadedBackbone:
    _llm_tune, vision_tune = resolve_tune_modes(args)
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=args.latent_token_count,
    )
    processor = processor_bundle.processor
    token_id_map = processor_bundle.token_id_map
    added_count = processor_bundle.added_special_token_count
    pair_parallel, load_kwargs = _load_kwargs(device)
    resume_state_path = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    resume_adapter = resume_ckpt_dir / "adapter_config.json" if resume_ckpt_dir else None
    base_model_path = args.model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **load_kwargs,
    )
    _configure_shape(model, processor, args, token_id_map, added_count)

    if args.resume and resume_state_path is not None and resume_state_path.exists() and resume_adapter.is_file():
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires a LoRA tune mode")
        if saved.get("base_model_path"):
            base_model_path = Path(saved["base_model_path"])
        model = configure_qwen_tuning(model, args)
        report = load_adapter_state(model, resume_ckpt_dir)
        if is_main():
            print(json.dumps({
                "resume_lora_adapter": str(resume_ckpt_dir),
                "base_model_path": str(base_model_path),
                "missing_keys": report.missing_keys,
                "unexpected_keys": report.unexpected_keys,
                "vision_full_state_loaded": report.vision_full_state_loaded,
            }))
    elif args.resume and resume_state_path is not None and resume_state_path.exists() and (resume_ckpt_dir / "config.json").is_file():
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with LoRA tuning")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_ckpt_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
            **load_kwargs,
        )
        _configure_shape(model, processor, args, token_id_map, 0)
        model = configure_qwen_tuning(model, args)
    else:
        model = configure_qwen_tuning(model, args)

    query_adapter = None
    if args.query_tune == "adapter":
        query_token_ids = [
            token_id_map[token]
            for token in latent_state_tokens(args.latent_token_count)
        ]
        query_adapter = install_query_embedding_adapter(model, query_token_ids)
    if not pair_parallel:
        model.to(device)
    backbone = Qwen25VLBackbone(
        model,
        token_id_map=token_id_map,
        device=device,
        latent_token_count=args.latent_token_count,
        lora=uses_lora(args),
        vision_tune=vision_tune,
    )
    return LoadedBackbone(
        backbone=backbone,
        processor=processor,
        token_id_map=token_id_map,
        added_special_token_count=added_count,
        base_model_path=base_model_path,
        query_adapter=query_adapter,
        pair_parallel=pair_parallel,
    )


def load_rl_backbone(
    args,
    *,
    output_dir: Path,
    device: torch.device,
) -> LoadedBackbone:
    _llm_tune, vision_tune = resolve_tune_modes(args)
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=1,
    )
    processor = processor_bundle.processor
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
    model.resize_token_embeddings(len(processor.tokenizer))

    resume_aux_dir: Path | None = None
    if args.resume and resume_state_path.is_file() and resume_adapter.is_file():
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires a LoRA tune mode")
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if saved.get("base_model_path"):
            base_model_path = str(saved["base_model_path"])
        model = configure_qwen_tuning(model, args)
        load_adapter_state(model, resume_dir)
        resume_aux_dir = resume_dir
    elif args.resume and resume_state_path.is_file() and (resume_dir / "config.json").is_file():
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
        model.resize_token_embeddings(len(processor.tokenizer))
        model = configure_qwen_tuning(model, args)
        resume_aux_dir = resume_dir
    else:
        model = configure_qwen_tuning(model, args)

    validate_agent_policy_protocol(model.config)
    model.to(device)
    backbone = Qwen25VLBackbone(
        model,
        token_id_map=processor_bundle.token_id_map,
        device=device,
        latent_token_count=1,
        lora=uses_lora(args),
        vision_tune=vision_tune,
    )
    return LoadedBackbone(
        backbone=backbone,
        processor=processor,
        token_id_map=processor_bundle.token_id_map,
        added_special_token_count=processor_bundle.added_special_token_count,
        base_model_path=base_model_path,
        resume_aux_dir=resume_aux_dir,
    )


def build_vision_ema(
    *,
    enabled: bool,
    decay: float,
    llm: torch.nn.Module,
    resume_path: Path | None,
    device: torch.device,
):
    if not enabled:
        return None
    ema = VisionEncoderEMA(decay=decay)
    ema.reset(llm)
    if resume_path is not None and resume_path.is_file():
        loaded = VisionEncoderEMA.load_checkpoint(resume_path, map_location=device)
        ema.decay = loaded.decay
        ema.shadow = {key: value.to(device) for key, value in loaded.shadow.items()}
    return ema


def build_sft2_batch_builder(
    loaded: LoadedBackbone,
    *,
    device: torch.device,
    max_length: int,
    latent_token_count: int,
    mask_latent_query_labels: bool,
) -> Qwen25VLBatchBuilder:
    """为 SFT2 构造 Qwen processor batch adapter。"""

    return Qwen25VLBatchBuilder(
        processor=loaded.processor,
        device=device,
        max_length=max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )


def build_rl_adapters(
    loaded: LoadedBackbone,
    *,
    model: torch.nn.Module,
    device: torch.device,
    temperature: float,
    top_p: float,
) -> RLBackboneAdapters:
    """为 RL 分别构造在线 policy、rollout encoder 和 PPO replay。"""

    policy = QwenAgentPolicy(
        model=model,
        processor=loaded.processor,
        device=device,
        temperature=temperature,
        top_p=top_p,
        latent_token_count=1,
        token_id_map=loaded.token_id_map,
    )
    return RLBackboneAdapters(
        policy=policy,
        rollout_encoder=QwenRolloutEncoder(
            model=model,
            processor=loaded.processor,
            token_id_map=loaded.token_id_map,
            device=device,
        ),
        policy_replay=QwenActionLogProbReplay(
            model=model,
            processor=loaded.processor,
            token_id_map=loaded.token_id_map,
            device=device,
        ),
    )


__all__ = [
    "build_vision_ema",
    "build_rl_adapters",
    "build_sft2_batch_builder",
    "load_rl_backbone",
    "load_sft2_backbone",
]
