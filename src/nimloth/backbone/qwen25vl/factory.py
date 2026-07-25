"""Qwen2.5-VL 模型与独立能力适配器的构造。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.base import LoadedBackbone
from nimloth.backbone.qwen25vl.checkpoint import load_adapter_state
from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.backbone.qwen25vl.policy import (
    QwenActionLogProbReplay,
    QwenAgentPolicy,
    validate_agent_policy_protocol,
)
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


def _load_kwargs(
    device: torch.device,
    *,
    model_parallel_size: int | None,
) -> tuple[bool, dict[str, Any]]:
    gpu_stride = (
        int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
        if model_parallel_size is None
        else int(model_parallel_size)
    )
    if gpu_stride < 1:
        raise ValueError(f"model_parallel_size must be positive, got {gpu_stride}")
    enabled = gpu_stride > 1 and torch.cuda.is_available()
    if not enabled:
        return False, {}
    primary_idx = int(str(device).split(":")[-1])
    pair = [primary_idx + offset for offset in range(gpu_stride)]
    if is_main():
        print(
            json.dumps(
                {
                    "qwen_pair_parallel": True,
                    "gpu_stride": gpu_stride,
                    "rank0_pair": pair,
                }
            )
        )
    return True, {
        # ``auto`` 会把能装进单卡的 3B 模型全部放在第一张卡，无法分摊 PPO
        # replay activation。``balanced`` 才表达每个训练副本跨卡的真实语义。
        "device_map": "balanced",
        "max_memory": {index: "74GiB" for index in pair} | {"cpu": "64GiB"},
        "low_cpu_mem_usage": True,
    }


def _cuda_device_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.device):
        return value.index if value.type == "cuda" else None
    if isinstance(value, str) and value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    return None


def _validate_model_parallel_placement(
    model: torch.nn.Module,
    *,
    input_device: torch.device,
    model_parallel_size: int,
) -> None:
    if model_parallel_size <= 1:
        return
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError("multi-GPU Qwen load did not produce hf_device_map")
    primary_idx = int(str(input_device).split(":")[-1])
    expected = set(range(primary_idx, primary_idx + model_parallel_size))
    actual = {
        index
        for value in device_map.values()
        if (index := _cuda_device_index(value)) is not None
    }
    offloaded = {
        str(value)
        for value in device_map.values()
        if _cuda_device_index(value) is None
    }
    if actual != expected or offloaded:
        raise RuntimeError(
            "Qwen model-parallel placement does not match the configured local "
            f"GPU group: expected={sorted(expected)}, actual={sorted(actual)}, "
            f"offloaded={sorted(offloaded)}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "qwen_model_parallel_placement": "validated",
                    "devices": sorted(actual),
                }
            )
        )


def model_output_device(
    model: torch.nn.Module,
    *,
    default: torch.device,
) -> torch.device:
    """返回 Qwen final norm/lm_head 所在设备，供下游 WM/value 对齐。"""

    root = getattr(model, "module", model)
    device_map = getattr(root, "hf_device_map", {}) or {}
    mapped = device_map.get("lm_head")
    if mapped is None:
        mapped = device_map.get("model.language_model.norm")
    index = _cuda_device_index(mapped)
    return torch.device(f"cuda:{index}") if index is not None else default


def _configure_shape(
    model: torch.nn.Module,
    processor: Any,
    args: Any,
    token_id_map: dict[str, int],
    added_count: int,
    *,
    latent_token_count: int,
) -> None:
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.resize_token_embeddings(len(processor.tokenizer))
    if added_count > 0:
        initialize_extra_latent_token_embeddings(
            model,
            token_id_map,
            latent_token_count=latent_token_count,
        )
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.vocab_size = len(processor.tokenizer)


def _resume_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def load_backbone(
    args: Any,
    *,
    device: torch.device,
    latent_token_count: int,
    model_parallel_size: int | None = None,
    resume_dir: Path | None = None,
    resume_state_path: Path | None = None,
) -> LoadedBackbone:
    """加载一个 Qwen Backbone；调用方只提供 artifact 位置，不提供阶段名称。"""

    _llm_tune, vision_tune = resolve_tune_modes(args)
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=latent_token_count,
    )
    processor = processor_bundle.processor
    token_id_map = processor_bundle.token_id_map
    added_count = processor_bundle.added_special_token_count
    pair_parallel, load_kwargs = _load_kwargs(
        device,
        model_parallel_size=model_parallel_size,
    )
    resolved_model_parallel_size = (
        int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
        if model_parallel_size is None
        else int(model_parallel_size)
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model_path: Path | str = args.model
    resume_applied = False

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **load_kwargs,
    )
    _configure_shape(
        model,
        processor,
        args,
        token_id_map,
        added_count,
        latent_token_count=latent_token_count,
    )

    saved = _resume_metadata(resume_state_path)
    if saved.get("base_model_path"):
        base_model_path = saved["base_model_path"]
    can_resume = bool(getattr(args, "resume", False) and saved and resume_dir)
    if can_resume and (resume_dir / "adapter_config.json").is_file():
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires a LoRA tune mode")
        model = configure_qwen_tuning(model, args)
        report = load_adapter_state(model, resume_dir)
        resume_applied = True
        if is_main():
            print(
                json.dumps(
                    {
                        "resume_lora_adapter": str(resume_dir),
                        "base_model_path": str(base_model_path),
                        "missing_keys": report.missing_keys,
                        "unexpected_keys": report.unexpected_keys,
                        "vision_full_state_loaded": report.vision_full_state_loaded,
                    }
                )
            )
    elif can_resume and (resume_dir / "config.json").is_file():
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with LoRA tuning")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_dir,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
            **load_kwargs,
        )
        _configure_shape(
            model,
            processor,
            args,
            token_id_map,
            0,
            latent_token_count=latent_token_count,
        )
        model = configure_qwen_tuning(model, args)
        resume_applied = True
    else:
        model = configure_qwen_tuning(model, args)

    query_adapter = None
    if getattr(args, "query_tune", "freeze") == "adapter":
        query_token_ids = [
            token_id_map[token]
            for token in latent_state_tokens(latent_token_count)
        ]
        query_adapter = install_query_embedding_adapter(model, query_token_ids)
    if not pair_parallel:
        model.to(device)
    else:
        _validate_model_parallel_placement(
            model,
            input_device=device,
            model_parallel_size=resolved_model_parallel_size,
        )
    return LoadedBackbone(
        backbone=Qwen25VLBackbone(
            model,
            token_id_map=token_id_map,
            device=device,
            latent_token_count=latent_token_count,
            lora=uses_lora(args),
            vision_tune=vision_tune,
        ),
        processor=processor,
        token_id_map=token_id_map,
        added_special_token_count=added_count,
        base_model_path=base_model_path,
        query_adapter=query_adapter,
        pair_parallel=pair_parallel,
        resume_aux_dir=resume_dir if resume_applied else None,
    )


def build_input_builder(
    loaded: LoadedBackbone,
    *,
    max_length: int,
    latent_token_count: int,
    mask_latent_query_labels: bool = True,
) -> Qwen25VLInputBuilder:
    """构造训练阶段无关的 Qwen prompt 输入适配器。"""

    return Qwen25VLInputBuilder(
        processor=loaded.processor,
        max_length=max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )


def build_agent_policy(
    loaded: LoadedBackbone,
    *,
    model: torch.nn.Module,
    device: torch.device,
    temperature: float,
    top_p: float,
):
    """构造在线动作 policy；协议检查属于该能力本身。"""

    model_config = getattr(getattr(model, "module", model), "config")
    latent_token_count = validate_agent_policy_protocol(model_config)
    return QwenAgentPolicy(
        model=model,
        processor=loaded.processor,
        device=device,
        temperature=temperature,
        top_p=top_p,
        latent_token_count=latent_token_count,
        token_id_map=loaded.token_id_map,
    )


def build_action_log_prob_replay(
    loaded: LoadedBackbone,
    *,
    model: torch.nn.Module,
    device: torch.device,
    token_value_head: torch.nn.Module | None = None,
):
    """构造 PPO 当前策略概率重放器。"""

    model_config = getattr(getattr(model, "module", model), "config")
    validate_agent_policy_protocol(model_config)
    return QwenActionLogProbReplay(
        model=model,
        processor=loaded.processor,
        token_id_map=loaded.token_id_map,
        device=device,
        token_value_head=token_value_head,
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


__all__ = [
    "build_action_log_prob_replay",
    "build_agent_policy",
    "build_input_builder",
    "build_vision_ema",
    "load_backbone",
]
