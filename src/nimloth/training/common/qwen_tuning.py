"""Configure Qwen2.5-VL LLM / vision tuning modes for training."""

from __future__ import annotations

import argparse
from typing import Literal

from transformers import Qwen2_5_VLForConditionalGeneration

TuneMode = Literal["freeze", "lora", "full"]

LLM_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LORA_TARGETS = ("qkv", "proj", "linear_fc1", "linear_fc2")


def _is_vision_param(name: str) -> bool:
    return ".visual." in name or name.startswith("visual.")


def is_vision_param(name: str) -> bool:
    return _is_vision_param(name)


def _is_llm_param(name: str) -> bool:
    return ".language_model." in name or name.startswith("language_model.")


def resolve_tune_modes(args: argparse.Namespace) -> tuple[TuneMode, TuneMode]:
    if getattr(args, "lora", False):
        return "lora", "freeze"
    return getattr(args, "llm_tune", "freeze"), getattr(args, "vision_tune", "freeze")


def _lora_target_modules(llm_tune: TuneMode, vision_tune: TuneMode) -> list[str]:
    targets: list[str] = []
    if llm_tune == "lora":
        targets.extend(LLM_LORA_TARGETS)
    if vision_tune == "lora":
        targets.extend(VISION_LORA_TARGETS)
    return targets


def _set_requires_grad(module, predicate, enabled: bool) -> None:
    for name, param in module.named_parameters():
        if predicate(name):
            param.requires_grad = enabled


def configure_qwen_tuning(
    model: Qwen2_5_VLForConditionalGeneration,
    args: argparse.Namespace,
) -> Qwen2_5_VLForConditionalGeneration:
    """Apply per-submodule freeze / LoRA / full fine-tune."""

    llm_tune, vision_tune = resolve_tune_modes(args)
    for param in model.parameters():
        param.requires_grad = False

    uses_lora = llm_tune == "lora" or vision_tune == "lora"
    if uses_lora:
        from peft import LoraConfig, get_peft_model

        modules_to_save: list[str] = []
        if llm_tune == "lora":
            modules_to_save.extend(["embed_tokens", "lm_head"])
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_lora_target_modules(llm_tune, vision_tune),
            modules_to_save=modules_to_save or None,
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

    if llm_tune == "full":
        _set_requires_grad(model, _is_llm_param, True)
    if vision_tune == "full":
        _set_requires_grad(model, _is_vision_param, True)

    return model
