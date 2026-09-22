"""Qwen2.5-VL backbone: LLM / vision tuning modes."""

from __future__ import annotations

import argparse
import json
import os
from typing import Literal

from transformers import Qwen2_5_VLForConditionalGeneration

TuneMode = Literal["freeze", "lora", "full"]

LLM_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LORA_TARGETS = ("qkv", "proj", "linear_fc1", "linear_fc2")


def _is_vision_param(name: str) -> bool:
    return ".visual." in name or name.startswith("visual.")


def is_vision_param(name: str) -> bool:
    return _is_vision_param(name)


def _parameter_regions(model):
    """Resolve ownership from actual modules before PEFT adds wrapper prefixes."""
    decoder = model.get_decoder()
    visual = getattr(model, "visual", None)
    if visual is None:
        visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise ValueError("Qwen tuning requires an identifiable visual module")
    vision_ids = {id(p) for p in visual.parameters()}
    language_ids = {id(p) for p in decoder.parameters()} - vision_ids
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            language_ids.update(id(p) for p in module.parameters())
    vocab_ids = {id(p) for module in
                 (model.get_input_embeddings(), model.get_output_embeddings())
                 if module is not None for p in module.parameters()}
    dense_language_ids = language_ids - vocab_ids
    if not dense_language_ids:
        raise ValueError("Qwen tuning found no dense language parameters in get_decoder()")
    unknown = {id(p) for p in model.parameters()} - language_ids - vision_ids
    if unknown:
        raise ValueError("Qwen tuning found parameters outside language and vision modules")
    return language_ids, vision_ids, dense_language_ids


def resolve_tune_modes(args: argparse.Namespace) -> tuple[TuneMode, TuneMode]:
    if getattr(args, "lora", False):
        return "lora", "freeze"
    return getattr(args, "llm_tune", "freeze"), getattr(args, "vision_tune", "freeze")


def uses_lora(args: argparse.Namespace) -> bool:
    llm_tune, vision_tune = resolve_tune_modes(args)
    return llm_tune == "lora" or vision_tune == "lora"


def configure_qwen_tuning(
    model: Qwen2_5_VLForConditionalGeneration,
    args: argparse.Namespace,
) -> Qwen2_5_VLForConditionalGeneration:
    """Apply per-submodule freeze / LoRA / full fine-tune."""

    llm_tune, vision_tune = resolve_tune_modes(args)
    language_ids, vision_ids, dense_language_ids = _parameter_regions(model)
    for param in model.parameters():
        param.requires_grad = False

    uses_lora_flag = llm_tune == "lora" or vision_tune == "lora"
    if uses_lora_flag:
        from peft import LoraConfig, get_peft_model

        try:
            import peft.tuners.lora.model as peft_lora_model

            def _dispatch_torchao_disabled(*args, **kwargs):
                return None

            peft_lora_model.dispatch_torchao = _dispatch_torchao_disabled
        except Exception:
            pass

        # FSDP+LoRA 与 modules_to_save 的包装层会发生冲突，因此保持为空。
        modules_to_save: list[str] = []
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                (list(LLM_LORA_TARGETS) if llm_tune == "lora" else [])
                + (list(VISION_LORA_TARGETS) if vision_tune == "lora" else [])
            ),
            modules_to_save=modules_to_save or None,
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

    if llm_tune == "full":
        for param in model.parameters():
            if id(param) in language_ids:
                param.requires_grad = True
    if vision_tune == "full":
        for param in model.parameters():
            if id(param) in vision_ids:
                param.requires_grad = True

    if llm_tune == "full" and not any(
        p.requires_grad and id(p) in dense_language_ids for p in model.parameters()
    ):
        raise ValueError("Full language tuning has no trainable dense language parameters")
    if int(os.environ.get("RANK", "0")) == 0:
        counts = {
            "language_dense": sum(p.numel() for p in model.parameters()
                                  if id(p) in dense_language_ids and p.requires_grad),
            "language_total": sum(p.numel() for p in model.parameters()
                                  if id(p) in language_ids and p.requires_grad),
            "vision": sum(p.numel() for p in model.parameters()
                          if id(p) in vision_ids and p.requires_grad),
        }
        print(json.dumps({"qwen_tuning_trainable_parameters": counts,
                          "llm_tune": llm_tune, "vision_tune": vision_tune}), flush=True)
    return model
