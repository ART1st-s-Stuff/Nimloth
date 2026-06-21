#!/usr/bin/env python3
"""Export an SFT2 LoRA checkpoint, optionally with full-vision state, as a complete HF model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _find_visual(module: torch.nn.Module) -> torch.nn.Module:
    for path in ("base_model.model.model.visual", "base_model.model.visual", "model.visual", "visual"):
        cur = module
        for name in path.split("."):
            cur = getattr(cur, name, None)
            if cur is None:
                break
        if isinstance(cur, torch.nn.Module):
            return cur
    raise RuntimeError(f"Could not locate visual module in {type(module)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="SFT2 checkpoint dir containing adapter + training_state.pt")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output full HF model dir")
    ap.add_argument("--base-model", type=Path, default=None, help="Override base model path; default from training_state.pt")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--require-vision-full-state", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    state_path = args.checkpoint / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    base_model = args.base_model or Path(state["base_model_path"])
    vision_state = args.checkpoint / "vision_full_state.pt"
    if args.require_vision_full_state and not vision_state.is_file():
        raise FileNotFoundError(f"missing vision full state: {vision_state}")

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.vocab_size = len(processor.tokenizer)

    try:
        import peft.tuners.lora.model as peft_lora_model

        def _dispatch_torchao_disabled(*args, **kwargs):
            return None

        peft_lora_model.dispatch_torchao = _dispatch_torchao_disabled
    except Exception:
        pass

    peft_model = PeftModel.from_pretrained(model, args.checkpoint)
    loaded_vision_full = False
    if vision_state.is_file():
        visual = _find_visual(peft_model)
        visual.load_state_dict(torch.load(vision_state, map_location="cpu", weights_only=True))
        loaded_vision_full = True
    merged = peft_model.merge_and_unload()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    processor.save_pretrained(args.out_dir)
    with (args.out_dir / "sft2_export_metadata.json").open("w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "base_model": str(base_model),
                "vision_full_state": str(vision_state) if vision_state.is_file() else None,
                "loaded_vision_full": loaded_vision_full,
                "source_step": state.get("step"),
                "source_epoch": state.get("epoch"),
                "llm_tune": state.get("llm_tune"),
                "vision_tune": state.get("vision_tune"),
            },
            f,
            indent=2,
        )
    print(json.dumps({"exported": str(args.out_dir), "source": str(args.checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
