"""Export a train_qwen_planner_lora training_state checkpoint to a PEFT adapter dir."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.vlm.qwen_adapter import QwenVLMAdapter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = p.parse_args()

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
        device_map="auto",
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    adapter.enable_language_lora(
        r=int(args.lora_r),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=list(args.target_modules),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = adapter._model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("unexpected_sample", list(unexpected)[:20])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter._model.save_pretrained(out)
    adapter._processor.save_pretrained(out)
    print(f"saved_lora_adapter={out}")


if __name__ == "__main__":
    main()
