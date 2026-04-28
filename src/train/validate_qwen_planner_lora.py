"""Validate Qwen planner LoRA JSON format, action prior, and latent marker extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import (
    extract_latent_marker_hidden_state,
    generate_planner_response,
    load_jsonl,
    parse_planner_json,
    validate_planner_output,
)


def _topk_contains(probs: list[float], target: int, k: int) -> bool:
    ranked = sorted(range(len(probs)), key=lambda idx: float(probs[idx]), reverse=True)
    return int(target) in ranked[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", default="datasets/EB-Nav/phase2_qwen_planner_sft.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--extract-latent", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.sft_jsonl, limit=int(args.limit))
    records = [item for item in records if Path(str(item.get("image", ""))).exists()]
    if not records:
        raise RuntimeError(f"No valid validation records found in {args.sft_jsonl}")

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    if args.adapter_path:
        adapter.load_lora_adapter(args.adapter_path, trainable=False)
    model = adapter._model
    processor = adapter._processor
    model.eval()

    parse_ok = 0
    top1 = 0
    top3 = 0
    latent_ok = 0
    failures: list[dict[str, str]] = []
    for item in records:
        text = generate_planner_response(
            model=model,
            processor=processor,
            image_path=str(item["image"]),
            prompt=str(item["prompt"]),
            max_new_tokens=int(args.max_new_tokens),
        )
        try:
            obj = parse_planner_json(text)
            valid, reason = validate_planner_output(obj)
            if not valid:
                raise ValueError(reason)
            parse_ok += 1
            probs = [float(x) for x in obj["action_prior"]["probabilities"]]
            action_id = int(item["action_id"])
            top1 += int(_topk_contains(probs, action_id, 1))
            top3 += int(_topk_contains(probs, action_id, 3))
            if args.extract_latent:
                hidden = extract_latent_marker_hidden_state(
                    model=model,
                    processor=processor,
                    image_path=str(item["image"]),
                    prompt=str(item["prompt"]),
                    response=json.dumps(obj, ensure_ascii=False),
                )
                latent_ok += int(torch.isfinite(hidden).all().item())
        except Exception as exc:
            failures.append({"id": str(item.get("id", "")), "error": str(exc), "text": text[:300]})

    total = len(records)
    metrics = {
        "num_records": total,
        "parse_rate": parse_ok / max(1, total),
        "action_prior_top1": top1 / max(1, parse_ok),
        "action_prior_top3": top3 / max(1, parse_ok),
        "latent_extract_rate": latent_ok / max(1, parse_ok) if args.extract_latent else None,
        "num_failures": len(failures),
        "failures": failures[:5],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
