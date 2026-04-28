"""LoRA SFT for Qwen planner JSON/action-prior output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import load_jsonl


class PlannerSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, limit: int = 0) -> None:
        records = load_jsonl(jsonl_path, limit=limit)
        self.records = [item for item in records if Path(str(item.get("image", ""))).exists()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def _build_collate_fn(processor: Any):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [Image.open(str(item["image"])).convert("RGB") for item in batch]
        prompt_texts = []
        full_texts = []
        for item in batch:
            prompt = str(item["prompt"])
            response = str(item["response"])
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
            prompt_texts.append(
                processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            )

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        for row_idx, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            labels[row_idx, :prompt_len] = -100
        if pad_id is not None:
            labels[inputs["input_ids"] == pad_id] = -100
        inputs["labels"] = labels
        return inputs

    return _collate


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", default="datasets/EB-Nav/phase2_qwen_planner_sft.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="models/qwen_planner_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    trainable = adapter.enable_language_lora(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
    )
    model = adapter._model
    processor = adapter._processor
    model.train()
    print(f"language_lora_trainable_params={trainable}")

    dataset = PlannerSFTDataset(args.sft_jsonl, limit=args.limit)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid SFT records found in {args.sft_jsonl}")
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=_build_collate_fn(processor),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    device = next(model.parameters()).device
    accum = max(1, int(args.gradient_accumulation_steps))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(max(1, int(args.epochs))):
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / float(accum)
            loss.backward()
            if (batch_idx + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if global_step % 10 == 0:
                print(f"epoch={epoch + 1} step={global_step} loss={float(loss.item() * accum):.6f}")
            global_step += 1
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved_lora_adapter={output_dir}")


if __name__ == "__main__":
    main()
