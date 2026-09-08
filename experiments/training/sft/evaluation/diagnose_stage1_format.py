"""只读诊断 SFT1 格式生成；产物仅包含结构、类别和概率，不保存私有文本。"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from pathlib import Path


def classify(text: str, *, latent_token_count: int = 1) -> dict:
    from nimloth.latent import latent_state_tokens
    from nimloth.training.sft.stage1.trainer import (
        action_block_format_correct,
        nimloth_format_correct,
    )

    markers = [
        "<think>",
        "</think>",
        *latent_state_tokens(latent_token_count),
        "<|action_start|>",
        "<|action_end|>",
    ]
    positions = [text.find(marker) for marker in markers]
    return {
        "marker_counts": [text.count(marker) for marker in markers],
        "all_markers_ordered": all(p >= 0 for p in positions)
        and positions == sorted(positions),
        "action_token_count": len(re.findall(r"<\|action_\(\d+\)\|>", text)),
        "full_format_pass": nimloth_format_correct(
            text, latent_token_count=latent_token_count
        ),
        "action_format_pass": action_block_format_correct(text),
    }


def token_category(token_id: int, tokenizer, latent_token_count: int) -> str:
    from nimloth.latent import latent_state_tokens

    token = tokenizer.convert_ids_to_tokens(token_id)
    if token in ("<think>", "</think>"):
        return "thought_marker"
    if token in latent_state_tokens(latent_token_count):
        return "latent"
    if token in ("<|action_start|>", "<|action_end|>"):
        return "action_marker"
    if re.fullmatch(r"<\|action_\(\d+\)\|>", token or ""):
        return "action"
    if token_id in tokenizer.all_special_ids:
        return "other_special"
    return "text"


def termination(ids: list[int], eos_ids, limit: int) -> dict:
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return {
        "generated_tokens": len(ids),
        "ended_with_eos": bool(ids) and ids[-1] in (eos_ids or []),
        "reached_token_limit": len(ids) >= limit,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "val-jsonl", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in (
        ("max-samples", 32),
        ("max-new-tokens", 128),
        ("max-length", 12000),
        ("max-pixels", 100352),
        ("min-pixels", 3136),
        ("latent-token-count", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    for name in (
        "max_samples",
        "max_new_tokens",
        "max_length",
        "max_pixels",
        "min_pixels",
        "latent_token_count",
    ):
        if getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    return args


def main() -> int:
    args = parse_args()
    # 延迟导入使 --help 无需 GPU 或 ML 依赖。
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.latent import (
        add_special_tokens,
        latent_state_block,
        normalize_latent_state_blocks,
        special_token_ids,
    )
    from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
    from nimloth.training.sft.stage1.data import (
        NimlothVLSFTDataset,
        collate_fn,
        collect_images,
    )
    from nimloth.training.sft.stage1.trainer import (
        prepare_query_vocabulary,
        prompt_messages_before_first_assistant,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("diagnosis requires an allocated remote GPU")
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "hostname": socket.gethostname(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            },
            default=str,
            indent=2,
        )
    )
    torch.manual_seed(42)
    print(json.dumps({"phase": "load_processor"}), flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    added = add_special_tokens(
        processor.tokenizer, latent_token_count=args.latent_token_count
    )
    saved_processor = AutoProcessor.from_pretrained(
        args.adapter, trust_remote_code=True
    )
    if processor.tokenizer.get_vocab() != saved_processor.tokenizer.get_vocab():
        raise ValueError(
            "base plus special token vocabulary differs from saved adapter processor"
        )
    print(json.dumps({"phase": "load_base_model"}), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    prepare_query_vocabulary(
        model,
        len(processor.tokenizer),
        special_token_ids(
            processor.tokenizer, latent_token_count=args.latent_token_count
        ),
        added_tokens=added,
        latent_token_count=args.latent_token_count,
    )
    model = get_peft_model(model, LoraConfig.from_pretrained(args.adapter))
    print(json.dumps({"phase": "load_and_verify_adapter"}), flush=True)
    load_lora_adapter_state(model, args.adapter)
    model.requires_grad_(False).to("cuda").eval()
    print(json.dumps({"phase": "model_ready"}), flush=True)
    dataset = NimlothVLSFTDataset(
        args.val_jsonl, processor, max_records=args.max_samples
    )
    counts = {
        "samples": 0,
        "free_full_pass": 0,
        "conditional_action_pass": 0,
        "conditional_samples": 0,
        "teacher_correct": 0,
        "teacher_tokens": 0,
    }

    def generate(text, images):
        inputs = processor(
            text=[normalize_latent_state_blocks(text, args.latent_token_count)],
            images=images or None,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        ids = output[0, inputs["input_ids"].shape[1] :].tolist()
        result = classify(
            processor.decode(ids, skip_special_tokens=False),
            latent_token_count=args.latent_token_count,
        )
        result.update(
            termination(ids, model.generation_config.eos_token_id, args.max_new_tokens)
        )
        return result

    with (
        (args.output_dir / "samples.jsonl").open("x") as stream,
        torch.inference_mode(),
    ):
        for index in range(len(dataset)):
            print(json.dumps({"phase": "sample_start", "index": index}), flush=True)
            messages = dataset.get_messages(index)
            prompt = prompt_messages_before_first_assistant(messages)
            reference = next(m["content"] for m in messages if m["role"] == "assistant")
            reference = normalize_latent_state_blocks(
                str(reference), args.latent_token_count
            )
            images = collect_images(prompt)
            text = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            row = {
                "index": index,
                "reference": classify(
                    reference, latent_token_count=args.latent_token_count
                ),
                "reference_tokens": len(
                    processor.tokenizer.encode(reference, add_special_tokens=False)
                ),
            }
            # 每个阶段分别落盘；后续失败仍保留已经完成的生成证据。
            row["free"] = generate(text, images)
            stream.write(json.dumps({"phase": "free", **row}) + "\n")
            stream.flush()
            thought = re.search(r"<think>.*?</think>", reference, re.DOTALL)
            row["conditional"] = None
            if thought:
                row["conditional"] = generate(
                    text
                    + thought.group(0)
                    + latent_state_block(args.latent_token_count),
                    images,
                )
                stream.write(
                    json.dumps(
                        {
                            "phase": "conditional",
                            "index": index,
                            "result": row["conditional"],
                        }
                    )
                    + "\n"
                )
                stream.flush()
            batch = collate_fn(
                [
                    {
                        "messages": prompt
                        + [next(m for m in messages if m["role"] == "assistant")]
                    }
                ],
                processor,
                args.max_length,
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=False,
                require_complete=True,
            )
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels")[0]
            logits = model(**batch).logits[0]
            positions = (labels[1:] != -100).nonzero().flatten() + 1
            tokens = []
            for offset, position in enumerate(positions.tolist()):
                target = int(labels[position])
                scores = logits[position - 1].float()
                tokens.append(
                    {
                        "position": offset,
                        "category": token_category(
                            target, processor.tokenizer, args.latent_token_count
                        ),
                        "target_probability": float(
                            torch.exp(scores[target] - torch.logsumexp(scores, dim=0))
                        ),
                        "top1_correct": int(scores.argmax()) == target,
                    }
                )
            del logits, batch
            row["teacher_scope"] = "first_assistant_only_matching_free_generation"
            row["teacher_forced_tokens"] = tokens
            stream.write(json.dumps({"phase": "complete", **row}) + "\n")
            stream.flush()
            counts["samples"] += 1
            counts["free_full_pass"] += int(row["free"]["full_format_pass"])
            if row["conditional"] is not None:
                counts["conditional_samples"] += 1
                counts["conditional_action_pass"] += int(
                    row["conditional"]["action_format_pass"]
                )
            counts["teacher_tokens"] += len(tokens)
            counts["teacher_correct"] += sum(t["top1_correct"] for t in tokens)
            (args.output_dir / "summary.json").write_text(json.dumps(counts, indent=2))
            print(json.dumps({"completed_samples": counts["samples"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
