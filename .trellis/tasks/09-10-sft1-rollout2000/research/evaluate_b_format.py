"""Free-generation B evaluation using an explicit base, dataset and optional adapter."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
from nimloth.training.sft.stage1.data import (
    NimlothVLSFTDataset,
    collect_images,
    render_stage_text,
)
from nimloth.training.sft.stage1.trainer import (
    apply_lora,
    nimloth_format_correct,
    prompt_messages_before_first_assistant,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not (args.base_model / "INITIALIZATION_COMPLETE").is_file():
        raise ValueError("B base is not verified")
    init_hash = hashlib.sha256(
        (args.base_model / "initialization_manifest.json").read_bytes()
    ).hexdigest()
    if (
        json.loads((args.base_model / "INITIALIZATION_COMPLETE").read_text())[
            "manifest_sha256"
        ]
        != init_hash
    ):
        raise ValueError("B initialization manifest hash mismatch")
    committed = None
    if args.checkpoint:
        committed = json.loads((args.checkpoint / "COMMITTED").read_text())
        state = torch.load(
            args.checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        identity = state["identity"]
        expected = {
            "stage": "format",
            "action_token_loss_weight": 8,
            "model": str(args.base_model.resolve()),
            "val_jsonl": str(args.data_jsonl.resolve()),
            "val_jsonl_sha256": hashlib.sha256(
                args.data_jsonl.read_bytes()
            ).hexdigest(),
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
        }
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("B checkpoint objective/base/data identity mismatch")
        if (
            state["step"] != 20
            or committed.get("step") != 20
            or state["base_model_path"] != str(args.base_model)
        ):
            raise ValueError("B checkpoint step/base mismatch")
        adapter = json.loads((args.checkpoint / "adapter_config.json").read_text())
        targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if (
            adapter["r"] != 64
            or adapter["lora_alpha"] != 128
            or adapter["lora_dropout"] != 0.05
            or set(adapter["target_modules"]) != targets
            or set(adapter["modules_to_save"]) != {"embed_tokens", "lm_head"}
        ):
            raise ValueError("B adapter architecture mismatch")
        del state
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(args.checkpoint or args.base_model)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )
    if args.checkpoint:
        model = apply_lora(
            model,
            argparse.Namespace(
                lora_r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                gradient_checkpointing=False,
            ),
        )
        load_lora_adapter_state(model, args.checkpoint)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    dataset = NimlothVLSFTDataset(args.data_jsonl, processor, max_records=args.samples)
    metadata = {
        "base": str(args.base_model),
        "checkpoint": str(args.checkpoint),
        "committed": committed,
        "data": str(args.data_jsonl),
        "data_sha256": hashlib.sha256(args.data_jsonl.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "samples": len(dataset),
        "max_new_tokens": 128,
        "do_sample": False,
        "device": "cuda:0",
        "precision": "BF16/FA2",
        "started_unix": started,
        "scope": "first heldout prompts; no environment execution or constrained decoding",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    strict = re.compile(
        r"<think><observation>[^<>]*</observation><reasoning>[^<>]*</reasoning><prediction>[^<>]*</prediction></think><\|action_start\|><\|action_\([0-7]\)\|><\|action_end\|>",
        re.DOTALL,
    )
    results = []
    with (args.output_dir / "responses.jsonl").open("x") as stream:
        for index in range(len(dataset)):
            messages = dataset.get_messages(index)
            prompt = prompt_messages_before_first_assistant(messages)
            text = render_stage_text(
                processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                ),
                None,
            )
            images = collect_images(prompt)
            inputs = processor(
                text=[text], images=images or None, return_tensors="pt"
            ).to("cuda:0")

            def generate(limit, inputs=inputs):
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=limit, do_sample=False
                    )
                ids = output[0, inputs["input_ids"].shape[1] :].tolist()
                raw = processor.decode(ids, skip_special_tokens=False)
                content_ids = ids[:-1] if ids and ids[-1] in eos_ids else ids
                clean = processor.decode(content_ids, skip_special_tokens=False)
                return {
                    "token_ids": ids,
                    "decoded_raw": raw,
                    "token_count": len(ids),
                    "ended_eos": bool(ids and ids[-1] in eos_ids),
                    "reached_limit": len(ids) >= limit,
                    "legacy_format_correct": nimloth_format_correct(
                        raw, latent_token_count=None
                    ),
                    "strict_format_correct": bool(strict.fullmatch(clean)),
                }

            row = {
                "index": index,
                "id": dataset.records[index]["id"],
                "prompt_text": text,
                "reference": render_stage_text(
                    next(m["content"] for m in messages if m["role"] == "assistant"),
                    None,
                ),
                "primary": generate(128),
            }
            if row["primary"]["reached_limit"] and not row["primary"]["ended_eos"]:
                row["long_budget_diagnostic"] = generate(512)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(row)
            print(
                json.dumps(
                    dict(
                        index=index,
                        **{
                            k: v
                            for k, v in row["primary"].items()
                            if k not in ("token_ids", "decoded_raw")
                        },
                    )
                ),
                flush=True,
            )
    summary = {
        "samples": len(results),
        "legacy_correct": sum(r["primary"]["legacy_format_correct"] for r in results),
        "strict_correct": sum(r["primary"]["strict_format_correct"] for r in results),
        "length_limit_count": sum(r["primary"]["reached_limit"] for r in results),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "COMPLETED.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
