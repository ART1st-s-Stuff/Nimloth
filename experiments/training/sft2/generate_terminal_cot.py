#!/usr/bin/env python3
"""Add persisted SFT1-generated terminal CoT state prefixes to SFT2 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.rollout.transitions import TERMINAL_ASSISTANT_PREFIX_FIELD
from nimloth.training.sft2.data.terminal_cot import (
    TerminalCoTFormatError,
    generate_terminal_cot_prefix,
    write_augmented_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if TERMINAL_ASSISTANT_PREFIX_FIELD in record:
                raise ValueError(
                    f"{path}:{line_number} already contains "
                    f"{TERMINAL_ASSISTANT_PREFIX_FIELD!r}"
                )
            yield record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--max-reasoning-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--format-failure-policy",
        choices=("fail", "exclude"),
        default="fail",
        help=(
            "fail preserves strict generation; exclude writes an auditable "
            "sidecar and omits only TerminalCoTFormatError records"
        ),
    )
    args = parser.parse_args()

    if args.max_reasoning_tokens < 1:
        raise ValueError("--max-reasoning-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    sampling_config = (
        args.temperature,
        args.top_p,
        args.top_k,
        args.do_sample,
        args.n,
    )
    if sampling_config != (0.0, 1.0, -1, False, 1):
        raise ValueError(
            "terminal CoT sampling must match VAGEN validation: "
            "temperature=0, top_p=1, top_k=-1, do_sample=false, n=1"
        )
    if args.input_jsonl.resolve() == args.output_jsonl.resolve():
        raise ValueError("input and output JSONL must be different files")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    latent_token_count = validate_agent_policy_protocol(model_config)
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=latent_token_count,
    )
    if processor_bundle.added_special_token_count:
        raise ValueError(
            "SFT1 initialization checkpoint tokenizer is missing Nimloth tokens"
        )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device)
    if model.get_input_embeddings().num_embeddings != len(
        processor_bundle.processor.tokenizer
    ):
        raise ValueError("SFT1 checkpoint model/tokenizer vocabulary mismatch")
    model.eval()

    reasoning_token_counts: list[int] = []
    excluded_records: list[dict[str, Any]] = []

    def augmented_records() -> Iterator[dict[str, Any]]:
        for record_index, record in enumerate(_records(args.input_jsonl), start=1):
            try:
                generated = generate_terminal_cot_prefix(
                    record=record,
                    model=model,
                    processor=processor_bundle.processor,
                    token_id_map=processor_bundle.token_id_map,
                    device=device,
                    latent_token_count=latent_token_count,
                    max_reasoning_tokens=args.max_reasoning_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    n=args.n,
                )
            except TerminalCoTFormatError as error:
                if args.format_failure_policy == "fail":
                    raise
                exclusion = {
                    "input_record_index": record_index,
                    "id": str(record.get("id", "")),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
                excluded_records.append(exclusion)
                print(
                    json.dumps({"excluded_terminal_cot_record": exclusion}),
                    flush=True,
                )
                continue
            reasoning_token_counts.append(generated.reasoning_token_count)
            yield {**record, TERMINAL_ASSISTANT_PREFIX_FIELD: generated.prefix}
            if record_index % 10 == 0:
                print(
                    json.dumps(
                        {
                            "processed_records": record_index,
                            "generated_records": len(reasoning_token_counts),
                            "excluded_records": len(excluded_records),
                        }
                    ),
                    flush=True,
                )

    record_count = write_augmented_records(args.output_jsonl, augmented_records())
    input_record_count = record_count + len(excluded_records)
    excluded_path = args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".excluded.jsonl"
    )
    write_augmented_records(excluded_path, excluded_records)
    manifest = {
        "format": "nimloth_terminal_cot_v2",
        "model": str(args.model.resolve()),
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": _sha256(args.input_jsonl),
        "input_record_count": input_record_count,
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": _sha256(args.output_jsonl),
        "record_count": record_count,
        "format_failure_policy": args.format_failure_policy,
        "excluded_records_jsonl": str(excluded_path.resolve()),
        "excluded_records_sha256": _sha256(excluded_path),
        "excluded_record_count": len(excluded_records),
        "latent_token_count": latent_token_count,
        "max_pixels": args.max_pixels,
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "n": args.n,
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "reasoning_tokens": {
            "min": min(reasoning_token_counts, default=0),
            "max": max(reasoning_token_counts, default=0),
            "mean": (
                sum(reasoning_token_counts) / len(reasoning_token_counts)
                if reasoning_token_counts
                else 0.0
            ),
        },
    }
    manifest_path = args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".manifest.json"
    )
    write_augmented_records(manifest_path, [manifest])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
