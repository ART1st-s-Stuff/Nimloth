"""Mandatory static Stage 1 format gate using the production vLLM generator."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.backbone.qwen25vl.early_generation import (
    EarlyVLLMGenerator,
    RawGeneration,
    validate_stage1_raw_generation,
)
from nimloth.rollout.early_records import write_json
from nimloth.rollout.fresh import file_artifact_fingerprint, policy_artifact_fingerprint
from nimloth.training.sft.evaluation.config import EvaluationConfig

FORMAT_GATE_SIZE = 32
FORMAT_GATE_MINIMUM_CORRECT = 31


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_selected_records(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"format-gate line {line_number} lacks a string id")
            if record_id in ids:
                raise ValueError(f"duplicate format-gate record id: {record_id}")
            ids.add(record_id)
            rows.append(row)
    if len(rows) < FORMAT_GATE_SIZE:
        raise ValueError(
            f"Stage 1 format gate requires at least {FORMAT_GATE_SIZE} full-heldout "
            f"records; found {len(rows)}"
        )
    return rows[:FORMAT_GATE_SIZE]


def _prompt_and_lineage(
    row: dict[str, Any], source: Path, protocol: EarlyProtocol
) -> tuple[list[dict[str, Any]], list[Image.Image], dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise TypeError(f"record {row['id']!r} lacks messages")
    prompt: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise TypeError(f"record {row['id']!r} has invalid message")
        if message["role"] == "assistant":
            break
        if not isinstance(message.get("content"), str):
            raise TypeError(
                f"record {row['id']!r} prompt content must be an unbound string"
            )
        prompt.append(
            {
                "role": message["role"],
                "content": protocol.prompt(message["content"]),
            }
        )
    if not prompt:
        raise ValueError(f"record {row['id']!r} has no prompt before assistant")
    placeholder_count = sum(
        message["content"].count("<image>") for message in prompt
    )
    raw_paths = row.get("image_paths", [])
    if not isinstance(raw_paths, list) or any(not isinstance(path, str) for path in raw_paths):
        raise ValueError(f"record {row['id']!r} has invalid image_paths")
    if len(raw_paths) < placeholder_count:
        raise ValueError(
            f"record {row['id']!r} has {placeholder_count} prompt image placeholders "
            f"but only {len(raw_paths)} image paths"
        )
    used_paths = raw_paths[:placeholder_count]
    image_lineage: list[dict[str, str]] = []
    images: list[Image.Image] = []
    for raw_path in used_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        image_lineage.append(
            {"path": str(path), "sha256": file_artifact_fingerprint(path)}
        )
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    lineage = {
        "record_id": row["id"],
        "source_identity": row.get("source_identity"),
        "prompt_messages_sha256": _canonical_hash(prompt),
        "image_lineage": image_lineage,
    }
    return prompt, images, lineage


def run_stage1_format_gate(
    config: EvaluationConfig,
    protocol: EarlyProtocol,
    *,
    generator: Any | None = None,
    allow_generate: bool = True,
) -> tuple[bool, Any | None]:
    """Evaluate the fixed first 32 heldout records and publish exact evidence."""

    if config.stage != "stage1" or protocol.stage != "stage1":
        raise ValueError("static format gate is Stage 1 only")
    if config.format_gate_jsonl is None:
        raise ValueError("Stage 1 format gate requires full-heldout JSONL")
    source = config.format_gate_jsonl.resolve()
    selected = _load_selected_records(source)
    prepared = [_prompt_and_lineage(row, source, protocol) for row in selected]
    gate_output = config.output_dir / "format_gate"
    contract = {
        "evaluation": "stage1_static_format_gate_v1",
        "checkpoint": str(config.checkpoint.resolve()),
        "policy_fingerprint": policy_artifact_fingerprint(config.checkpoint),
        "full_heldout_jsonl": str(source),
        "full_heldout_sha256": file_artifact_fingerprint(source),
        "selection": "first_32_records_in_jsonl_order",
        "denominator": FORMAT_GATE_SIZE,
        "minimum_correct": FORMAT_GATE_MINIMUM_CORRECT,
        "generation": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_response_tokens": config.max_response_tokens,
            "generation_seed": config.generation_seed,
            "tensor_parallel_size": config.tensor_parallel_size,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_pixels": config.max_pixels,
            "vllm_mm_processor_cache_gb": config.vllm_mm_processor_cache_gb,
            "vllm_enable_prefix_caching": config.vllm_enable_prefix_caching,
            "vllm_distributed_executor_backend": (
                config.vllm_distributed_executor_backend
            ),
        },
        "selected": [lineage for _prompt, _images, lineage in prepared],
    }
    if config.episode_concurrency != 1:
        contract["generation"]["episode_concurrency"] = config.episode_concurrency
    from .cli import write_or_validate_contract

    write_or_validate_contract(gate_output, contract, resume=config.resume)
    records_dir = gate_output / "records"
    completed: list[dict[str, Any]] = []
    pending = any(
        not (records_dir / f"{index:03d}.json").is_file()
        for index in range(FORMAT_GATE_SIZE)
    )
    if pending and not allow_generate:
        raise ValueError("Stage 1 format gate is incomplete; read-only verification refused")
    if pending and generator is None:
        generator = EarlyVLLMGenerator(config, protocol)
    validation_tokenizer = generator.tokenizer if generator is not None else None
    if validation_tokenizer is None:
        from transformers import AutoProcessor

        validation_tokenizer = AutoProcessor.from_pretrained(
            str(config.checkpoint)
        ).tokenizer
    generated_pending = {}
    for index, ((prompt, images, lineage), row) in enumerate(
        zip(prepared, selected, strict=True)
    ):
        output_path = records_dir / f"{index:03d}.json"
        selection = {"index": index, **lineage}
        if output_path.is_file():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            if set(saved) != {
                "selection",
                "prompt_messages",
                "generation",
                "termination_validation",
                "reference_success",
            }:
                raise ValueError(f"invalid saved format-gate record: {output_path}")
            if saved.get("selection") != selection:
                raise ValueError(f"format-gate result identity changed: {output_path}")
            if saved.get("prompt_messages") != prompt:
                raise ValueError(f"format-gate prompt evidence changed: {output_path}")
            generation_payload = saved.get("generation")
            expected_generation_fields = {
                "text",
                "sampled_token_ids",
                "inserted_token_ids",
                "finish_reason",
                "stop_reason",
            }
            if (
                not isinstance(generation_payload, dict)
                or set(generation_payload) != expected_generation_fields
            ):
                raise ValueError(f"invalid saved format-gate generation: {output_path}")
            if (
                not isinstance(generation_payload["text"], str)
                or not isinstance(generation_payload["sampled_token_ids"], list)
                or any(
                    type(token_id) is not int
                    for token_id in generation_payload["sampled_token_ids"]
                )
                or generation_payload["inserted_token_ids"] != []
                or not isinstance(generation_payload["finish_reason"], str)
                or not (
                    generation_payload["stop_reason"] is None
                    or isinstance(generation_payload["stop_reason"], (int, str))
                )
            ):
                raise ValueError(f"invalid saved format-gate generation types: {output_path}")
            restored_generation = RawGeneration(
                text=generation_payload["text"],
                sampled_token_ids=tuple(generation_payload["sampled_token_ids"]),
                inserted_token_ids=tuple(generation_payload["inserted_token_ids"]),
                finish_reason=generation_payload["finish_reason"],
                stop_reason=generation_payload["stop_reason"],
            )
            recomputed = asdict(
                validate_stage1_raw_generation(
                    restored_generation,
                    # Existing evidence is only accepted against this checkpoint's
                    # tokenizer. A pending sibling initializes the shared generator.
                    validation_tokenizer,
                )
            )
            from nimloth.backbone.qwen25vl.early_generation import decode_response

            canonical_text = decode_response(
                validation_tokenizer,
                generation_payload["sampled_token_ids"],
            )
            if generation_payload["text"] != canonical_text:
                raise ValueError(f"format-gate generation text changed: {output_path}")
            if saved["reference_success"] != row.get("success"):
                raise ValueError(f"format-gate reference label changed: {output_path}")
            if saved.get("termination_validation") != recomputed:
                raise ValueError(
                    f"saved format-gate validation cannot be reproduced: {output_path}"
                )
            completed.append(saved)
            continue
        if generator is None:
            raise RuntimeError("format-gate generator was not initialized")
        if index not in generated_pending:
            batch_indices = [candidate for candidate in range(index, FORMAT_GATE_SIZE)
                             if not (records_dir / f"{candidate:03d}.json").is_file()
                             ][:config.episode_concurrency]
            inputs = [(prepared[candidate][0], prepared[candidate][1])
                      for candidate in batch_indices]
            generations = ([generator.generate(*inputs[0])]
                           if config.episode_concurrency == 1
                           else generator.generate_batch(inputs))
            generated_pending.update(zip(batch_indices, generations, strict=True))
        generated = generated_pending.pop(index)
        validation = validate_stage1_raw_generation(generated, generator.tokenizer)
        result = {
            "selection": selection,
            "prompt_messages": prompt,
            "generation": asdict(generated),
            "termination_validation": asdict(validation),
            "reference_success": row.get("success"),
        }
        write_json(output_path, result)
        completed.append(result)
    correct = sum(
        bool(row["termination_validation"]["format_correct"])
        for row in completed
    )
    summary = {
        "schema": "stage1_static_format_gate_v1",
        "correct": correct,
        "denominator": FORMAT_GATE_SIZE,
        "format_correct_rate": correct / FORMAT_GATE_SIZE,
        "minimum_correct": FORMAT_GATE_MINIMUM_CORRECT,
        "passed": correct >= FORMAT_GATE_MINIMUM_CORRECT,
    }
    summary_path = gate_output / "summary.json"
    if summary_path.is_file():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("saved Stage 1 format-gate summary is inconsistent")
    else:
        write_json(summary_path, summary)
    print(json.dumps({"stage1_format_gate": summary}), flush=True)
    return bool(summary["passed"]), generator
