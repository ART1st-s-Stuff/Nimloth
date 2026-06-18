#!/usr/bin/env python3
"""Convert VAGEN rollout dumps into strict Nimloth SFT records.

Input layout is the rollout-only validation tree produced by
`sft1_rollouts_vagen50_ws2_2node_externalenv.slurm`:

    validation/{train,val,test}/shard_*/{step}.jsonl
    validation/{train,val,test}/shard_*/image_{step}/images_<record_idx>/*.png

The converter preserves split boundaries, rewrites VAGEN assistant actions from
`<action>move_forward</action>` into the Nimloth prompt/action format described
in DESIGN_DOCS.md, and stores exact image paths for every `<image>` placeholder.

Training policy for sft1_exp step 3: only successful train rollouts are emitted
into `train_success.jsonl`. Validation/test records are emitted separately and
include both successful and failed rollouts so downstream eval can decide what to
measure without contaminating training.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import sys

_VAGEN_ROOT = Path(__file__).resolve().parents[3] / "external" / "VAGEN"
if _VAGEN_ROOT.is_dir() and str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))

from vagen.envs.navigation.utils.nimloth_format import (
    ACTION_NAMES,
    ACTION_TO_IDX,
    ACTION_TOKEN,
    NIMLOTH_FORMAT_INSTRUCTION,
    SPECIAL_TOKENS,
)

ACTION_NAMES = list(ACTION_NAMES)
ACTION_TO_IDX = dict(ACTION_TO_IDX)
ACTION_TOKEN = dict(ACTION_TOKEN)
SPECIAL_TOKENS = list(SPECIAL_TOKENS)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
ACTION_RE = re.compile(r"<action>\s*([^<]+?)\s*</action>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass(frozen=True)
class SourceRecord:
    split: str
    shard: str
    jsonl_path: Path
    line_index: int
    payload: dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def rewrite_prompt_instruction(content: str) -> str:
    """Rewrite VAGEN action-format instructions into Nimloth format."""
    replacements = [
        (
            "You can optionally think first, then give your action. Respond in this format:\n"
            "<think>...</think><action>some_action</action>",
            "You can optionally think first, then give your action. " + NIMLOTH_FORMAT_INSTRUCTION,
        ),
        (
            "Respond in this format:\n<think>...</think><action>some_action</action>",
            NIMLOTH_FORMAT_INSTRUCTION,
        ),
        (
            "<think>...</think><action>some_action</action>",
            "<think>...</think><|latent_state|><|action_start|><|action_(idx)|><|action_end|>",
        ),
        (
            "<action>{action_example}</action>",
            "<|action_start|><|action_(idx)|><|action_end|>",
        ),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    # Avoid stale XML-action wording in instructions where possible.
    content = content.replace("<action>...</action>", "<|action_start|><|action_(idx)|><|action_end|>")
    return content


def parse_im_messages(text: str) -> list[dict[str, str]]:
    """Parse a Qwen chat-template string into role/content messages."""
    messages: list[dict[str, str]] = []
    pos = 0
    token_re = re.compile(r"<\|im_start\|>(system|user|assistant)\n", re.S)
    while True:
        m = token_re.search(text, pos)
        if not m:
            break
        role = m.group(1)
        content_start = m.end()
        end = text.find(IM_END, content_start)
        if end < 0:
            content = text[content_start:]
            pos = len(text)
        else:
            content = text[content_start:end]
            pos = end + len(IM_END)
        messages.append({"role": role, "content": content})
    return messages


def parse_output_messages(text: str) -> list[dict[str, str]]:
    """Parse VAGEN output; the first assistant response may omit im_start."""
    if text.startswith(IM_START):
        return parse_im_messages(text)
    first_start = text.find(IM_START)
    if first_start < 0:
        leading = text
        rest = ""
    else:
        leading = text[:first_start]
        rest = text[first_start:]
    messages: list[dict[str, str]] = []
    if leading:
        if leading.endswith(IM_END):
            leading = leading[: -len(IM_END)]
        leading = leading.strip("\n")
        if leading:
            messages.append({"role": "assistant", "content": leading})
    messages.extend(parse_im_messages(rest))
    return messages


def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # VAGEN parse accepts separators in some prompts; SFT1 single-action design
    # uses the first extracted primitive.
    for sep in [",", ";", "\n", " and "]:
        if sep in raw:
            raw = raw.split(sep)[0].strip()
    raw = raw.strip(" []'\"")
    if raw in ACTION_TO_IDX:
        return raw
    for name in ACTION_NAMES:
        if name in raw:
            return name
    return None


def convert_assistant(content: str) -> tuple[str, str | None, str | None]:
    think_m = THINK_RE.search(content)
    think = think_m.group(1).strip() if think_m else ""
    action = extract_action(content)
    if action is None:
        # Keep malformed/non-action responses auditable but not trainable.
        converted = f"<think>{think}</think><|latent_state|><|action_start|><|action_end|>"
        return converted, None, think
    converted = f"<think>{think}</think><|latent_state|><|action_start|>{ACTION_TOKEN[action]}<|action_end|>"
    return converted, action, think


def split_messages(src: SourceRecord) -> tuple[list[dict[str, str]], list[str], list[str], list[str]]:
    obj = src.payload
    messages: list[dict[str, str]] = []
    actions: list[str] = []
    thinks: list[str] = []
    warnings: list[str] = []

    input_messages = parse_im_messages(obj.get("input", ""))
    output_messages = parse_output_messages(obj.get("output", ""))

    # VAGEN input stores the assistant generation prompt as an empty trailing
    # assistant message. It is not a supervised response and must be dropped.
    if input_messages and input_messages[-1]["role"] == "assistant" and not input_messages[-1]["content"].strip():
        input_messages = input_messages[:-1]

    if not input_messages:
        warnings.append("missing_input_messages")
    for msg in input_messages:
        if msg["role"] == "assistant":
            converted, action, think = convert_assistant(msg["content"])
            msg = {"role": "assistant", "content": converted}
            if action:
                actions.append(action)
            else:
                warnings.append("missing_action_in_input_assistant")
            if think is not None:
                thinks.append(think)
        else:
            msg = {"role": msg["role"], "content": rewrite_prompt_instruction(msg["content"])}
        messages.append(msg)

    # `output` starts with the first assistant response and then alternates
    # user/assistant for later turns. Drop any duplicate leading messages already
    # present in input if VAGEN ever changes the split point; current dumps have
    # input = system+initial user only.
    for msg in output_messages:
        if msg["role"] == "assistant":
            converted, action, think = convert_assistant(msg["content"])
            msg = {"role": "assistant", "content": converted}
            if action:
                actions.append(action)
            else:
                warnings.append("missing_action_in_output_assistant")
            if think is not None:
                thinks.append(think)
        else:
            msg = {"role": msg["role"], "content": rewrite_prompt_instruction(msg["content"])}
        messages.append(msg)

    # Remove accidental adjacent duplicate system+user prefix if present.
    deduped: list[dict[str, str]] = []
    for msg in messages:
        if deduped and msg == deduped[-1]:
            warnings.append("dropped_adjacent_duplicate_message")
            continue
        deduped.append(msg)

    return deduped, actions, thinks, warnings


def image_paths_for(source_jsonl: Path, step: int, record_idx: int) -> list[str]:
    image_dir = source_jsonl.parent / f"image_{step}" / f"images_{record_idx}"
    if not image_dir.exists():
        return []

    def key(p: Path) -> tuple[int, str]:
        try:
            return (int(p.stem), p.name)
        except ValueError:
            return (10**9, p.name)

    return [str(p.resolve()) for p in sorted(image_dir.glob("*.png"), key=key)]


def validate_record(messages: list[dict[str, str]], image_paths: list[str], actions: list[str]) -> list[str]:
    issues: list[str] = []
    if not messages or messages[0].get("role") != "system":
        issues.append("first_message_not_system")
    if not any(m.get("role") == "assistant" for m in messages):
        issues.append("no_assistant_messages")
    if not actions:
        issues.append("no_parsed_actions")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    if assistant_count != len(actions):
        issues.append(f"assistant_action_count_mismatch:{assistant_count}!={len(actions)}")
    image_placeholders = sum(str(m.get("content", "")).count("<image>") for m in messages)
    if image_placeholders != len(image_paths):
        issues.append(f"image_count_mismatch:{image_placeholders}!={len(image_paths)}")
    for m in messages:
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if "<|latent_state|>" not in c or "<|action_start|>" not in c or "<|action_end|>" not in c:
                issues.append("assistant_missing_nimloth_tokens")
                break
            if "<action>" in c or "</action>" in c:
                issues.append("assistant_still_has_vagen_action_xml")
                break
        else:
            c = m.get("content", "")
            if "<action>some_action</action>" in c or "<think>...</think><action>" in c:
                issues.append("prompt_still_has_vagen_action_instruction")
                break
    return issues


def convert_one(src: SourceRecord) -> dict[str, Any]:
    obj = src.payload
    step = int(obj.get("step", 50))
    messages, actions, thinks, warnings = split_messages(src)
    image_paths = image_paths_for(src.jsonl_path, step, src.line_index)
    issues = validate_record(messages, image_paths, actions)
    success = float(obj.get("traj_success", 0.0) or 0.0) >= 1.0
    return {
        "id": f"{src.split}/{src.shard}/{src.line_index:06d}",
        "split": src.split,
        "shard": src.shard,
        "source_jsonl": str(src.jsonl_path.resolve()),
        "source_line_index": src.line_index,
        "step": step,
        "success": success,
        "traj_success": float(obj.get("traj_success", 0.0) or 0.0),
        "reward": float(obj.get("reward", obj.get("score", 0.0)) or 0.0),
        "score": float(obj.get("score", 0.0) or 0.0),
        "messages": messages,
        "image_paths": image_paths,
        "actions": actions,
        "action_indices": [ACTION_TO_IDX[a] for a in actions],
        "think_texts": thinks,
        "warnings": warnings,
        "validation_issues": issues,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def maybe_write_parquet(jsonl_path: Path, parquet_path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--checkpoint-hf", type=Path, required=True)
    ap.add_argument("--checkpoint-step", type=int, default=50)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.output_root
    if out.exists():
        if not args.force:
            raise SystemExit(f"output exists: {out}; pass --force to replace deterministic conversion outputs")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "input_root": str(args.input_root.resolve()),
        "output_root": str(out.resolve()),
        "checkpoint_hf": str(args.checkpoint_hf.resolve()),
        "checkpoint_step": args.checkpoint_step,
        "splits": list(args.splits),
        "action_names": ACTION_NAMES,
        "action_to_idx": ACTION_TO_IDX,
        "special_tokens": SPECIAL_TOKENS,
        "format": "Nimloth SFT v1: assistant=<think>...</think><|latent_state|><|action_start|><|action_(idx)|><|action_end|>",
        "split_policy": {
            "train_success": "only split=train records with traj_success >= 1.0 and no validation issues",
            "val_all/test_all": "all rollout records for held-out validation/test auditing/eval",
        },
    }

    all_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in args.splits}

    for split in args.splits:
        split_dir = args.input_root / split
        if not split_dir.exists():
            raise SystemExit(f"missing split dir: {split_dir}")
        jsonl_paths = sorted(split_dir.glob(f"shard_*/{args.checkpoint_step}.jsonl"))
        if not jsonl_paths:
            raise SystemExit(
                f"no rollout jsonl found for split={split} step={args.checkpoint_step} under {split_dir}"
            )
        for jsonl_path in jsonl_paths:
            shard = jsonl_path.parent.name
            for line_index, payload in iter_jsonl(jsonl_path):
                rec = convert_one(SourceRecord(split, shard, jsonl_path, line_index, payload))
                all_by_split[split].append(rec)

    train_success = [r for r in all_by_split.get("train", []) if r["success"] and not r["validation_issues"]]
    train_all = all_by_split.get("train", [])
    val_all = all_by_split.get("val", [])
    test_all = all_by_split.get("test", [])

    counts: dict[str, Any] = {}
    outputs = {
        "train_success": train_success,
        "train_all": train_all,
        "val_all": val_all,
        "test_all": test_all,
    }
    for name, records in outputs.items():
        if name == "test_all" and "test" not in args.splits:
            continue
        if name == "val_all" and "val" not in args.splits:
            continue
        if name in {"train_success", "train_all"} and "train" not in args.splits:
            continue
        jsonl_path = out / f"{name}.jsonl"
        counts[name] = write_jsonl(jsonl_path, records)
        counts[f"{name}_parquet"] = maybe_write_parquet(jsonl_path, out / f"{name}.parquet")

    split_stats: dict[str, Any] = {}
    for split, records in all_by_split.items():
        split_stats[split] = {
            "records": len(records),
            "success": sum(1 for r in records if r["success"]),
            "with_validation_issues": sum(1 for r in records if r["validation_issues"]),
            "with_warnings": sum(1 for r in records if r["warnings"]),
            "image_placeholders": sum(sum(m["content"].count("<image>") for m in r["messages"]) for r in records),
            "image_paths": sum(len(r["image_paths"]) for r in records),
            "assistant_turns": sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in records),
            "actions": sum(len(r["actions"]) for r in records),
        }

    manifest["counts"] = counts
    manifest["split_stats"] = split_stats
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# SFT1 converted rollout records\n\nSource: `{manifest['input_root']}`\n\nCheckpoint HF init for SFT: `{manifest['checkpoint_hf']}`\n\nFormat: `{manifest['format']}`\n\nFiles:\n\n- `train_success.jsonl`: successful training-split rollouts only; intended SFT train file.\n- `train_all.jsonl`: all training-split rollouts for audit.\n- `val_all.jsonl`: validation split, held out from SFT train.\n- `test_all.jsonl`: test split, held out from SFT train.\n- `manifest.json`: action mapping, counts, and conversion metadata.\n\nCounts:\n\n```json\n{json.dumps(counts, indent=2)}\n```\n\nSplit stats:\n\n```json\n{json.dumps(split_stats, indent=2)}\n```\n"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"output_root": str(out), "counts": counts, "split_stats": split_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
