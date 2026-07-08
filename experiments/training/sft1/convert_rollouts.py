#!/usr/bin/env python3
"""Convert VAGEN rollout dumps into strict Nimloth SFT records.

Input layout is the rollout-only validation tree produced by
`sft1_rollouts_vagen50_ws2_2node_externalenv.slurm`:

    validation/{train,val,test}/shard_*/{step}.jsonl
    validation/{train,val,test}/shard_*/image_{step}/images_<record_idx>/*.png

The converter preserves split boundaries, rewrites VAGEN assistant actions from
`<action>move_forward</action>` / `<answer>moveahead, moveleft</answer>` into the
Nimloth prompt/action format described in DESIGN_DOCS.md, and stores exact image
paths for every `<image>` placeholder. Multi-action turns are represented by
multiple Nimloth action tokens inside one action block.

Training policy: both `train_all.jsonl` and `train_success.jsonl` are emitted.
`train_all.jsonl` is the default SFT train file (all train-split rollouts, including
failed trajectories). `train_success.jsonl` is a success-only subset for ablations.
Validation/test records include both successful and failed rollouts for held-out eval.
"""

from __future__ import annotations

import argparse
import importlib.util
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

try:
    from vagen.envs.navigation.utils.nimloth_format import (
        ACTION_NAMES,
        ACTION_TO_IDX,
        ACTION_TOKEN,
        NIMLOTH_FORMAT_INSTRUCTION,
        SPECIAL_TOKENS,
    )
except ModuleNotFoundError:
    # nimloth/vagen-legacy-dev moved navigation helpers from
    # vagen.envs.navigation.* to vagen.env.navigation.*. Load the constants
    # by file path to avoid importing vagen.env.__init__ and its optional env deps.
    _NIMLOTH_FORMAT_PATH = _VAGEN_ROOT / "vagen" / "env" / "navigation" / "nimloth_format.py"
    _spec = importlib.util.spec_from_file_location("nimloth_navigation_format", _NIMLOTH_FORMAT_PATH)
    if _spec is None or _spec.loader is None:
        raise
    _nimloth_format = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_nimloth_format)
    ACTION_NAMES = _nimloth_format.ACTION_NAMES
    ACTION_TO_IDX = _nimloth_format.ACTION_TO_IDX
    ACTION_TOKEN = _nimloth_format.ACTION_TOKEN
    NIMLOTH_FORMAT_INSTRUCTION = _nimloth_format.NIMLOTH_FORMAT_INSTRUCTION
    SPECIAL_TOKENS = _nimloth_format.SPECIAL_TOKENS

ACTION_NAMES = list(ACTION_NAMES)
ACTION_TO_IDX = dict(ACTION_TO_IDX)
ACTION_TOKEN = dict(ACTION_TOKEN)
SPECIAL_TOKENS = list(SPECIAL_TOKENS)
ACTION_ALIASES = {
    "moveahead": "move_forward",
    "moveforward": "move_forward",
    "move_forward": "move_forward",
    "moveback": "move_backward",
    "movebackward": "move_backward",
    "move_backward": "move_backward",
    "moveright": "move_right",
    "move_right": "move_right",
    "moveleft": "move_left",
    "move_left": "move_left",
    "rotateright": "turn_right",
    "turnright": "turn_right",
    "turn_right": "turn_right",
    "rotateleft": "turn_left",
    "turnleft": "turn_left",
    "turn_left": "turn_left",
    "lookup": "look_up",
    "look_up": "look_up",
    "lookdown": "look_down",
    "look_down": "look_down",
}

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
ACTION_RE = re.compile(r"<action>\s*([^<]+?)\s*</action>", re.S)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
PLAIN_ACTION_RE = re.compile(r"(?:^|\n)\s*action\s*[:：]?\s*([^\n<|]+)", re.I)
NIMLOTH_ACTION_RE = re.compile(r"<\|action_\((\d+)\)\|>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
PLAIN_THINK_RE = re.compile(r"(?:^|\n)\s*think\s*[:：]?\s*(.*?)(?=(?:\n\s*action\s*[:：]?)|$)", re.I | re.S)


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


def nimloth_action_block_for_instruction(max_actions_per_step: int = 1) -> str:
    if max_actions_per_step <= 1:
        return "<|latent_state|><|action_start|><|action_(idx)|><|action_end|>"
    return "<|latent_state|><|action_start|><|action_(idx)|>[<|action_(idx)|>...]<|action_end|>"


def nimloth_format_instruction(max_actions_per_step: int = 1) -> str:
    if max_actions_per_step <= 1:
        return NIMLOTH_FORMAT_INSTRUCTION
    legend = ", ".join(f"{idx}={name}" for idx, name in enumerate(ACTION_NAMES))
    return (
        "Respond in this format:\n"
        f"<think>...</think>{nimloth_action_block_for_instruction(max_actions_per_step)}\n"
        f"Output 1 to {max_actions_per_step} action token(s) in execution order; "
        f"each idx is one of: {legend}."
    )


def rewrite_prompt_instruction(content: str, max_actions_per_step: int = 1) -> str:
    """Rewrite VAGEN action-format instructions into Nimloth format."""
    target_instruction = nimloth_format_instruction(max_actions_per_step)
    target_action_block = nimloth_action_block_for_instruction(max_actions_per_step)
    replacements = [
        (
            "You can optionally think first, then give your action. Respond in this format:\n"
            "<think>...</think><action>some_action</action>",
            "You can optionally think first, then give your action. " + target_instruction,
        ),
        (
            "Respond in this format:\n<think>...</think><action>some_action</action>",
            target_instruction,
        ),
        (
            "<think>...</think><action>some_action</action>",
            f"<think>...</think>{target_action_block}",
        ),
        (
            "<think>...</think><answer>some_action</answer>",
            f"<think>...</think>{target_action_block}",
        ),
        (
            "<action>{action_example}</action>",
            "<|action_start|><|action_(idx)|><|action_end|>" if max_actions_per_step <= 1 else "<|action_start|><|action_(idx)|>[<|action_(idx)|>...]<|action_end|>",
        ),
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    if max_actions_per_step > 1:
        content = re.sub(
            r"You can take up to \d+ action\(s\) at a time, separated by '[^']*'\.",
            f"You can take up to {max_actions_per_step} action(s) at a time. Output them as ordered action index tokens inside the Nimloth action block.",
            content,
        )
    # Avoid stale XML-action wording in instructions and examples where possible.
    content = content.replace("<action>...</action>", target_action_block)
    content = content.replace("<answer>...</answer>", target_action_block)
    content = re.sub(r"<action>[^<]*</action>", target_action_block, content)
    content = re.sub(r"<answer>[^<]*</answer>", target_action_block, content)
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


def _normalize_action(raw: str) -> str | None:
    raw = raw.strip().lower()
    raw = raw.strip(" []'\".，。")
    raw_compact = raw.replace(" ", "_").replace("-", "_")
    if raw_compact in ACTION_ALIASES:
        return ACTION_ALIASES[raw_compact]
    if raw_compact in ACTION_TO_IDX:
        return raw_compact
    for alias, name in ACTION_ALIASES.items():
        if alias in raw_compact:
            return name
    for name in ACTION_NAMES:
        if name in raw_compact:
            return name
    return None


def _split_action_candidates(raw: str) -> list[str]:
    raw = raw.replace("，", ",").replace("；", ";")
    return [part.strip() for part in re.split(r"\s*(?:,|;|\||\n|\band\b)\s*", raw, flags=re.I) if part.strip()]


def _normalize_actions(raw: str, max_actions: int | None = None) -> list[str]:
    actions: list[str] = []
    for part in _split_action_candidates(raw):
        action = _normalize_action(part)
        if action is None:
            continue
        actions.append(action)
        if max_actions is not None and len(actions) >= max_actions:
            break
    return actions


def extract_actions(text: str, max_actions: int | None = None) -> list[str]:
    token_actions: list[str] = []
    for m_tok in NIMLOTH_ACTION_RE.finditer(text):
        idx = int(m_tok.group(1))
        if 0 <= idx < len(ACTION_NAMES):
            token_actions.append(ACTION_NAMES[idx])
            if max_actions is not None and len(token_actions) >= max_actions:
                break
    if token_actions:
        return token_actions

    for regex in (ACTION_RE, ANSWER_RE, PLAIN_ACTION_RE):
        actions: list[str] = []
        for m in regex.finditer(text):
            remaining = None if max_actions is None else max_actions - len(actions)
            if remaining is not None and remaining <= 0:
                break
            actions.extend(_normalize_actions(m.group(1), remaining))
        if actions:
            return actions
    return []


def convert_assistant(content: str, max_actions_per_turn: int | None = None) -> tuple[str, list[str], str | None]:
    think_m = THINK_RE.search(content)
    if think_m:
        think = think_m.group(1).strip()
    else:
        plain_think_m = PLAIN_THINK_RE.search(content)
        think = plain_think_m.group(1).strip() if plain_think_m else ""
    actions = extract_actions(content, max_actions=max_actions_per_turn)
    if not actions:
        # Keep malformed/non-action responses auditable but not trainable.
        converted = f"<think>{think}</think><|latent_state|><|action_start|><|action_end|>"
        return converted, [], think
    action_tokens = "".join(ACTION_TOKEN[action] for action in actions)
    converted = f"<think>{think}</think><|latent_state|><|action_start|>{action_tokens}<|action_end|>"
    return converted, actions, think


def split_messages(
    src: SourceRecord,
    *,
    target_max_actions_per_step: int = 1,
    max_actions_per_turn: int | None = None,
) -> tuple[list[dict[str, str]], list[str], list[list[str]], list[str], list[str]]:
    obj = src.payload
    messages: list[dict[str, str]] = []
    actions: list[str] = []
    action_groups: list[list[str]] = []
    thinks: list[str] = []
    warnings: list[str] = []

    if "output_str" in obj and not (obj.get("input") or obj.get("output")):
        raw_messages = parse_im_messages(obj.get("output_str", ""))
        if raw_messages and raw_messages[-1]["role"] == "assistant" and not raw_messages[-1]["content"].strip():
            raw_messages = raw_messages[:-1]
        if not raw_messages:
            warnings.append("missing_output_str_messages")
    else:
        input_messages = parse_im_messages(obj.get("input", ""))
        output_messages = parse_output_messages(obj.get("output", ""))
        # VAGEN input stores the assistant generation prompt as an empty trailing
        # assistant message. It is not a supervised response and must be dropped.
        if input_messages and input_messages[-1]["role"] == "assistant" and not input_messages[-1]["content"].strip():
            input_messages = input_messages[:-1]
        if not input_messages:
            warnings.append("missing_input_messages")
        raw_messages = input_messages + output_messages

    for msg in raw_messages:
        if msg["role"] == "assistant":
            converted, turn_actions, think = convert_assistant(
                msg["content"],
                max_actions_per_turn=max_actions_per_turn,
            )
            msg = {"role": "assistant", "content": converted}
            if turn_actions:
                action_groups.append(turn_actions)
                actions.extend(turn_actions)
            else:
                warnings.append("missing_action_in_assistant")
            if think is not None:
                thinks.append(think)
        else:
            msg = {
                "role": msg["role"],
                "content": rewrite_prompt_instruction(msg["content"], target_max_actions_per_step),
            }
        messages.append(msg)

    # Remove accidental adjacent duplicate system+user prefix if present.
    deduped: list[dict[str, str]] = []
    for msg in messages:
        if deduped and msg == deduped[-1]:
            warnings.append("dropped_adjacent_duplicate_message")
            continue
        deduped.append(msg)

    return deduped, actions, action_groups, thinks, warnings


def image_paths_for(source_jsonl: Path, step: int, record_idx: int, payload: dict[str, Any] | None = None) -> list[str]:
    if payload:
        explicit = payload.get("image_paths")
        if isinstance(explicit, list) and explicit:
            return [str(Path(p).resolve()) for p in explicit]
    image_dir = source_jsonl.parent / f"image_{step}" / f"images_{record_idx}"
    if not image_dir.exists():
        return []

    def key(p: Path) -> tuple[int, str]:
        try:
            return (int(p.stem), p.name)
        except ValueError:
            return (10**9, p.name)

    return [str(p.resolve()) for p in sorted(image_dir.glob("*.png"), key=key)]


def validate_record(messages: list[dict[str, str]], image_paths: list[str], action_groups: list[list[str]]) -> list[str]:
    issues: list[str] = []
    if not messages or messages[0].get("role") != "system":
        issues.append("first_message_not_system")
    if not any(m.get("role") == "assistant" for m in messages):
        issues.append("no_assistant_messages")
    if not action_groups:
        issues.append("no_parsed_actions")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    if assistant_count != len(action_groups):
        issues.append(f"assistant_action_group_count_mismatch:{assistant_count}!={len(action_groups)}")
    image_placeholders = sum(str(m.get("content", "")).count("<image>") for m in messages)
    if image_placeholders != len(image_paths):
        issues.append(f"image_count_mismatch:{image_placeholders}!={len(image_paths)}")
    for m in messages:
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if "<|latent_state|>" not in c or "<|action_start|>" not in c or "<|action_end|>" not in c:
                issues.append("assistant_missing_nimloth_tokens")
                break
            if "<action>" in c or "</action>" in c or "<answer>" in c or "</answer>" in c:
                issues.append("assistant_still_has_vagen_action_xml")
                break
        else:
            c = m.get("content", "")
            if (
                "<action>some_action</action>" in c
                or "<think>...</think><action>" in c
                or "<answer>some_action</answer>" in c
                or "<think>...</think><answer>" in c
            ):
                issues.append("prompt_still_has_vagen_action_instruction")
                break
    return issues


def convert_one(
    src: SourceRecord,
    *,
    target_max_actions_per_step: int = 1,
    max_actions_per_turn: int | None = None,
) -> dict[str, Any]:
    obj = src.payload
    metrics = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
    step = int(obj.get("step", 50))
    messages, actions, action_groups, thinks, warnings = split_messages(
        src,
        target_max_actions_per_step=target_max_actions_per_step,
        max_actions_per_turn=max_actions_per_turn,
    )
    image_paths = image_paths_for(src.jsonl_path, step, src.line_index, obj)
    issues = validate_record(messages, image_paths, action_groups)
    traj_success = obj.get("traj_success", metrics.get("success", 0.0))
    success = bool(traj_success) if isinstance(traj_success, bool) else float(traj_success or 0.0) >= 1.0
    return {
        "id": f"{src.split}/{src.shard}/{src.line_index:06d}",
        "split": src.split,
        "shard": src.shard,
        "source_jsonl": str(src.jsonl_path.resolve()),
        "source_line_index": src.line_index,
        "step": step,
        "success": success,
        "traj_success": float(traj_success or 0.0),
        "reward": float(obj.get("reward", obj.get("score", metrics.get("score", 0.0))) or 0.0),
        "score": float(obj.get("score", metrics.get("score", 0.0)) or 0.0),
        "messages": messages,
        "image_paths": image_paths,
        "actions": actions,
        "action_indices": [ACTION_TO_IDX[a] for a in actions],
        "action_groups": action_groups,
        "action_indices_by_turn": [[ACTION_TO_IDX[a] for a in group] for group in action_groups],
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
    ap.add_argument(
        "--target-max-actions-per-step",
        type=int,
        default=1,
        help="Prompt rewrite capacity for Nimloth target records; use >1 for multi-action SFT prompts.",
    )
    ap.add_argument(
        "--max-actions-per-turn",
        type=int,
        default=0,
        help="Optional converter-side truncation; 0 preserves all parsed actions in each assistant turn.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.target_max_actions_per_step < 1:
        raise SystemExit("--target-max-actions-per-step must be >= 1")
    if args.max_actions_per_turn < 0:
        raise SystemExit("--max-actions-per-turn must be >= 0")
    max_actions_per_turn = args.max_actions_per_turn or None

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
        "target_max_actions_per_step": args.target_max_actions_per_step,
        "max_actions_per_turn": max_actions_per_turn,
        "format": "Nimloth SFT v1: assistant=<think>...</think>" + nimloth_action_block_for_instruction(args.target_max_actions_per_step),
        "split_policy": {
            "train_all": "all train-split rollouts; default SFT train file (success + failed)",
            "train_success": "train-split rollouts with traj_success >= 1.0 and no validation issues; optional ablation",
            "val_all/test_all": "all rollout records for held-out validation/test eval",
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
                rec = convert_one(
                    SourceRecord(split, shard, jsonl_path, line_index, payload),
                    target_max_actions_per_step=args.target_max_actions_per_step,
                    max_actions_per_turn=max_actions_per_turn,
                )
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
        group_lengths = [len(group) for r in records for group in r["action_groups"]]
        split_stats[split] = {
            "records": len(records),
            "success": sum(1 for r in records if r["success"]),
            "with_validation_issues": sum(1 for r in records if r["validation_issues"]),
            "with_warnings": sum(1 for r in records if r["warnings"]),
            "image_placeholders": sum(sum(m["content"].count("<image>") for m in r["messages"]) for r in records),
            "image_paths": sum(len(r["image_paths"]) for r in records),
            "assistant_turns": sum(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in records),
            "action_groups": len(group_lengths),
            "multi_action_groups": sum(1 for n in group_lengths if n > 1),
            "max_actions_per_group": max(group_lengths, default=0),
            "actions": sum(len(r["actions"]) for r in records),
        }

    manifest["counts"] = counts
    manifest["split_stats"] = split_stats
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# SFT1 converted rollout records\n\nSource: `{manifest['input_root']}`\n\nCheckpoint HF init for SFT: `{manifest['checkpoint_hf']}`\n\nFormat: `{manifest['format']}`\n\nFiles:\n\n- `train_all.jsonl`: all training-split rollouts; **default SFT train file** (includes failed trajectories).\n- `train_success.jsonl`: successful training-split rollouts only; optional ablation / audit.\n- `val_all.jsonl`: validation split, held out from SFT train.\n- `test_all.jsonl`: test split, held out from SFT train.\n- `manifest.json`: action mapping, counts, and conversion metadata.\n\nCounts:\n\n```json\n{json.dumps(counts, indent=2)}\n```\n\nSplit stats:\n\n```json\n{json.dumps(split_stats, indent=2)}\n```\n"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"output_root": str(out), "counts": counts, "split_stats": split_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
