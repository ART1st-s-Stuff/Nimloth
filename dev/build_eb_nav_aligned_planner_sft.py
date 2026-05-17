"""Build leak-free pre-action Qwen planner SFT JSONL from custom EB-Nav manifests.

The old phase2 planner SFT used executable_plan.img_path, which is the post-action
image. This script realigns those teacher responses to the current pre-action
state used by the ranker/WM manifests: image = history_images[-1], label =
future_action_ids[0]. It optionally reuses the old response/COT by looking it up
with future_images[0] + label, but writes the pre-action image in the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.eb_nav_dataset import ACTION_NAMES
from src.vlm.qwen_planner import build_planner_special_response, validate_planner_special_output


def _repo_rel(path: str, repo: Path) -> str:
    raw = Path(str(path))
    try:
        return str(raw.resolve().relative_to(repo.resolve()))
    except Exception:
        s = str(path)
        marker = "/flower/"
        if marker in s:
            return s.split(marker, 1)[1]
        return s


def _image_suffix(path: str) -> str:
    s = str(path)
    marker = "datasets/EB-Nav/images/"
    if marker in s:
        return s.split(marker, 1)[1]
    marker2 = "images/"
    if marker2 in s:
        return s.split(marker2, 1)[1]
    return s


def _eval_set_from_path(path: str) -> str:
    parts = Path(str(path)).parts
    if "images" in parts:
        i = parts.index("images")
        if len(parts) > i + 2:
            return parts[i + 2]
    return ""


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            meta = d.get("metadata", {}) or {}
            cur_img = str(d["history_images"][-1])
            key = (
                str(meta.get("model_name", "")),
                _eval_set_from_path(cur_img),
                str(meta.get("episode_id", "")),
            )
            rows.append({"idx": idx, "data": d, "split_key": key})
    return rows


def _load_old_sft(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            try:
                action_id = int(d.get("action_id"))
            except Exception:
                continue
            out[(_image_suffix(str(d.get("image", ""))), action_id)] = d
    return out


def _fallback_cot(instruction: str, action_id: int) -> str:
    action_name = ACTION_NAMES.get(int(action_id), f"action_{action_id}")
    return (
        "I inspect the current observation and instruction, then choose the next navigation action "
        f"that best follows the expert trajectory. The goal is: {instruction}. "
        f"The selected next action is {action_name}."
    )


def _build_records(rows: list[dict[str, Any]], old_lookup: dict[tuple[str, int], dict[str, Any]], repo: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    stats = {"rows": 0, "reused_old_response": 0, "fallback_response": 0, "invalid_response": 0}
    for item in rows:
        stats["rows"] += 1
        d = item["data"]
        label = int(d["future_action_ids"][0])
        cur_img = str(d["history_images"][-1])
        next_img = str(d["future_images"][0])
        old = old_lookup.get((_image_suffix(next_img), label))
        if old and str(old.get("response", "")).strip():
            response = str(old["response"])
            cot = str(old.get("cot", ""))
            stats["reused_old_response"] += 1
        else:
            cot = _fallback_cot(str(d.get("instruction", "")), label)
            response = build_planner_special_response(cot=cot, action_id=label)
            stats["fallback_response"] += 1
        valid, reason, parsed_action = validate_planner_special_output(response)
        if not valid or parsed_action != label:
            cot = _fallback_cot(str(d.get("instruction", "")), label)
            response = build_planner_special_response(cot=cot, action_id=label)
            stats["invalid_response"] += 1
        meta = d.get("metadata", {}) or {}
        row_idx = int(item["idx"])
        records.append({
            "id": f"{split}_manifest_{row_idx:06d}",
            "image": _repo_rel(cur_img, repo),
            "prompt": str(d.get("prompt") or d.get("instruction") or ""),
            "response": response,
            "instruction": str(d.get("instruction", "")),
            "cot": cot,
            "action_id": label,
            "action_name": ACTION_NAMES.get(label, f"action_{label}"),
            "source_manifest_idx": row_idx,
            "source_history_image": _repo_rel(cur_img, repo),
            "source_future_image": _repo_rel(next_img, repo),
            "split_key": list(item["split_key"]),
            "metadata": meta,
        })
    return records, stats


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--old-sft-jsonl", default="datasets/EB-Nav/phase2_qwen_planner_sft_old_post_action.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-root", default=".")
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    train_rows = _manifest_rows(Path(args.train_manifest))
    test_rows = _manifest_rows(Path(args.test_manifest))
    train_keys = {r["split_key"] for r in train_rows}
    test_keys = {r["split_key"] for r in test_rows}
    overlap = train_keys & test_keys
    filtered_train = [r for r in train_rows if r["split_key"] not in overlap]
    old_lookup = _load_old_sft(Path(args.old_sft_jsonl))

    train_records, train_stats = _build_records(filtered_train, old_lookup, repo, "train")
    test_records, test_stats = _build_records(test_rows, old_lookup, repo, "test")

    out = Path(args.output_dir)
    _write_jsonl(out / "qwen_planner_sft_preaction_train.jsonl", train_records)
    _write_jsonl(out / "qwen_planner_sft_preaction_test.jsonl", test_records)

    assert {tuple(r["split_key"]) for r in train_records}.isdisjoint({tuple(r["split_key"]) for r in test_records})
    summary = {
        "train_manifest": args.train_manifest,
        "test_manifest": args.test_manifest,
        "old_sft_jsonl": args.old_sft_jsonl,
        "split_key": ["model_name", "eval_set_from_image_path", "episode_id"],
        "raw_train_rows": len(train_rows),
        "raw_test_rows": len(test_rows),
        "overlap_episode_keys_removed_from_train": len(overlap),
        "removed_train_rows": len(train_rows) - len(filtered_train),
        "train_rows": len(train_records),
        "test_rows": len(test_records),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "overlap_keys": [list(k) for k in sorted(overlap)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
