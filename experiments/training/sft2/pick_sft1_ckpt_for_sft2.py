#!/usr/bin/env python3
"""Pick SFT1 LoRA epoch for SFT2 init using greedy rollout val success rate (early-stop).

Early-stop rule: among epochs with val success_rate >= (max - margin), pick the smallest epoch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_EPOCH_RE = re.compile(r"epoch_?(\d+)$", re.I)


def _epoch_from_name(name: str) -> int | None:
    m = _EPOCH_RE.search(name)
    return int(m.group(1)) if m else None


def summarize_jsonl(path: Path) -> dict:
    total = 0
    success = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            if float(obj.get("traj_success", 0.0) or 0.0) >= 1.0:
                success += 1
    return {
        "records": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
    }


def val_success_for_eval_root(eval_root: Path, step: str = "0") -> dict | None:
    split_dir = eval_root / "validation" / "val"
    if not split_dir.is_dir():
        return None
    shards: list[dict] = []
    for jsonl in sorted(split_dir.glob(f"*/{step}.jsonl")):
        row = summarize_jsonl(jsonl)
        if row["records"] > 0:
            row["shard"] = jsonl.parent.name
            shards.append(row)
    if not shards:
        return None
    total = sum(r["records"] for r in shards)
    succ = sum(r["success"] for r in shards)
    return {
        "records": total,
        "success": succ,
        "success_rate": succ / total if total else 0.0,
        "shards": len(shards),
    }


def test_success_for_eval_root(eval_root: Path, step: str = "0") -> dict | None:
    split_dir = eval_root / "validation" / "test"
    if not split_dir.is_dir():
        return None
    shards: list[dict] = []
    for jsonl in sorted(split_dir.glob(f"*/{step}.jsonl")):
        row = summarize_jsonl(jsonl)
        if row["records"] > 0:
            shards.append(row)
    if not shards:
        return None
    total = sum(r["records"] for r in shards)
    succ = sum(r["success"] for r in shards)
    return {
        "records": total,
        "success": succ,
        "success_rate": succ / total if total else 0.0,
    }


def collect_epoch_metrics(runs_root: Path, eval_tag_prefix: str) -> list[dict]:
    rows: list[dict] = []
    pattern = f"sft1_eval_vagen79_{eval_tag_prefix}_epoch*"
    for eval_dir in sorted(runs_root.glob(pattern)):
        ep = _epoch_from_name(eval_dir.name)
        if ep is None:
            continue
        val = val_success_for_eval_root(eval_dir)
        if val is None:
            continue
        test = test_success_for_eval_root(eval_dir)
        rows.append(
            {
                "epoch": ep,
                "eval_dir": str(eval_dir),
                "val": val,
                "test": test,
            }
        )
    return rows


def pick_earlystop(rows: list[dict], margin: float) -> dict:
    if not rows:
        raise ValueError("no epoch with rollout val success data")
    max_sr = max(r["val"]["success_rate"] for r in rows)
    threshold = max_sr - margin
    candidates = [r for r in rows if r["val"]["success_rate"] >= threshold]
    chosen = min(candidates, key=lambda r: r["epoch"])
    chosen = dict(chosen)
    chosen["pick_rule"] = "early_stop_by_val_success"
    chosen["max_val_success_rate"] = max_sr
    chosen["margin"] = margin
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft1-run", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=None, help="parent of sft1_eval_* dirs")
    ap.add_argument("--eval-tag-prefix", default="alltrain_8gpu_lora_cache")
    ap.add_argument("--margin", type=float, default=0.0, help="early-stop margin below max val success")
    ap.add_argument("--force-epoch", type=int, default=None, help="override epoch (skip pick)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON manifest")
    args = ap.parse_args()

    runs_root = args.runs_root or args.sft1_run.parent
    sft1_run = args.sft1_run.resolve()

    if args.force_epoch is not None:
        ep = args.force_epoch
        adapter = sft1_run / f"epoch_{ep:03d}"
        if not (adapter / "adapter_config.json").is_file():
            print(json.dumps({"error": f"missing adapter at {adapter}"}), file=sys.stderr)
            return 1
        manifest = {
            "epoch": ep,
            "adapter_dir": str(adapter),
            "hf_merged": str(adapter / "hf_merged"),
            "pick_rule": "force_epoch",
        }
    else:
        rows = collect_epoch_metrics(runs_root, args.eval_tag_prefix)
        if not rows:
            print(
                json.dumps(
                    {
                        "error": "no rollout val success data",
                        "hint": "fix SFT1 env eval or pass --force-epoch N",
                        "searched": str(runs_root / f"sft1_eval_vagen79_{args.eval_tag_prefix}_epoch*"),
                    }
                ),
                file=sys.stderr,
            )
            return 1
        chosen = pick_earlystop(rows, args.margin)
        ep = int(chosen["epoch"])
        adapter = sft1_run / f"epoch_{ep:03d}"
        manifest = {
            "epoch": ep,
            "adapter_dir": str(adapter),
            "hf_merged": str(adapter / "hf_merged"),
            "val_success_rate": chosen["val"]["success_rate"],
            "val_records": chosen["val"]["records"],
            "test_success_rate": (chosen["test"] or {}).get("success_rate"),
            "pick_rule": chosen["pick_rule"],
            "max_val_success_rate": chosen["max_val_success_rate"],
            "margin": chosen["margin"],
            "eval_dir": chosen["eval_dir"],
            "all_epochs_scored": [
                {"epoch": r["epoch"], "val_success_rate": r["val"]["success_rate"]} for r in rows
            ],
        }

    out_path = args.out or (sft1_run / "sft2_init_pick.json")
    out_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
