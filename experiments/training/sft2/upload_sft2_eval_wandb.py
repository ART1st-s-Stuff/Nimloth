#!/usr/bin/env python3
"""Upload VAGEN greedy rollout eval metrics (per checkpoint) to wandb."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def _find_wandb_run_info(train_out: Path) -> tuple[str | None, str | None]:
    wandb_root = train_out / "wandb" / "wandb"
    latest = wandb_root / "latest-run"
    run_dir: Path | None = None
    if latest.is_symlink():
        run_dir = latest.resolve()
    else:
        runs = sorted(wandb_root.glob("run-*"))
        if runs:
            run_dir = runs[-1]

    run_id: str | None = None
    meta_project: str | None = None
    if run_dir is not None:
        if m := re.match(r"run-.*-(.+)$", run_dir.name):
            run_id = m.group(1)
        meta_path = run_dir / "files" / "wandb-metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_project = meta.get("project")
    return run_id, meta_project


def _ckpt_step(ckpt: str, train_out: Path) -> int:
    if ckpt == "init":
        return 0
    if ckpt == "best":
        state_path = train_out / "best" / "training_state.pt"
        if state_path.is_file():
            try:
                import torch

                state = torch.load(state_path, map_location="cpu", weights_only=False)
                return int(state.get("epoch", state.get("step", 0)))
            except Exception:
                pass
        return 10_000
    if m := re.fullmatch(r"epoch_(\d+)", ckpt):
        return int(m.group(1))
    return 0


def _load_checkpoints(train_out: Path, tag_prefix: str, summary_all: Path | None) -> list[dict[str, Any]]:
    if summary_all and summary_all.is_file():
        payload = json.loads(summary_all.read_text(encoding="utf-8"))
        return list(payload.get("checkpoints", []))

    eval_root = train_out / "eval_rollouts"
    rows: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return rows

    patterns = (f"sft2_eval_{tag_prefix}_*", f"sft1_eval_{tag_prefix}_*")
    seen: set[str] = set()
    for pattern in patterns:
        for run_dir in sorted(eval_root.glob(pattern)):
            key = str(run_dir)
            if key in seen:
                continue
            seen.add(key)
            ckpt = run_dir.name
            for prefix in (f"sft2_eval_{tag_prefix}_", f"sft1_eval_{tag_prefix}_"):
                if ckpt.startswith(prefix):
                    ckpt = ckpt[len(prefix) :]
                    break
            summary_path = run_dir / "summary_0.json"
            summary = None
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "ckpt": ckpt,
                "run_dir": str(run_dir),
                "summary": summary,
            }
            if summary:
                for split in ("val", "test"):
                    agg = summary.get(f"{split}_aggregate")
                    if agg:
                        row[f"{split}_success_rate"] = agg.get("success_rate")
                        row[f"{split}_records"] = agg.get("records")
                        row[f"{split}_mean_reward"] = agg.get("mean_reward")
            rows.append(row)
    return rows


def _ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = []
    for row in rows:
        if row.get("val_success_rate") is not None or row.get("test_success_rate") is not None:
            ready.append(row)
        elif row.get("summary"):
            summary = row["summary"]
            for split in ("val", "test"):
                agg = summary.get(f"{split}_aggregate")
                if agg and agg.get("success_rate") is not None:
                    row[f"{split}_success_rate"] = agg["success_rate"]
                    row[f"{split}_records"] = agg.get("records")
            if row.get("val_success_rate") is not None or row.get("test_success_rate") is not None:
                ready.append(row)
    return ready


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--tag-prefix", default="sft2_ckpt")
    ap.add_argument("--summary-all", type=Path, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--mode", default=os.environ.get("WANDB_MODE", "online"))
    ap.add_argument("--run-id", default=None, help="Resume training wandb run (auto-detect if omitted)")
    ap.add_argument("--name", default=None, help="Only used when creating a new wandb run")
    args = ap.parse_args()

    summary_all = args.summary_all or (args.train_out / "eval_rollouts" / "summary_all.json")
    rows = _ready_rows(_load_checkpoints(args.train_out, args.tag_prefix, summary_all))
    if not rows:
        print(json.dumps({"uploaded": False, "reason": "no completed eval summaries"}))
        return 0

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set")

    import wandb

    detected_run_id, meta_project = _find_wandb_run_info(args.train_out)
    run_id = args.run_id or detected_run_id
    project = args.project or meta_project or os.environ.get("WANDB_PROJECT", "nimloth")
    out_dir = args.train_out / "wandb_eval_upload"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(out_dir))

    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": args.entity,
        "dir": str(out_dir),
        "mode": args.mode,
        "config": {
            "source": "vagen_greedy_rollout_eval",
            "train_out": str(args.train_out),
            "tag_prefix": args.tag_prefix,
        },
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["name"] = args.name or f"{args.train_out.name}-eval"

    run = wandb.init(**init_kwargs)
    assert run is not None

    wandb.define_metric("eval_ckpt_step")
    wandb.define_metric("eval_online/*", step_metric="eval_ckpt_step")

    table_rows: list[list[Any]] = []
    best_val = (-1.0, "")
    best_test = (-1.0, "")

    for row in sorted(rows, key=lambda r: _ckpt_step(str(r["ckpt"]), args.train_out)):
        ckpt = str(row["ckpt"])
        step = _ckpt_step(ckpt, args.train_out)
        payload: dict[str, Any] = {"eval_ckpt_step": step, "eval_online/ckpt": ckpt}
        for split in ("val", "test"):
            rate = row.get(f"{split}_success_rate")
            records = row.get(f"{split}_records")
            if rate is not None:
                payload[f"eval_online/{split}_success_rate"] = float(rate)
                if float(rate) > best_val and split == "val":
                    best_val = (float(rate), ckpt)
                if float(rate) > best_test and split == "test":
                    best_test = (float(rate), ckpt)
            if records is not None:
                payload[f"eval_online/{split}_records"] = int(records)
        wandb.log(payload, step=step)
        table_rows.append(
            [
                ckpt,
                step,
                row.get("val_success_rate"),
                row.get("val_records"),
                row.get("test_success_rate"),
                row.get("test_records"),
            ]
        )

    table = wandb.Table(
        columns=["ckpt", "eval_ckpt_step", "val_success_rate", "val_records", "test_success_rate", "test_records"],
        data=table_rows,
    )
    wandb.log({"eval_online/checkpoint_table": table})

    if best_val[1]:
        run.summary["eval_online/best_val_success_rate"] = best_val[0]
        run.summary["eval_online/best_val_ckpt"] = best_val[1]
    if best_test[1]:
        run.summary["eval_online/best_test_success_rate"] = best_test[0]
        run.summary["eval_online/best_test_ckpt"] = best_test[1]

    if summary_all.is_file():
        wandb.save(str(summary_all), base_path=str(args.train_out))

    run.finish()
    print(
        json.dumps(
            {
                "uploaded": True,
                "wandb_run": run.url,
                "run_id": run_id,
                "checkpoints": len(rows),
                "project": project,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
