#!/usr/bin/env python3
"""Stop one exact SFT process after a fully committed target epoch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def find_epoch_metric(path: Path, epoch: int) -> dict | None:
    if not path.is_file():
        return None
    found = None
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("epoch") == epoch:
            found = row
    return found


def write_status(status_dir: Path, name: str, payload: dict) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {"observed_at": time.time(), **payload}
    tmp = status_dir / f".{name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(status_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--torchrun-pid", type=int, required=True)
    parser.add_argument("--expected-output-dir", required=True)
    parser.add_argument("--target-epoch", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args()

    train_dir = args.train_dir.resolve()
    expected = str(Path(args.expected_output_dir).resolve())
    if str(train_dir) != expected:
        raise SystemExit(f"train-dir mismatch: {train_dir} != {expected}")

    marker = train_dir / f"epoch_{args.target_epoch:03d}" / "COMMITTED"
    metrics_path = train_dir / "validation_metrics.jsonl"
    required_fragments = (
        "torch.distributed.run",
        "nimloth.training.sft.stage2",
        f"--output-dir {expected}",
    )

    initial_cmdline = read_cmdline(args.torchrun_pid)
    if not all(fragment in initial_cmdline for fragment in required_fragments):
        write_status(
            args.status_dir,
            "MONITOR_REFUSED.json",
            {"pid": args.torchrun_pid, "cmdline": initial_cmdline,
             "reason": "target process identity mismatch"},
        )
        return 2

    write_status(
        args.status_dir,
        "MONITOR_ACTIVE.json",
        {"pid": args.torchrun_pid, "target_epoch": args.target_epoch,
         "train_dir": expected, "cmdline": initial_cmdline},
    )

    while True:
        metric = find_epoch_metric(metrics_path, args.target_epoch)
        committed = marker.is_file()
        cmdline = read_cmdline(args.torchrun_pid)

        if metric is not None and committed:
            if cmdline:
                if not all(fragment in cmdline for fragment in required_fragments):
                    write_status(
                        args.status_dir,
                        "MONITOR_REFUSED.json",
                        {"pid": args.torchrun_pid, "cmdline": cmdline,
                         "reason": "process identity changed before stop"},
                    )
                    return 2
                os.kill(args.torchrun_pid, signal.SIGTERM)
                action = "sent_sigterm"
            else:
                action = "already_exited_after_target"
            write_status(
                args.status_dir,
                "TARGET_REACHED.json",
                {"pid": args.torchrun_pid, "target_epoch": args.target_epoch,
                 "checkpoint_marker": str(marker), "validation": metric,
                 "action": action},
            )
            return 0

        if not cmdline:
            write_status(
                args.status_dir,
                "STOPPED_BEFORE_TARGET.json",
                {"pid": args.torchrun_pid, "target_epoch": args.target_epoch,
                 "checkpoint_committed": committed, "validation": metric},
            )
            return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
