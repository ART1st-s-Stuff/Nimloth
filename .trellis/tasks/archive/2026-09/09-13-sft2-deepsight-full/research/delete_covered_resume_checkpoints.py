#!/usr/bin/env python3
"""Delete only resume checkpoints covered by a committed target epoch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.experiment_root.resolve(strict=True)
    train = (root / "train").resolve(strict=True)
    marker = train / f"epoch_{args.target_epoch:03d}" / "COMMITTED"
    stopped = root / "stop_after_epoch5_monitor" / "TARGET_REACHED.json"
    if not marker.is_file() or not stopped.is_file():
        raise RuntimeError("target epoch is not both committed and monitor-confirmed")
    target = json.loads(stopped.read_text())
    validation = target.get("validation") or {}
    if validation.get("epoch") != args.target_epoch:
        raise RuntimeError("monitor target epoch mismatch")
    if validation.get("global_step") != args.target_step:
        raise RuntimeError("monitor target step mismatch")

    active = []
    train_fragment = str(train).encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if train_fragment in cmdline:
            active.append({"pid": int(proc.name), "cmdline": cmdline.replace(b"\0", b" ").decode()})
    if active:
        raise RuntimeError(f"matching active processes: {active}")

    pattern = re.compile(r"resume_step_(\d{8})$")
    candidates = []
    for path in sorted(train.glob("resume_step_*")):
        match = pattern.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"unsafe resume candidate: {path}")
        resolved = path.resolve(strict=True)
        if resolved.parent != train:
            raise RuntimeError(f"candidate escaped train root: {resolved}")
        step = int(match.group(1))
        if step > args.target_step:
            raise RuntimeError(f"resume step {step} is newer than target {args.target_step}")
        size = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
        candidates.append({"path": str(resolved), "step": step, "bytes": size})

    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    payload = {
        "created_at": time.time(),
        "experiment_root": str(root),
        "covering_epoch": args.target_epoch,
        "covering_global_step": args.target_step,
        "committed_marker": str(marker),
        "monitor_record": str(stopped),
        "candidates": candidates,
        "total_bytes": sum(item["bytes"] for item in candidates),
        "status": "deleting",
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.sync()

    for item in candidates:
        shutil.rmtree(item["path"])

    payload["completed_at"] = time.time()
    payload["status"] = "complete"
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
