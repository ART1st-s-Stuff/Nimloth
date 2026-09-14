#!/usr/bin/env python3
"""Hard-link exact duplicate files in best to the validated best epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-epoch", type=int, default=None)
    args = parser.parse_args()

    root = args.experiment_root.resolve(strict=True)
    train = (root / "train").resolve(strict=True)
    best = (train / "best").resolve(strict=True)
    metrics_path = train / "validation_metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not metrics:
        raise RuntimeError("no validation metrics")
    metric_minimum = min(metrics, key=lambda item: item["validation_total_loss"])
    best_epoch = (
        int(args.canonical_epoch)
        if args.canonical_epoch is not None
        else int(metric_minimum["epoch"])
    )
    best_metric = next(item for item in metrics if int(item["epoch"]) == best_epoch)
    epoch = (train / f"epoch_{best_epoch:03d}").resolve(strict=True)
    if not (epoch / "COMMITTED").is_file():
        raise RuntimeError("best epoch is not committed")

    train_fragment = str(train).encode()
    active = []
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

    entries = []
    for best_file in sorted(path for path in best.rglob("*") if path.is_file()):
        if best_file.is_symlink():
            raise RuntimeError(f"unexpected symlink in best: {best_file}")
        relative = best_file.relative_to(best)
        epoch_file = epoch / relative
        if not epoch_file.is_file() or epoch_file.is_symlink():
            entries.append({"path": str(relative), "status": "missing_in_epoch"})
            continue
        best_stat = best_file.stat()
        epoch_stat = epoch_file.stat()
        if best_stat.st_dev != epoch_stat.st_dev:
            raise RuntimeError("best and epoch are on different filesystems")
        if best_stat.st_size != epoch_stat.st_size:
            entries.append({
                "path": str(relative), "status": "different_size",
                "best_bytes": best_stat.st_size, "epoch_bytes": epoch_stat.st_size,
            })
            continue
        best_hash = sha256(best_file)
        epoch_hash = sha256(epoch_file)
        entries.append({
            "path": str(relative), "bytes": best_stat.st_size,
            "sha256": best_hash,
            "status": "duplicate" if best_hash == epoch_hash else "different_hash",
            **({"epoch_sha256": epoch_hash} if best_hash != epoch_hash else {}),
        })

    duplicates = [entry for entry in entries if entry["status"] == "duplicate"]
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    payload = {
        "created_at": time.time(),
        "experiment_root": str(root),
        "selection_metric": "validation_total_loss",
        "best_metric": best_metric,
        "metric_minimum": metric_minimum,
        "best_dir": str(best),
        "canonical_epoch_dir": str(epoch),
        "entries": entries,
        "duplicate_bytes": sum(item["bytes"] for item in duplicates),
        "status": "linking",
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.sync()

    for entry in duplicates:
        best_file = best / entry["path"]
        epoch_file = epoch / entry["path"]
        temporary = best_file.with_name(f".{best_file.name}.dedup-tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        os.link(epoch_file, temporary)
        os.replace(temporary, best_file)
        if best_file.stat().st_ino != epoch_file.stat().st_ino:
            raise RuntimeError(f"hard-link verification failed: {best_file}")

    payload["completed_at"] = time.time()
    payload["status"] = "complete"
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(args.manifest),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "linked_files": len(duplicates),
        "duplicate_bytes": payload["duplicate_bytes"],
        "different_files": [entry for entry in entries if entry["status"] != "duplicate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
