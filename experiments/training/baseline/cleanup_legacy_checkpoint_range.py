#!/usr/bin/env python3
"""Safely remove global_step_N checkpoint paths in an explicit step range."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

STEP_RE = re.compile(r"global_step_(\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-step", required=True, type=int)
    parser.add_argument("--max-step", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")
    if args.min_step < 0 or args.max_step < args.min_step:
        raise SystemExit("invalid step range")

    matches: list[dict[str, object]] = []
    for path in args.root.rglob("global_step_*"):
        match = STEP_RE.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if not (args.min_step <= step <= args.max_step):
            continue
        # Check the lexical path rather than resolving symlinks: retry symlinks
        # may point to another checkpoint under the same root.
        absolute = path.absolute()
        if root not in absolute.parents:
            raise SystemExit(f"refusing path outside root: {path}")
        matches.append(
            {
                "path": str(absolute),
                "step": step,
                "kind": "symlink" if path.is_symlink() else "directory",
                "target": str(path.resolve()) if path.is_symlink() else None,
            }
        )

    matches.sort(key=lambda item: (int(item["step"]), str(item["path"])))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "min_step": args.min_step,
        "max_step": args.max_step,
        "execute": args.execute,
        "count": len(matches),
        "paths": matches,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if not args.execute:
        return 0

    # Remove links first, then real directories. Deepest paths go first so a
    # parent cleanup cannot obscure an independently recorded nested path.
    ordered = sorted(
        matches,
        key=lambda item: (item["kind"] != "symlink", -len(Path(str(item["path"])).parts)),
    )
    for item in ordered:
        path = Path(str(item["path"]))
        if path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise RuntimeError(f"refusing non-directory checkpoint path: {path}")
        print(f"removed {item['kind']} step={item['step']} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
