#!/usr/bin/env bash
set -euo pipefail

TASK_BRANCH="task/archive-legacy-trellis-knowledge"
TASK_DIR=".trellis/tasks/09-03-archive-legacy-trellis-knowledge"
MANIFEST="$TASK_DIR/research/legacy-move-manifest.jsonl"
PATCH="$TASK_DIR/research/approved-rewrite-patch.json"
VALIDATOR="$TASK_DIR/tools/legacy_archive_plan.py"
APPROVED_MOVE_MANIFEST_SHA256="990a4bbb907e156c87b1e6301899a8de9049502505110fa0a5b2a11aa078e465"
APPROVED_REWRITE_PATCH_SHA256="ff0075f148dac52c5632c9c9a30d73d17e9a575ae70ee52112a4e940ce923716"
EXECUTE=0

case "${1:-}" in
  "") ;;
  --execute-approved-moves) EXECUTE=1 ;;
  *) echo "usage: $0 [--execute-approved-moves]" >&2; exit 2 ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[[ "$(git branch --show-current)" == "$TASK_BRANCH" ]] || {
  echo "refusing: expected branch $TASK_BRANCH" >&2
  exit 1
}

if (( EXECUTE == 1 )); then
  current_manifest_sha256="$(sha256sum -- "$MANIFEST" | awk '{print $1}')"
  current_patch_sha256="$(sha256sum -- "$PATCH" | awk '{print $1}')"
  [[ "$current_manifest_sha256" == "$APPROVED_MOVE_MANIFEST_SHA256" ]] || {
    echo "refusing: move manifest differs from reviewed SHA-256 $APPROVED_MOVE_MANIFEST_SHA256" >&2
    exit 1
  }
  [[ "$current_patch_sha256" == "$APPROVED_REWRITE_PATCH_SHA256" ]] || {
    echo "refusing: rewrite patch differs from reviewed SHA-256 $APPROVED_REWRITE_PATCH_SHA256" >&2
    exit 1
  }
fi

python3 "$VALIDATOR" --check --mode pre >/dev/null

if (( EXECUTE == 0 )); then
  python3 - "$MANIFEST" "$PATCH" <<'PY'
import json, shlex, sys
from pathlib import Path
moves=[json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
patch=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
parents=sorted({str(Path(row["destination"]).parent) for row in moves})
print(f"DRY-RUN: pre-create/validate {len(parents)} destination directories")
for parent in parents:
    print(f"mkdir -p -- {shlex.quote(parent)}")
print(f"DRY-RUN: {len(moves)} byte-preserving git moves")
for row in moves:
    print(f"git mv -- {shlex.quote(row['source'])} {shlex.quote(row['destination'])}")
print(f"DRY-RUN: hash-bound rewrites in {len(patch['files'])} files")
for entry in patch["files"]:
    print(f"rewrite --pre-sha256 {entry['pre_sha256']} --post-sha256 {entry['post_sha256']} -- {shlex.quote(entry['path'])}")
PY
  echo "No files moved or rewritten. Mutation requires the exact --execute-approved-moves flag after destructive approval."
  exit 0
fi

# The Python block owns one exact operation: all-directory preflight, 254 moves,
# reviewed hash-bound rewrites, then post-move integrity/context/link/focused checks.
python3 - "$VALIDATOR" <<'PY'
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

validator=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location("legacy_archive_plan", validator)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {validator}")
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
root=tool.ROOT
rows=tool.load_jsonl(tool.MOVES)
patch=tool.load_rewrite_patch()

tool.check(mode="pre")
tool.prepare_destination_directories(root, rows)
# No mutation occurs until every destination directory and every destination
# absence condition has passed. Recheck root/source/ancestors/destination for
# every individual move to close the approval-to-mutation race window.
for row in rows:
    tool.assert_exact_repo_root(root)
    source=root / row["source"]
    errors=tool.validate_destination_path(root, row["destination"], require_absent=True)
    if errors:
        raise SystemExit("\n".join(errors))
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"source changed immediately before move: {row['source']}")
    if tool.sha256_file(source) != row["sha256"] or tool.worktree_blob_id(root, row["source"]) != row["blob"]:
        raise SystemExit(f"source bytes/blob changed immediately before move: {row['source']}")
    subprocess.run(["git", "mv", "--", row["source"], row["destination"]], cwd=root, check=True)

tool.apply_rewrite_patch(root, patch)
summary=tool.check(mode="post", run_focused_tests=True)
print(tool.json.dumps(summary, sort_keys=True))
PY
