#!/usr/bin/env python3
"""Nimloth lightweight memory CLI."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if os.environ.get("NIMLOTH_ROOT"):
    ROOT = Path(os.environ["NIMLOTH_ROOT"]).resolve()
else:
    ROOT = Path(__file__).resolve().parents[4]

STORE_REPO = "repo"
STORE_LOCAL = "local"
STORE_ALL = "all"
VALID_STORE_CHOICES = {STORE_REPO, STORE_LOCAL, STORE_ALL}
STORE_DIRS = {
    STORE_REPO: ROOT / ".memory",
    STORE_LOCAL: ROOT / ".local" / "memory",
}
STORE_PATHS = {scope: store_dir / "memories.jsonl" for scope, store_dir in STORE_DIRS.items()}
LEVEL_PENDING = "pending-human-verification"
LEVEL_VERIFIED = "verified"
LEVEL_ARCHIVED = "archived"
VALID_LEVELS = {LEVEL_PENDING, LEVEL_VERIFIED, LEVEL_ARCHIVED}
TRIGGER_STALE_DAYS = 7
UPVOTE_STALE_DAYS = 14


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def store_dir(scope: str) -> Path:
    if scope not in STORE_DIRS:
        raise SystemExit(f"Unsupported store: {scope}")
    return STORE_DIRS[scope]


def store_path(scope: str) -> Path:
    return STORE_PATHS[scope]


def resolve_single_store(scope: str) -> str:
    if scope == STORE_ALL:
        raise SystemExit("This command requires --store repo or --store local.")
    if scope not in {STORE_REPO, STORE_LOCAL}:
        raise SystemExit(f"Unsupported store: {scope}")
    return scope


def load_memories(scope: str) -> list[dict[str, Any]]:
    path = store_path(scope)
    store_dir(scope).mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {path}:{line_no}: {exc}")
    return out


def save_memories(scope: str, memories: list[dict[str, Any]]) -> None:
    path = store_path(scope)
    store_dir(scope).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for mem in memories:
            f.write(json.dumps(mem, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def next_id(memories: list[dict[str, Any]]) -> str:
    max_n = 0
    for mem in memories:
        m = re.fullmatch(r"M(\d{4,})", str(mem.get("id", "")))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"M{max_n + 1:04d}"


def find_memory(memories: list[dict[str, Any]], mid: str) -> dict[str, Any]:
    for mem in memories:
        if mem.get("id") == mid:
            return mem
    raise SystemExit(f"Memory not found: {mid}")


def validate_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SystemExit("evidence must be a JSON list")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise SystemExit(f"evidence[{i}] must be an object")
        filename = item.get("filename")
        line_start = item.get("line_start")
        total_lines = item.get("total_lines")
        if not isinstance(filename, str) or not filename:
            raise SystemExit(f"evidence[{i}].filename must be a non-empty string")
        if not isinstance(line_start, int) or line_start < 1:
            raise SystemExit(f"evidence[{i}].line_start must be a positive integer")
        if not isinstance(total_lines, int) or total_lines < 1:
            raise SystemExit(f"evidence[{i}].total_lines must be a positive integer")
        out.append({"filename": filename, "line_start": line_start, "total_lines": total_lines})
    return out


def validate_tags(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise SystemExit("tags must be a JSON list of non-empty strings")
    seen: set[str] = set()
    out: list[str] = []
    for tag in value:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def lazy_archive(memories: list[dict[str, Any]]) -> bool:
    changed = False
    now = datetime.now(timezone.utc)
    for mem in memories:
        if mem.get("level") != LEVEL_VERIFIED:
            continue
        created = parse_time(mem.get("created_at")) or now
        last_triggered = parse_time(mem.get("last_triggered_verification_at")) or created
        last_upvoted = parse_time(mem.get("last_upvoted_at")) or created
        reasons = []
        if now - last_triggered > timedelta(days=TRIGGER_STALE_DAYS):
            reasons.append(f"not trigger-verified for >{TRIGGER_STALE_DAYS} days")
        if now - last_upvoted > timedelta(days=UPVOTE_STALE_DAYS):
            reasons.append(f"not upvoted for >{UPVOTE_STALE_DAYS} days")
        if reasons:
            mem["level"] = LEVEL_ARCHIVED
            mem["archived_at"] = now_iso()
            mem["archive_reason"] = "; ".join(reasons)
            mem["updated_at"] = now_iso()
            changed = True
    return changed


def compact(mem: dict[str, Any], scope: str | None = None) -> str:
    tags = ",".join(mem.get("tags", []))
    suffix = f" tags=[{tags}]" if tags else ""
    prefix = f"[{scope}] " if scope else ""
    return f"{prefix}{mem.get('id')} [{mem.get('level')}] {mem.get('title')}{suffix}"


def evidence_text(mem: dict[str, Any]) -> str:
    return " ".join(f"{ev['filename']}:{ev['line_start']}+{ev['total_lines']}" for ev in mem.get("evidence", []))


def cmd_add(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    memories = load_memories(scope)
    lazy_archive(memories)
    ts = now_iso()
    mem = {
        "id": next_id(memories),
        "title": args.title,
        "content": args.content,
        "evidence": [],
        "tags": [],
        "level": LEVEL_PENDING,
        "created_at": ts,
        "updated_at": ts,
        "last_triggered_verification_at": None,
        "last_upvoted_at": None,
        "archived_at": None,
        "archive_reason": None,
    }
    memories.append(mem)
    save_memories(scope, memories)
    print(compact(mem, scope))
    print(f"Created pending {scope} memory. Add evidence/tags with ./skill memory set --store {scope}, then ask human to approve via ./skill human memory-approve --store {scope}.")
    return 0


def parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"Expected field=value assignment, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"Empty field in assignment: {raw}")
    return key, value


def cmd_set(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    memories = load_memories(scope)
    lazy_archive(memories)
    mem = find_memory(memories, args.id)
    for raw in args.assignments:
        key, value = parse_assignment(raw)
        if key == "title":
            mem["title"] = value
        elif key == "content":
            mem["content"] = value
        elif key in {"evidence", "evidences"}:
            mem["evidence"] = validate_evidence(json.loads(value))
        elif key in {"tag", "tags"}:
            mem["tags"] = validate_tags(json.loads(value))
        else:
            raise SystemExit(f"Unsupported field: {key}. Use title, content, evidence, or tags.")
    mem["updated_at"] = now_iso()
    save_memories(scope, memories)
    print(compact(mem, scope))
    return 0


def searchable_blob(mem: dict[str, Any], field: str) -> str:
    if field == "title":
        return str(mem.get("title", ""))
    if field == "content":
        return str(mem.get("content", ""))
    if field == "evidence.filename":
        return " ".join(str(ev.get("filename", "")) for ev in mem.get("evidence", []))
    if field == "tags":
        return " ".join(mem.get("tags", []))
    if field == "all":
        return "\n".join([
            searchable_blob(mem, "title"),
            searchable_blob(mem, "content"),
            searchable_blob(mem, "evidence.filename"),
            searchable_blob(mem, "tags"),
        ])
    raise SystemExit(f"Unsupported field: {field}")


def cmd_search(args: argparse.Namespace) -> int:
    scopes = [STORE_REPO, STORE_LOCAL] if args.store == STORE_ALL else [resolve_single_store(args.store)]
    records: list[tuple[str, dict[str, Any]]] = []
    try:
        regex = re.compile(args.regex, re.IGNORECASE)
    except re.error as exc:
        raise SystemExit(f"Invalid regex: {exc}")
    for scope in scopes:
        memories = load_memories(scope)
        if lazy_archive(memories):
            save_memories(scope, memories)
        for mem in memories:
            if mem.get("level") == LEVEL_ARCHIVED and not args.include_archived:
                continue
            if args.level and mem.get("level") != args.level:
                continue
            if args.tag and args.tag not in mem.get("tags", []):
                continue
            if regex.search(searchable_blob(mem, args.field)):
                records.append((scope, mem))
    records.sort(key=lambda item: (item[0], item[1].get("id", "")))
    for scope, mem in records:
        print(compact(mem, scope))
        print(f"  {mem.get('content')}")
        suggestions = mem.get("human_suggestions") or []
        if suggestions:
            print("  HUMAN SUGGESTIONS:")
            for sug in suggestions:
                print(f"  - {sug.get('text')} ({sug.get('created_at')})")
        ev = evidence_text(mem)
        if ev:
            print(f"  evidence: {ev}")
    if not records:
        print("No memories found.")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    memories = load_memories(scope)
    changed = lazy_archive(memories)
    mem = find_memory(memories, args.id)
    print(json.dumps(mem, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nBefore relying on this memory, inspect each evidence file segment. If it is useful after verification, run: ./skill memory upvote --store {scope} {mem.get('id')}")
    if changed:
        save_memories(scope, memories)
    return 0


def cmd_upvote(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    memories = load_memories(scope)
    lazy_archive(memories)
    mem = find_memory(memories, args.id)
    if mem.get("level") != LEVEL_VERIFIED:
        raise SystemExit("Only verified memories can be upvoted. Ask the human to approve pending memories first.")
    ts = now_iso()
    mem["last_triggered_verification_at"] = ts
    mem["last_upvoted_at"] = ts
    mem["updated_at"] = ts
    save_memories(scope, memories)
    print(compact(mem, scope))
    print("Upvoted: evidence was verified and the memory was useful for the current task.")
    return 0


def cmd_human_verify(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    memories = load_memories(scope)
    lazy_archive(memories)
    mem = find_memory(memories, args.id)
    if mem.get("level") == LEVEL_ARCHIVED:
        raise SystemExit("Archived memory cannot be submitted for human verification. Create or correct a pending memory instead.")
    mem["level"] = LEVEL_PENDING
    mem["updated_at"] = now_iso()
    save_memories(scope, memories)
    print(compact(mem, scope))
    print(f"Submitted {scope} memory for human verification. Human should run: ./skill human memory-approve --store {scope}")
    return 0


def cmd_verify_ai_memory(args: argparse.Namespace) -> int:
    scope = resolve_single_store(args.store)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("Human-only memory approval requires an interactive TTY. AI agents must not run this command.")
    print("HUMAN-ONLY COMMAND: AI agents must not run memory approval.")
    phrase = "I am the human developer"
    if input(f"Type exactly '{phrase}' to continue: ") != phrase:
        raise SystemExit("Confirmation phrase mismatch; aborting.")
    memories = load_memories(scope)
    lazy_archive(memories)
    pending = [m for m in memories if m.get("level") == LEVEL_PENDING]
    if not pending:
        save_memories(scope, memories)
        print("No pending memories.")
        return 0
    print("Human memory verification. Choose: a=approve, r=reject/delete, s=skip, q=quit")
    kept: list[dict[str, Any]] = []
    pending_ids = {m["id"] for m in pending}
    for mem in memories:
        if mem.get("id") not in pending_ids:
            kept.append(mem)
            continue
        while True:
            print("\n" + "=" * 72)
            print(compact(mem, scope))
            print(mem.get("content"))
            ev = evidence_text(mem)
            if ev:
                print("evidence:", ev)
            if mem.get("tags"):
                print("tags:", ", ".join(mem.get("tags", [])))
            suggestions = mem.get("human_suggestions") or []
            if suggestions:
                print("human suggestions for AI to follow before approval:")
                for sug in suggestions:
                    print(f"- {sug.get('text')} ({sug.get('created_at')})")
            raw_choice = input("Approve/reject/skip/quit, or type a suggestion for the AI [a/r/s/q]: ").strip()
            choice = raw_choice.lower()
            if choice in {"a", "approve"}:
                ts = now_iso()
                mem["level"] = LEVEL_VERIFIED
                mem["updated_at"] = ts
                mem["last_triggered_verification_at"] = ts
                mem["last_upvoted_at"] = ts
                mem["human_verified_at"] = ts
                mem.pop("human_suggestions", None)
                kept.append(mem)
                print("Approved. Human suggestions were removed from the verified memory.")
                break
            if choice in {"r", "reject"}:
                print("Rejected and deleted.")
                break
            if choice in {"s", "skip", ""}:
                kept.append(mem)
                print("Skipped.")
                break
            if choice in {"q", "quit"}:
                kept.append(mem)
                seen = {m.get("id") for m in kept}
                for rest in memories:
                    if rest.get("id") not in seen:
                        kept.append(rest)
                save_memories(scope, kept)
                print("Quit.")
                return 0
            mem.setdefault("human_suggestions", []).append({"text": raw_choice, "created_at": now_iso()})
            mem["level"] = LEVEL_PENDING
            mem["updated_at"] = now_iso()
            kept.append(mem)
            print("Added human suggestion; memory remains pending human approval.")
            break
    save_memories(scope, kept)
    return 0


def add_store_arg(parser: argparse.ArgumentParser, *, default: str, allow_all: bool = False, help_suffix: str = "") -> None:
    choices = [STORE_REPO, STORE_LOCAL]
    if allow_all:
        choices.append(STORE_ALL)
    parser.add_argument(
        "--store",
        default=default,
        choices=choices,
        help=f"Memory store ({', '.join(choices)}). {help_suffix}".strip(),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memory", description="Nimloth lightweight memory CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add pending memory")
    add_store_arg(a, default=STORE_REPO, help_suffix="Default is repo.")
    a.add_argument("title")
    a.add_argument("content")
    a.set_defaults(func=cmd_add)
    s = sub.add_parser("set", help="set title/content/evidence/tags")
    add_store_arg(s, default=STORE_REPO, help_suffix="Default is repo.")
    s.add_argument("id")
    s.add_argument("assignments", nargs="+")
    s.set_defaults(func=cmd_set)
    se = sub.add_parser("search", help="regex search title/content/evidence filename/tags")
    add_store_arg(se, default=STORE_ALL, allow_all=True, help_suffix="Default is all stores.")
    se.add_argument("regex")
    se.add_argument("--field", default="all", choices=["all", "title", "content", "evidence.filename", "tags"])
    se.add_argument("--tag")
    se.add_argument("--level", choices=sorted(VALID_LEVELS))
    se.add_argument("--include-archived", action="store_true")
    se.set_defaults(func=cmd_search)
    g = sub.add_parser("get", help="show full memory")
    add_store_arg(g, default=STORE_REPO, help_suffix="Default is repo.")
    g.add_argument("id")
    g.set_defaults(func=cmd_get)
    u = sub.add_parser("upvote", help="mark verified memory as verified-and-useful for current task")
    add_store_arg(u, default=STORE_REPO, help_suffix="Default is repo.")
    u.add_argument("id")
    u.set_defaults(func=cmd_upvote)
    hv = sub.add_parser("human-verify", help="submit memory for human verification")
    add_store_arg(hv, default=STORE_REPO, help_suffix="Default is repo.")
    hv.add_argument("id")
    hv.set_defaults(func=cmd_human_verify)
    vai = sub.add_parser("verify-ai-memory", help="human-only interactive approval UI")
    add_store_arg(vai, default=STORE_REPO, help_suffix="Default is repo.")
    vai.set_defaults(func=cmd_verify_ai_memory)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
