"""Read-only Trellis task-tree and implement.md work-item projection."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

_EXPLICIT_ID = re.compile(r"^W-\d{3,}$")
_CHECKBOX = re.compile(r"^\s*[-*+]\s+\[([ xX])\]\s+(.*?)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_ID_PREFIX = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def _issue(code: str, message: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "severity": "error", "message": message, **fields}


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _legacy_ref(task_ref: str, heading_path: list[str], text: str) -> str:
    payload = json.dumps(
        [task_ref, heading_path, _normalize_text(text)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "legacy-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_implement_markdown(task_ref: str, source: str) -> dict[str, Any]:
    """Parse headings and checkboxes without modifying Markdown.

    Explicit IDs are ``W-`` plus at least three digits. An attempted malformed
    W-ID never falls back to a legacy ID: the item remains unaddressable.
    """
    headings: list[tuple[int, str]] = []
    sections: list[dict[str, Any]] = []
    section_by_path: dict[tuple[str, ...], int] = {}
    items: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    fence: tuple[str, int] | None = None

    def section_index(path: list[str], level: int, line: int) -> int:
        key = tuple(path)
        if key not in section_by_path:
            section_by_path[key] = len(sections)
            sections.append({
                "title": path[-1] if path else "",
                "level": level,
                "headingPath": list(path),
                "line": line,
                "itemRefs": [],
            })
        return section_by_path[key]

    current_section = section_index([], 0, 0)
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        stripped = raw_line.strip()
        fence_marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence is None and fence_marker:
            marker = fence_marker.group(1)
            fence = (marker[0], len(marker))
            continue
        if fence is not None:
            marker_char, marker_length = fence
            closing = re.fullmatch(rf"{re.escape(marker_char)}{{{marker_length},}}\s*", stripped)
            if closing:
                fence = None
            continue
        heading = _HEADING.match(raw_line)
        if heading:
            level = len(heading.group(1))
            title = _normalize_text(heading.group(2))
            headings = [(n, text) for n, text in headings if n < level]
            headings.append((level, title))
            current_section = section_index([text for _, text in headings], level, line_number)
            continue
        checkbox = _CHECKBOX.match(raw_line)
        if not checkbox:
            continue
        checked = checkbox.group(1).lower() == "x"
        remainder = checkbox.group(2).strip()
        attempted = _ID_PREFIX.match(remainder)
        ref: str | None = None
        stable = False
        malformed = False
        text = remainder
        if attempted and re.match(r"(?i)^W(?:$|[^A-Za-z])", attempted.group(1)):
            candidate = attempted.group(1)
            text = attempted.group(2).strip()
            if _EXPLICIT_ID.fullmatch(candidate):
                ref = candidate
                stable = True
            else:
                malformed = True
                issues.append(_issue(
                    "malformed_work_item_id",
                    f"Malformed work-item ID {candidate!r}",
                    taskRef=task_ref,
                    line=line_number,
                    workItemRef=candidate,
                ))
        else:
            ref = _legacy_ref(task_ref, [text for _, text in headings], text)
        if ref is not None:
            if ref in seen:
                issues.append(_issue(
                    "duplicate_work_item_id",
                    f"Duplicate work-item ID {ref}",
                    taskRef=task_ref,
                    line=line_number,
                    firstLine=seen[ref],
                    workItemRef=ref,
                ))
            else:
                seen[ref] = line_number
        item = {
            "ref": ref,
            "stable": stable,
            "malformed": malformed,
            "text": _normalize_text(text),
            "checked": checked,
            "planState": "done" if checked else "pending",
            "line": line_number,
            "sectionIndex": current_section,
            "headingPath": [text for _, text in headings],
        }
        items.append(item)
        if ref is not None:
            sections[current_section]["itemRefs"].append(ref)

    counts = {
        "total": len(items),
        "done": sum(1 for item in items if item["checked"]),
        "pending": sum(1 for item in items if not item["checked"]),
        "stable": sum(1 for item in items if item["stable"]),
        "legacy": sum(1 for item in items if item["ref"] and not item["stable"]),
    }
    return {
        "valid": not issues,
        "sections": sections,
        "items": items,
        "counts": counts,
        "issues": issues,
    }


def _read_task(path: Path, location: str) -> dict[str, Any] | None:
    try:
        data = json.loads((path / "task.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    children = data.get("children", [])
    return {
        "taskRef": path.name,
        "path": path,
        "location": location,
        "id": data.get("id") or data.get("name") or path.name,
        "title": data.get("title") or data.get("name") or path.name,
        "status": data.get("status", "unknown"),
        "parent": data.get("parent") if isinstance(data.get("parent"), str) else None,
        "children": [child for child in children if isinstance(child, str)] if isinstance(children, list) else [],
        "raw": data,
    }


def discover_tasks(repo_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tasks_dir = repo_root.resolve() / ".trellis" / "tasks"
    tasks: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    candidates: list[tuple[Path, str]] = []
    if tasks_dir.is_dir():
        candidates.extend((path, "active") for path in sorted(tasks_dir.iterdir()) if path.is_dir() and path.name != "archive")
        archive = tasks_dir / "archive"
        if archive.is_dir():
            for month in sorted(archive.iterdir()):
                if month.is_dir():
                    candidates.extend((path, f"archive/{month.name}") for path in sorted(month.iterdir()) if path.is_dir())
    for path, location in candidates:
        task = _read_task(path, location)
        if task is None:
            issues.append(_issue("malformed_task", f"Cannot read {path / 'task.json'}", taskRef=path.name))
            continue
        ref = task["taskRef"]
        if ref in tasks:
            issues.append(_issue("duplicate_task_ref", f"Duplicate task directory name {ref}", taskRef=ref))
            continue
        tasks[ref] = task
    return tasks, issues


def build_task_tree(repo_root: Path) -> dict[str, Any]:
    tasks, issues = discover_tasks(repo_root)
    for ref in sorted(tasks):
        node = tasks[ref]
        for child in node["children"]:
            if child not in tasks:
                issues.append(_issue("missing_child", f"Task {ref} references missing child {child}", taskRef=ref, childRef=child))
    colors: dict[str, int] = {}
    reported_cycles: set[tuple[str, ...]] = set()

    def visit(ref: str, stack: list[str]) -> None:
        state = colors.get(ref, 0)
        if state == 2:
            return
        if state == 1:
            start = stack.index(ref) if ref in stack else 0
            cycle = tuple(stack[start:] + [ref])
            canonical = min(tuple(cycle[i:-1] + cycle[:i] + (cycle[i],)) for i in range(len(cycle) - 1))
            if canonical not in reported_cycles:
                reported_cycles.add(canonical)
                issues.append(_issue("task_tree_cycle", "Task tree contains a cycle", taskRef=ref, cycle=list(cycle)))
            return
        colors[ref] = 1
        for child in tasks[ref]["children"]:
            if child in tasks:
                visit(child, stack + [ref])
        colors[ref] = 2

    for ref in sorted(tasks):
        visit(ref, [])
    roots = sorted(ref for ref, task in tasks.items() if not task["parent"] or task["parent"] not in tasks)
    nodes = {
        ref: {
            "taskRef": ref,
            "title": task["title"],
            "status": task["status"],
            "parent": task["parent"],
            "children": list(task["children"]),
            "location": task["location"],
        }
        for ref, task in sorted(tasks.items())
    }
    return {"valid": not issues, "roots": roots, "nodes": nodes, "issues": issues}


def resolve_active_task_dir(repo_root: Path, task_ref: str) -> Path | None:
    tasks, _ = discover_tasks(repo_root)
    task = tasks.get(task_ref)
    if task is None or task["location"] != "active":
        return None
    return task["path"]
