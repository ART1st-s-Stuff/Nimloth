"""Versioned read-only dashboard projection for Trellis consumers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .active_task import resolve_task_ref
from .approvals import artifact_review_package, root_fingerprint
from .execution import ExecutionStore, ContractError, SCHEMA_VERSION as EXECUTION_SCHEMA_VERSION, STALE_AFTER_SECONDS, HEARTBEAT_INTERVAL_SECONDS, validate_context_key
from .work_items import build_task_tree, discover_tasks, parse_implement_markdown

DASHBOARD_SCHEMA_VERSION = 1


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _selected_task(repo_root: Path, context_key: str | None) -> str | None:
    if not context_key:
        return None
    pointer = repo_root / ".trellis/.runtime/sessions" / f"{context_key}.json"
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    ref = data.get("current_task") if isinstance(data, dict) else None
    if not isinstance(ref, str):
        return None
    resolved = resolve_task_ref(ref, repo_root)
    if resolved is None or not resolved.is_dir():
        return None
    tasks, _ = discover_tasks(repo_root)
    for task_ref, task in tasks.items():
        if task["location"] == "active" and task["path"].resolve() == resolved:
            return task_ref
    return None


def _runtime_contexts(repo_root: Path, now: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_dir = repo_root / ".trellis/.runtime/execution"
    contexts: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if not runtime_dir.is_dir():
        return contexts, issues
    for path in sorted(runtime_dir.glob("*.json")):
        try:
            projection = ExecutionStore(repo_root, path.stem).project(now=now)
        except ContractError as exc:
            issues.append({"code": "malformed_runtime", "severity": "error", "contextKey": path.stem, "message": str(exc)})
            continue
        contexts.append(projection)
        issues.extend(projection["issues"])
    return contexts, issues


def build_dashboard(repo_root: Path, *, context_key: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    root = repo_root.resolve()
    if context_key is not None:
        validate_context_key(context_key)
    moment = now or datetime.now(timezone.utc)
    tree = build_task_tree(root)
    tasks, discovery_issues = discover_tasks(root)
    contexts, runtime_issues = _runtime_contexts(root, moment)
    primary_by_item: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for context in contexts:
        for assignment in context["assignments"]:
            if (
                assignment.get("role") == "primary"
                and assignment.get("validity") == "fresh"
                and assignment.get("executor", {}).get("kind") == "main"
            ):
                key = (str(assignment.get("taskRef")), str(assignment.get("workItemRef")))
                primary_by_item.setdefault(key, []).append({
                    "contextKey": context["contextKey"],
                    "assignmentId": assignment.get("assignmentId"),
                    "sessionId": assignment.get("executor", {}).get("sessionId"),
                })
    for (task_ref, work_item_ref), executors in sorted(primary_by_item.items()):
        if len(executors) > 1:
            runtime_issues.append({
                "code": "concurrent_primary_assignments",
                "severity": "error",
                "message": "Multiple main sessions declare the same work item",
                "taskRef": task_ref,
                "workItemRef": work_item_ref,
                "executors": executors,
            })
    requests = [request for context in contexts for request in context["approvalRequests"]]
    receipts = [receipt for context in contexts for receipt in context["approvalReceipts"]]
    latest_request_by_task: dict[str, dict[str, Any]] = {}

    def approval_order(request: dict[str, Any]) -> tuple[datetime, str, str]:
        created_at = datetime.fromisoformat(str(request["createdAt"]).replace("Z", "+00:00"))
        return (
            created_at.astimezone(timezone.utc),
            str(request.get("contextKey", "")),
            str(request.get("requestId", "")),
        )

    for request in requests:
        task_ref = request.get("taskRef")
        if not isinstance(task_ref, str):
            continue
        current = latest_request_by_task.get(task_ref)
        if current is None or approval_order(request) > approval_order(current):
            latest_request_by_task[task_ref] = request
    tasks_by_ref: dict[str, dict[str, Any]] = {}
    plan_issues: list[dict[str, Any]] = []
    for ref, task in sorted(tasks.items()):
        bound = latest_request_by_task.get(ref, {}).get("artifactHashes")
        review = artifact_review_package(task["path"], bound_hashes=bound if isinstance(bound, dict) else None)
        implement_artifact = review["artifacts"]["implement.md"]
        implement = implement_artifact["raw"] if isinstance(implement_artifact.get("raw"), str) else ""
        plan = parse_implement_markdown(ref, implement)
        for issue in review.get("issues", []):
            projected_issue = {**issue, "taskRef": ref}
            if issue.get("artifact") == "implement.md":
                plan["valid"] = False
                plan["issues"].append(projected_issue)
            else:
                plan_issues.append(projected_issue)
        plan_issues.extend(plan["issues"])
        tasks_by_ref[ref] = {
            "taskRef": ref,
            "id": task["id"],
            "title": task["title"],
            "status": task["status"],
            "parent": task["parent"],
            "children": task["children"],
            "location": task["location"],
            "plan": plan,
            "review": review,
        }
    projected_requests: list[dict[str, Any]] = []
    approval_issues: list[dict[str, Any]] = []
    receipt_by_request = {
        (receipt.get("contextKey"), receipt.get("requestId")): receipt
        for receipt in receipts
    }
    for request in requests:
        task = tasks_by_ref.get(str(request.get("taskRef")))
        current_hashes = task["review"]["artifactHashes"] if task else {}
        task_source = tasks.get(str(request.get("taskRef")))
        bound_hashes = request.get("artifactHashes")
        changes = (
            artifact_review_package(task_source["path"], bound_hashes=bound_hashes)["changesSinceRequest"]
            if task_source is not None and isinstance(bound_hashes, dict)
            else {"bound": True, "changed": True, "artifacts": {}}
        )
        current = request.get("artifactHashes") == current_hashes
        projected = dict(request)
        projected["current"] = current
        projected["changes"] = changes
        projected["receipt"] = receipt_by_request.get(
            (request.get("contextKey"), request.get("requestId"))
        )
        projected_requests.append(projected)
        if not current:
            approval_issues.append({"code": "stale_approval_request", "severity": "error", "requestId": request.get("requestId"), "message": "Approval request artifact hashes no longer match"})
    issues = tree["issues"] + discovery_issues + plan_issues + runtime_issues + approval_issues
    return {
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "kind": "trellis-work-item-dashboard",
        "generatedAt": _iso(moment),
        "root": {"path": str(root), "fingerprint": root_fingerprint(root)},
        "selected": {"contextKey": context_key, "taskRef": _selected_task(root, context_key)},
        "taskTree": tree,
        "tasksByRef": tasks_by_ref,
        "execution": {
            "schemaVersion": EXECUTION_SCHEMA_VERSION,
            "heartbeatIntervalSeconds": HEARTBEAT_INTERVAL_SECONDS,
            "staleAfterSeconds": STALE_AFTER_SECONDS,
            "contexts": contexts,
            "assignments": [assignment for context in contexts for assignment in context["assignments"]],
        },
        "approvalRequests": projected_requests,
        "issues": issues,
        "valid": not issues,
    }
