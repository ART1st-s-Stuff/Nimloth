"""Session-scoped Trellis work-item assignment runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .approvals import ARTIFACT_NAMES, APPROVAL_KINDS, DECISIONS, artifact_review_package, root_fingerprint, validate_approval_receipt
from .work_items import parse_implement_markdown, resolve_active_task_dir

SCHEMA_VERSION = 1
STALE_AFTER_SECONDS = 30
HEARTBEAT_INTERVAL_SECONDS = 10
LIVE_STATES = {"working", "verifying", "delegated"}
DECLARED_STATES = LIVE_STATES | {"waiting_human", "waiting_external", "blocked", "failed"}
EVIDENCE_KINDS = {"artifact", "test", "command", "commit", "job", "approval", "url"}
_TRANSITIONS = {
    "working": {"working", "verifying", "delegated", "waiting_human", "waiting_external", "blocked", "failed"},
    "verifying": {"verifying", "working"},
    "delegated": {"delegated", "working"},
    "waiting_human": {"waiting_human", "working"},
    "waiting_external": {"waiting_external", "working"},
    "blocked": {"blocked", "working"},
    "failed": {"failed", "working"},
}
_CONTEXT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}
_SECRET = re.compile(r"(?i)(api[_-]?key|password|passwd|secret|token)\s*[:=]|BEGIN [A-Z ]*PRIVATE KEY")


class ContractError(RuntimeError):
    pass


def validate_context_key(context_key: str) -> str:
    device_stem = context_key.split(".", 1)[0].upper() if isinstance(context_key, str) else ""
    if (
        not isinstance(context_key, str)
        or not _CONTEXT_KEY.fullmatch(context_key)
        or context_key.endswith(".")
        or device_stem in _WINDOWS_RESERVED_NAMES
    ):
        raise ContractError("invalid context key")
    return context_key


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _bounded(name: str, value: Any, maximum: int, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > maximum:
        raise ContractError(f"{name} must be a string up to {maximum} characters")
    return value.strip()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                # Some platforms do not support fsync on directory handles.
                pass
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class ExecutionStore:
    def __init__(self, repo_root: Path, context_key: str, *, now: Callable[[], datetime] | None = None):
        self.root = repo_root.resolve()
        self.context_key = validate_context_key(context_key)
        self.path = self.root / ".trellis" / ".runtime" / "execution" / f"{context_key}.json"
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _empty(self) -> dict[str, Any]:
        now = _iso(self._now())
        return {
            "schemaVersion": SCHEMA_VERSION,
            "rootFingerprint": root_fingerprint(self.root),
            "contextKey": self.context_key,
            "updatedAt": now,
            "session": {"active": True, "updatedAt": now},
            "assignments": [],
            "approvalRequests": [],
            "approvalReceipts": [],
        }

    def _validate_state(self, data: dict[str, Any]) -> None:
        top_fields = {
            "schemaVersion", "rootFingerprint", "contextKey", "updatedAt",
            "session", "assignments", "approvalRequests", "approvalReceipts",
        }
        extra = set(data) - top_fields
        if extra:
            raise ContractError(f"runtime has unexpected fields: {sorted(extra)}")
        if _parse_time(data.get("updatedAt")) is None:
            raise ContractError("runtime updatedAt is invalid")
        session = data.get("session")
        if (
            not isinstance(session, dict)
            or set(session) != {"active", "updatedAt"}
            or not isinstance(session.get("active"), bool)
            or _parse_time(session.get("updatedAt")) is None
        ):
            raise ContractError("runtime session metadata is invalid")

        assignment_fields = {
            "assignmentId", "role", "taskRef", "workItemRef", "declaredState",
            "executor", "since", "updatedAt", "heartbeatAt", "blocker",
            "nextAction", "evidence", "observedActivity", "releasedAt", "releaseReason",
        }
        executor_fields = {"kind", "sessionId", "agent", "runId", "toolCallId"}
        assignment_ids: set[str] = set()
        for assignment in data["assignments"]:
            if not isinstance(assignment, dict):
                raise ContractError("runtime assignment must be an object")
            extra = set(assignment) - assignment_fields
            if extra:
                raise ContractError(f"runtime assignment has unexpected fields: {sorted(extra)}")
            identifier = assignment.get("assignmentId")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", identifier):
                raise ContractError("runtime assignment id is invalid")
            if identifier in assignment_ids:
                raise ContractError("runtime assignment ids must be unique")
            assignment_ids.add(identifier)
            if assignment.get("role") not in {"primary", "delegated"}:
                raise ContractError("runtime assignment role is invalid")
            declared_state = assignment.get("declaredState")
            if declared_state not in DECLARED_STATES:
                raise ContractError("runtime assignment declared state is invalid")
            for field in ("taskRef", "workItemRef"):
                value = assignment.get(field)
                if not isinstance(value, str) or not value or len(value) > 512:
                    raise ContractError(f"runtime assignment {field} is invalid")
            executor = assignment.get("executor")
            if (
                not isinstance(executor, dict)
                or set(executor) - executor_fields
                or executor.get("kind") not in {"main", "subagent"}
            ):
                raise ContractError("runtime assignment executor is invalid")
            for field, maximum in (("sessionId", 512), ("agent", 128)):
                value = executor.get(field)
                if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                    raise ContractError("runtime assignment executor identity is invalid")
            for field in ("runId", "toolCallId"):
                value = executor.get(field)
                if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 512):
                    raise ContractError("runtime assignment executor identity is invalid")
            for field in ("since", "updatedAt", "heartbeatAt"):
                if _parse_time(assignment.get(field)) is None:
                    raise ContractError(f"runtime assignment {field} is invalid")
            released_at = assignment.get("releasedAt")
            if released_at is not None and _parse_time(released_at) is None:
                raise ContractError("runtime assignment releasedAt is invalid")
            for field, maximum in (("blocker", 200), ("nextAction", 500)):
                value = assignment.get(field)
                if value is not None and (not isinstance(value, str) or len(value) > maximum):
                    raise ContractError(f"runtime assignment {field} is invalid")
            if declared_state in {"waiting_human", "waiting_external", "blocked"} and not str(assignment.get("blocker") or "").strip():
                raise ContractError("runtime blocking assignment requires a blocker")
            release_reason = assignment.get("releaseReason")
            if released_at is not None and (not isinstance(release_reason, str) or not release_reason.strip() or len(release_reason) > 200):
                raise ContractError("runtime assignment releaseReason is invalid")
            if released_at is None and release_reason is not None:
                raise ContractError("active runtime assignment cannot have a releaseReason")
            evidence = assignment.get("evidence")
            if not isinstance(evidence, list) or len(evidence) > 20:
                raise ContractError("runtime assignment evidence is invalid")
            for item in evidence:
                if not isinstance(item, dict) or set(item) != {"kind", "ref", "summary", "at"} or item.get("kind") not in EVIDENCE_KINDS:
                    raise ContractError("runtime evidence kind is invalid")
                ref = item.get("ref")
                summary = item.get("summary")
                if not isinstance(ref, str) or not ref or len(ref) > 512 or not isinstance(summary, str) or not summary or len(summary) > 200:
                    raise ContractError("runtime evidence payload is invalid")
                if _SECRET.search(f"{ref}\n{summary}") or _parse_time(item.get("at")) is None:
                    raise ContractError("runtime evidence is unsafe or malformed")
            observed = assignment.get("observedActivity")
            if observed is not None:
                if (
                    not isinstance(observed, dict)
                    or set(observed) != {"toolName", "toolCallId", "status", "at"}
                    or observed.get("status") not in {"running", "succeeded", "failed", "update"}
                    or _parse_time(observed.get("at")) is None
                ):
                    raise ContractError("runtime observed activity is invalid")
                tool_name = observed.get("toolName")
                tool_call_id = observed.get("toolCallId")
                if not isinstance(tool_name, str) or not tool_name.strip() or len(tool_name) > 128:
                    raise ContractError("runtime observed activity is invalid")
                if tool_call_id is not None and (not isinstance(tool_call_id, str) or not tool_call_id.strip() or len(tool_call_id) > 512):
                    raise ContractError("runtime observed activity is invalid")

        request_fields = {
            "requestId", "rootFingerprint", "contextKey", "sessionId", "taskRef",
            "kind", "artifactHashes", "reviewSetHash", "scope", "exclusions",
            "validationCommands", "createdAt", "status",
        }
        request_ids: set[str] = set()
        requests: dict[str, dict[str, Any]] = {}
        for request in data["approvalRequests"]:
            if not isinstance(request, dict) or set(request) != request_fields or request.get("kind") not in APPROVAL_KINDS:
                raise ContractError("runtime approval request is invalid")
            identifier = request.get("requestId")
            if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 512 or identifier in request_ids:
                raise ContractError("runtime approval request id is invalid or duplicate")
            request_ids.add(identifier)
            requests[identifier] = request
            if request.get("rootFingerprint") != root_fingerprint(self.root) or request.get("contextKey") != self.context_key:
                raise ContractError("runtime approval request identity is invalid")
            for field in ("sessionId", "taskRef"):
                value = request.get(field)
                if not isinstance(value, str) or not value.strip() or len(value) > 512:
                    raise ContractError("runtime approval request identity is invalid")
            hashes = request.get("artifactHashes")
            if (
                not isinstance(hashes, dict)
                or any(name not in ARTIFACT_NAMES or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in hashes.items())
            ):
                raise ContractError("runtime approval request artifact hashes are invalid")
            expected_review_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if request.get("reviewSetHash") != expected_review_hash:
                raise ContractError("runtime approval request review-set hash is invalid")
            for field, maximum in (("scope", 1000), ("exclusions", 1000), ("validationCommands", 2000)):
                values = request.get(field)
                if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() and len(value) <= maximum for value in values):
                    raise ContractError("runtime approval request list payload is invalid")
            if _parse_time(request.get("createdAt")) is None or request.get("status") not in ({"pending"} | DECISIONS):
                raise ContractError("runtime approval request status is invalid")

        receipt_fields = {
            "receiptId", "requestId", "rootFingerprint", "contextKey", "sessionId",
            "taskRef", "kind", "artifactHashes", "reviewSetHash", "scope",
            "exclusions", "validationCommands", "decision", "comment", "respondedAt",
        }
        receipt_ids: set[str] = set()
        decision_by_request: dict[str, str] = {}
        identity_fields = (
            "requestId", "rootFingerprint", "contextKey", "sessionId", "taskRef",
            "kind", "artifactHashes", "reviewSetHash", "scope", "exclusions",
            "validationCommands",
        )
        for receipt in data["approvalReceipts"]:
            if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
                raise ContractError("runtime approval receipt is invalid")
            identifier = receipt.get("receiptId")
            if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 512 or identifier in receipt_ids or receipt.get("decision") not in DECISIONS:
                raise ContractError("runtime approval receipt is invalid or duplicate")
            receipt_ids.add(identifier)
            request = requests.get(str(receipt.get("requestId")))
            if request is None or any(receipt.get(field) != request.get(field) for field in identity_fields):
                raise ContractError("runtime approval receipt identity is invalid")
            if request["requestId"] in decision_by_request:
                raise ContractError("runtime approval request has more than one terminal receipt")
            comment = receipt.get("comment")
            if comment is not None and (not isinstance(comment, str) or len(comment) > 2000):
                raise ContractError("runtime approval receipt comment is invalid")
            if _parse_time(receipt.get("respondedAt")) is None:
                raise ContractError("runtime approval receipt timestamp is invalid")
            decision_by_request[request["requestId"]] = receipt["decision"]
        for request_id, request in requests.items():
            expected_status = decision_by_request.get(request_id, "pending")
            if request.get("status") != expected_status:
                raise ContractError("runtime approval request status does not match its receipts")

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"malformed runtime file: {exc}") from exc
        if not isinstance(data, dict) or data.get("schemaVersion") != SCHEMA_VERSION:
            raise ContractError("unsupported runtime schema version")
        if data.get("rootFingerprint") != root_fingerprint(self.root):
            raise ContractError("runtime root fingerprint mismatch")
        if data.get("contextKey") != self.context_key:
            raise ContractError("runtime context key mismatch")
        for key in ("assignments", "approvalRequests", "approvalReceipts"):
            if not isinstance(data.get(key), list):
                raise ContractError(f"runtime {key} must be a list")
        self._validate_state(data)
        return data

    def _write(self, state: dict[str, Any], at: datetime | None = None) -> None:
        timestamp = _iso(at or self._now())
        state["updatedAt"] = timestamp
        if state.get("schemaVersion") != SCHEMA_VERSION:
            raise ContractError("unsupported runtime schema version")
        if state.get("rootFingerprint") != root_fingerprint(self.root) or state.get("contextKey") != self.context_key:
            raise ContractError("runtime identity mismatch")
        for key in ("assignments", "approvalRequests", "approvalReceipts"):
            if not isinstance(state.get(key), list):
                raise ContractError(f"runtime {key} must be a list")
        self._validate_state(state)
        _atomic_write(self.path, state)

    def _valid_plan(self, task_ref: str) -> tuple[Path, dict[str, Any]]:
        task_dir = resolve_active_task_dir(self.root, task_ref)
        if task_dir is None:
            raise ContractError(f"unknown active task: {task_ref}")
        try:
            payload = (task_dir / "implement.md").read_bytes()
        except FileNotFoundError:
            source = ""
        except OSError as exc:
            raise ContractError(f"cannot read implement.md: {exc}") from exc
        else:
            try:
                source = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContractError(f"implement.md is not valid UTF-8: {exc}") from exc
        plan = parse_implement_markdown(task_ref, source)
        if not plan["valid"]:
            raise ContractError("plan is invalid; resolve duplicate or malformed work-item IDs")
        return task_dir, plan

    def _plan_item(self, task_ref: str, work_item_ref: str) -> tuple[Path, dict[str, Any]]:
        task_dir, plan = self._valid_plan(task_ref)
        matches = [item for item in plan["items"] if item["ref"] == work_item_ref]
        if len(matches) != 1:
            raise ContractError(f"unknown work item: {task_ref}#{work_item_ref}")
        if matches[0]["checked"]:
            raise ContractError(f"work item is already done: {task_ref}#{work_item_ref}")
        return task_dir, matches[0]

    def _validate_assignment_plan(self, assignment: dict[str, Any]) -> None:
        self._plan_item(str(assignment.get("taskRef", "")), str(assignment.get("workItemRef", "")))

    def _executor(self, executor: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(executor, dict) or executor.get("kind") not in {"main", "subagent"}:
            raise ContractError("executor kind must be main or subagent")
        result = {
            "kind": executor["kind"],
            "sessionId": _bounded("executor sessionId", executor.get("sessionId"), 512, required=True),
            "agent": _bounded("executor agent", executor.get("agent"), 128, required=True),
        }
        for key, maximum in (("runId", 512), ("toolCallId", 512)):
            value = _bounded(f"executor {key}", executor.get(key), maximum)
            if value is not None:
                result[key] = value
        return result

    def select(
        self, *, task_ref: str, work_item_ref: str, role: str,
        executor: dict[str, Any], assignment_id: str | None = None,
        next_action: str | None = None,
    ) -> dict[str, Any]:
        self._plan_item(task_ref, work_item_ref)
        if role not in {"primary", "delegated"}:
            raise ContractError("assignment role must be primary or delegated")
        executor_data = self._executor(executor)
        identifier = assignment_id or f"a-{uuid.uuid4().hex}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", identifier):
            raise ContractError("invalid assignment id")
        action = _bounded("next action", next_action, 500)
        state = self.read()
        now = _iso(self._now())
        if any(a.get("assignmentId") == identifier for a in state["assignments"]):
            raise ContractError(f"duplicate assignment id: {identifier}")
        if role == "primary":
            for current in state["assignments"]:
                if current.get("role") == "primary" and current.get("releasedAt") is None:
                    current["releasedAt"] = now
                    current["releaseReason"] = "superseded"
                    current["updatedAt"] = now
        assignment = {
            "assignmentId": identifier,
            "role": role,
            "taskRef": task_ref,
            "workItemRef": work_item_ref,
            "declaredState": "working",
            "executor": executor_data,
            "since": now,
            "updatedAt": now,
            "heartbeatAt": now,
            "blocker": None,
            "nextAction": action,
            "evidence": [],
            "observedActivity": None,
            "releasedAt": None,
        }
        state["session"] = {"active": True, "updatedAt": now}
        state["assignments"].append(assignment)
        self._write(state)
        return assignment

    def _active_assignment(self, state: dict[str, Any], assignment_id: str) -> dict[str, Any]:
        for assignment in state["assignments"]:
            if assignment.get("assignmentId") == assignment_id and assignment.get("releasedAt") is None:
                return assignment
        raise ContractError(f"unknown active assignment: {assignment_id}")

    def update(self, assignment_id: str, declared_state: str, *, blocker: str | None = None, next_action: str | None = None) -> dict[str, Any]:
        if declared_state not in DECLARED_STATES:
            raise ContractError(f"unknown declared state: {declared_state}")
        state = self.read()
        assignment = self._active_assignment(state, assignment_id)
        self._validate_assignment_plan(assignment)
        previous = assignment.get("declaredState")
        if declared_state not in _TRANSITIONS.get(previous, set()):
            raise ContractError(f"invalid state transition: {previous} -> {declared_state}")
        blocker_value = _bounded("blocker", blocker, 200)
        if declared_state in {"waiting_human", "waiting_external", "blocked"} and not blocker_value:
            raise ContractError(f"{declared_state} requires a blocker")
        assignment["declaredState"] = declared_state
        assignment["blocker"] = blocker_value
        if next_action is not None:
            assignment["nextAction"] = _bounded("next action", next_action, 500)
        now = _iso(self._now())
        assignment["updatedAt"] = now
        if declared_state in LIVE_STATES:
            assignment["heartbeatAt"] = now
        self._write(state)
        return assignment

    def heartbeat(self, assignment_id: str, *, observed: dict[str, Any] | None = None, at: datetime | None = None) -> dict[str, Any]:
        state = self.read()
        assignment = self._active_assignment(state, assignment_id)
        self._validate_assignment_plan(assignment)
        if assignment.get("declaredState") not in LIVE_STATES:
            raise ContractError("heartbeat is only valid for live assignments")
        moment = at or self._now()
        timestamp = _iso(moment)
        if observed is not None:
            if not isinstance(observed, dict) or observed.get("status") not in {"running", "succeeded", "failed", "update"}:
                raise ContractError("invalid observed activity")
            assignment["observedActivity"] = {
                "toolName": _bounded("observed toolName", observed.get("toolName"), 128, required=True),
                "toolCallId": _bounded("observed toolCallId", observed.get("toolCallId"), 512),
                "status": observed["status"],
                "at": timestamp,
            }
        assignment["heartbeatAt"] = timestamp
        assignment["updatedAt"] = timestamp
        state["session"] = {"active": True, "updatedAt": timestamp}
        self._write(state, moment)
        return assignment

    def heartbeat_live(self, *, observed: dict[str, Any] | None = None, executor_kind: str = "main", at: datetime | None = None) -> int:
        observed_data: dict[str, Any] | None = None
        if observed is not None:
            if not isinstance(observed, dict) or observed.get("status") not in {"running", "succeeded", "failed", "update"}:
                raise ContractError("invalid observed activity")
            observed_data = {
                "toolName": _bounded("observed toolName", observed.get("toolName"), 128, required=True),
                "toolCallId": _bounded("observed toolCallId", observed.get("toolCallId"), 512),
                "status": observed["status"],
            }
        state = self.read()
        if executor_kind not in {"main", "subagent"}:
            raise ContractError("executor kind must be main or subagent")
        active = [a for a in state["assignments"] if a.get("releasedAt") is None and a.get("declaredState") in LIVE_STATES and a.get("executor", {}).get("kind") == executor_kind]
        for assignment in active:
            self._validate_assignment_plan(assignment)
        moment = at or self._now()
        timestamp = _iso(moment)
        for assignment in active:
            assignment["heartbeatAt"] = timestamp
            assignment["updatedAt"] = timestamp
            if observed_data is not None:
                assignment["observedActivity"] = {**observed_data, "at": timestamp}
        if active:
            state["session"] = {"active": True, "updatedAt": timestamp}
            self._write(state, moment)
        return len(active)

    def add_evidence(self, assignment_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(evidence, dict) or evidence.get("kind") not in EVIDENCE_KINDS:
            raise ContractError("unknown evidence kind")
        ref = _bounded("evidence ref", evidence.get("ref"), 512, required=True)
        summary = _bounded("evidence summary", evidence.get("summary"), 200, required=True)
        if _SECRET.search(f"{ref}\n{summary}"):
            raise ContractError("secret-like evidence is forbidden")
        state = self.read()
        assignment = self._active_assignment(state, assignment_id)
        self._validate_assignment_plan(assignment)
        item = {"kind": evidence["kind"], "ref": ref, "summary": summary, "at": _iso(self._now())}
        assignment["evidence"].append(item)
        if len(assignment["evidence"]) > 20:
            assignment["evidence"] = assignment["evidence"][-20:]
        assignment["updatedAt"] = item["at"]
        self._write(state)
        return item

    def release(self, assignment_id: str, *, reason: str = "released") -> dict[str, Any]:
        state = self.read()
        assignment = self._active_assignment(state, assignment_id)
        # Checked items remain releasable for checkbox-first completion, but a
        # malformed plan cannot authorize any runtime mutation.
        self._valid_plan(str(assignment.get("taskRef", "")))
        now = _iso(self._now())
        assignment["releasedAt"] = now
        assignment["releaseReason"] = _bounded("release reason", reason, 200, required=True)
        assignment["updatedAt"] = now
        self._write(state)
        return assignment

    def shutdown(self, *, at: datetime | None = None) -> None:
        state = self.read()
        moment = at or self._now()
        state["session"] = {"active": False, "updatedAt": _iso(moment)}
        self._write(state, moment)

    def resume(self, *, at: datetime | None = None) -> None:
        state = self.read()
        moment = at or self._now()
        timestamp = _iso(moment)
        state["session"] = {"active": True, "updatedAt": timestamp}
        live_assignments = [
            assignment for assignment in state["assignments"]
            if assignment.get("releasedAt") is None and assignment.get("declaredState") in LIVE_STATES
        ]
        for assignment in live_assignments:
            self._validate_assignment_plan(assignment)
        for assignment in live_assignments:
            assignment["heartbeatAt"] = timestamp
            assignment["updatedAt"] = timestamp
        self._write(state, moment)

    def add_approval_request(self, request: dict[str, Any]) -> None:
        if request.get("rootFingerprint") != root_fingerprint(self.root) or request.get("contextKey") != self.context_key:
            raise ContractError("approval request identity mismatch")
        if request.get("kind") not in APPROVAL_KINDS or request.get("status") != "pending":
            raise ContractError("invalid approval request")
        task_dir = resolve_active_task_dir(self.root, str(request.get("taskRef", "")))
        if task_dir is None:
            raise ContractError("approval request task is missing")
        current = artifact_review_package(task_dir)
        if current.get("issues"):
            raise ContractError("approval request planning artifacts are unreadable")
        if request.get("artifactHashes") != current["artifactHashes"] or request.get("reviewSetHash") != current["reviewSetHash"]:
            raise ContractError("approval request artifact hashes do not match current artifacts")
        state = self.read()
        if any(r.get("requestId") == request.get("requestId") for r in state["approvalRequests"]):
            raise ContractError("duplicate approval request id")
        state["approvalRequests"].append(request)
        self._write(state)

    def add_approval_receipt(self, receipt: dict[str, Any]) -> None:
        if receipt.get("rootFingerprint") != root_fingerprint(self.root) or receipt.get("contextKey") != self.context_key or receipt.get("decision") not in DECISIONS:
            raise ContractError("invalid approval receipt")
        state = self.read()
        request = next((r for r in state["approvalRequests"] if r.get("requestId") == receipt.get("requestId")), None)
        if request is None:
            raise ContractError("approval receipt has no matching request")
        if request.get("status") != "pending" or any(
            item.get("requestId") == request.get("requestId")
            for item in state["approvalReceipts"]
        ):
            raise ContractError("approval request was already answered")
        task_dir = resolve_active_task_dir(self.root, request["taskRef"])
        if task_dir is None:
            raise ContractError("approval request task is missing")
        current = artifact_review_package(task_dir)["artifactHashes"]
        result = validate_approval_receipt(
            request=request, receipt=receipt, current_artifact_hashes=current,
            root=self.root, context_key=self.context_key,
            task_ref=request["taskRef"], kind=request["kind"],
        )
        if receipt.get("decision") == "approve" and not result["authorized"]:
            raise ContractError("approval receipt is invalid for current artifacts")
        state["approvalReceipts"].append(receipt)
        request["status"] = receipt["decision"]
        self._write(state)

    def project(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or self._now()
        state = self.read()
        projected: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        session_active = state.get("session", {}).get("active") is True
        for assignment in state["assignments"]:
            if assignment.get("releasedAt") is not None:
                continue
            item = dict(assignment)
            task_dir = resolve_active_task_dir(self.root, str(item.get("taskRef", "")))
            plan_item = None
            plan_valid = False
            if task_dir is not None:
                try:
                    payload = (task_dir / "implement.md").read_bytes()
                    source = payload.decode("utf-8")
                except FileNotFoundError:
                    source = ""
                except (OSError, UnicodeDecodeError):
                    source = ""
                else:
                    plan = parse_implement_markdown(item["taskRef"], source)
                    plan_valid = plan["valid"]
                    plan_item = next((candidate for candidate in plan["items"] if candidate["ref"] == item.get("workItemRef")), None)
            if task_dir is None or not plan_valid or plan_item is None:
                validity = "orphan"
                issues.append({"code": "orphan_assignment", "severity": "error", "message": "Assignment references a missing or invalid task/work item", "contextKey": self.context_key, "assignmentId": item.get("assignmentId")})
            elif plan_item["checked"]:
                validity = "conflict"
                issues.append({"code": "checkbox_runtime_conflict", "severity": "error", "message": "Done checkbox still has an active runtime assignment", "contextKey": self.context_key, "assignmentId": item.get("assignmentId")})
            elif item.get("declaredState") in LIVE_STATES:
                heartbeat = _parse_time(item.get("heartbeatAt"))
                age = (moment - heartbeat).total_seconds() if heartbeat else STALE_AFTER_SECONDS + 1
                validity = "fresh" if session_active and age <= STALE_AFTER_SECONDS else "stale"
                if validity == "stale":
                    issues.append({"code": "stale_assignment", "severity": "error", "message": "Live assignment heartbeat expired or session shut down", "contextKey": self.context_key, "assignmentId": item.get("assignmentId")})
            else:
                validity = "fresh"
            item["validity"] = validity
            projected.append(item)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "contextKey": self.context_key,
            "session": state["session"],
            "assignments": projected,
            "approvalRequests": state["approvalRequests"],
            "approvalReceipts": state["approvalReceipts"],
            "issues": issues,
        }
