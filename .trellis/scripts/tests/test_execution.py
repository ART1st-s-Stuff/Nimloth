from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from common.execution import (
    ContractError,
    ExecutionStore,
    STALE_AFTER_SECONDS,
)

NOW = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)


class ExecutionStoreTests(unittest.TestCase):
    def _repo(self, root: Path) -> None:
        task = root / ".trellis" / "tasks" / "task"
        task.mkdir(parents=True)
        (task / "task.json").write_text(json.dumps({
            "id": "task", "title": "Task", "status": "in_progress",
            "parent": None, "children": [],
        }), encoding="utf-8")
        (task / "implement.md").write_text(
            "## Work\n- [ ] [W-001] pending\n- [x] [W-002] done\n",
            encoding="utf-8",
        )

    def test_atomic_select_update_evidence_release_never_edits_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            plan_before = (root / ".trellis/tasks/task/implement.md").read_bytes()
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            assignment = store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
                next_action="write parser",
            )
            runtime = root / ".trellis/.runtime/execution/pi_session.json"
            self.assertTrue(runtime.is_file())
            self.assertFalse(any(runtime.parent.glob(f".{runtime.name}.*.tmp")))
            store.update(assignment["assignmentId"], "verifying", next_action="run tests")
            store.add_evidence(assignment["assignmentId"], {
                "kind": "test", "ref": "unittest:test_parser", "summary": "passed"
            })
            store.release(assignment["assignmentId"])
            state = store.read()
            self.assertIsNotNone(state["assignments"][0]["releasedAt"])
            self.assertEqual(plan_before, (root / ".trellis/tasks/task/implement.md").read_bytes())

    def test_invalid_transition_context_and_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            for invalid_context in ("../escape", "pi.", "CON", "LPT1.log"):
                with self.subTest(context=invalid_context):
                    with self.assertRaisesRegex(ContractError, "context key"):
                        ExecutionStore(root, invalid_context)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            assignment = store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            with self.assertRaisesRegex(ContractError, "transition"):
                store.update(assignment["assignmentId"], "failed")
                store.update(assignment["assignmentId"], "verifying")
            with self.assertRaisesRegex(ContractError, "evidence summary"):
                store.add_evidence(assignment["assignmentId"], {
                    "kind": "test", "ref": "test:x", "summary": "x" * 201,
                })
            with self.assertRaisesRegex(ContractError, "secret-like"):
                store.add_evidence(assignment["assignmentId"], {
                    "kind": "command", "ref": "api_key=secret", "summary": "bad",
                })

    def test_invalid_utf8_plan_fails_closed_without_runtime_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            (root / ".trellis/tasks/task/implement.md").write_bytes(b"- [ ] [W-001] bad\xff\n")
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            with self.assertRaisesRegex(ContractError, "not valid UTF-8"):
                store.select(
                    task_ref="task", work_item_ref="W-001", role="primary",
                    executor={"kind": "main", "sessionId": "session", "agent": "main"},
                )
            self.assertFalse(store.path.exists())

    def test_done_item_is_rejected_and_plan_changes_create_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            with self.assertRaisesRegex(ContractError, "already done"):
                store.select(
                    task_ref="task", work_item_ref="W-002", role="primary",
                    executor={"kind": "main", "sessionId": "session", "agent": "main"},
                )
            assignment = store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            plan = root / ".trellis/tasks/task/implement.md"
            plan.write_text(plan.read_text(encoding="utf-8").replace("[ ] [W-001]", "[x] [W-001]"), encoding="utf-8")
            projected = store.project(now=NOW)
            current = next(a for a in projected["assignments"] if a["assignmentId"] == assignment["assignmentId"])
            self.assertEqual(current["validity"], "conflict")
            self.assertIn("checkbox_runtime_conflict", [i["code"] for i in projected["issues"]])
            released = store.release(assignment["assignmentId"])
            self.assertIsNotNone(released["releasedAt"], "checkbox-first completion must remain releasable")

    def test_heartbeat_stale_waiting_persistence_shutdown_and_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            live = store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            later = NOW + timedelta(seconds=STALE_AFTER_SECONDS + 1)
            self.assertEqual(store.project(now=later)["assignments"][0]["validity"], "stale")
            store.heartbeat(live["assignmentId"], observed={
                "toolName": "read", "toolCallId": "tool-1", "status": "running"
            }, at=later)
            self.assertEqual(store.project(now=later)["assignments"][0]["validity"], "fresh")
            store.update(live["assignmentId"], "waiting_human", blocker="approval")
            store.shutdown(at=later)
            self.assertEqual(store.project(now=later + timedelta(days=1))["assignments"][0]["validity"], "fresh")
            (root / ".trellis/tasks/task/implement.md").write_text("## Empty\n", encoding="utf-8")
            projected = store.project(now=later)
            self.assertEqual(projected["assignments"][0]["validity"], "orphan")
            self.assertIn("orphan_assignment", [i["code"] for i in projected["issues"]])

    def test_mutation_fails_closed_after_plan_becomes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            assignment = store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            before = store.path.read_bytes()
            (root / ".trellis/tasks/task/implement.md").write_text(
                "- [ ] [W-1] malformed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ContractError, "plan is invalid"):
                store.update(assignment["assignmentId"], "verifying")
            with self.assertRaisesRegex(ContractError, "plan is invalid"):
                store.heartbeat_live(observed={
                    "toolName": "read", "toolCallId": "tool-1", "status": "running",
                })
            with self.assertRaisesRegex(ContractError, "plan is invalid"):
                store.release(assignment["assignmentId"])
            self.assertEqual(store.path.read_bytes(), before)

    def test_heartbeat_live_rejects_invalid_observed_without_corrupting_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            before = store.path.read_bytes()
            with self.assertRaisesRegex(ContractError, "observed activity"):
                store.heartbeat_live(observed={
                    "toolName": "read", "toolCallId": "tool-1", "status": "invented",
                })
            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(store.read()["assignments"][0]["declaredState"], "working")

    def test_persisted_observed_activity_and_forbidden_payload_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            state = json.loads(store.path.read_text(encoding="utf-8"))
            state["assignments"][0]["observedActivity"] = {
                "toolName": "read", "status": "invented", "at": "not-a-time",
            }
            store.path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "observed activity"):
                store.read()

            state["assignments"][0]["observedActivity"] = None
            state["assignments"][0]["toolArgs"] = {"token": "must-not-be-projected"}
            store.path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unexpected fields"):
                store.read()

    def test_naive_persisted_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            state = json.loads(store.path.read_text(encoding="utf-8"))
            state["assignments"][0]["heartbeatAt"] = "2026-08-31T08:00:00"
            store.path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "heartbeatAt"):
                store.read()

    def test_malformed_approval_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            state = store.read()
            state["approvalRequests"].append({
                "requestId": "request-1",
                "rootFingerprint": state["rootFingerprint"],
                "contextKey": "pi_session",
                "sessionId": "session",
                "taskRef": "task",
                "kind": "implementation",
                "artifactHashes": {},
                "reviewSetHash": "0" * 64,
                "scope": [".trellis/scripts"],
                "exclusions": [],
                "validationCommands": [123],
                "createdAt": "2026-08-31T08:00:00Z",
                "status": "pending",
            })
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "approval request"):
                store.read()

    def test_malformed_assignment_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            store = ExecutionStore(root, "pi_session", now=lambda: NOW)
            store.select(
                task_ref="task", work_item_ref="W-001", role="primary",
                executor={"kind": "main", "sessionId": "session", "agent": "main"},
            )
            state = json.loads(store.path.read_text(encoding="utf-8"))
            state["assignments"][0]["declaredState"] = "invented"
            store.path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "declared state"):
                store.read()

    def test_root_fingerprint_and_unknown_schema_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            runtime = root / ".trellis/.runtime/execution/pi_session.json"
            runtime.parent.mkdir(parents=True)
            runtime.write_text(json.dumps({"schemaVersion": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "schema"):
                ExecutionStore(root, "pi_session").read()


if __name__ == "__main__":
    unittest.main()
