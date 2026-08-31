from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from common.approvals import (
    artifact_review_package,
    create_approval_receipt,
    create_approval_request,
    validate_approval_receipt,
)
from common.dashboard import build_dashboard
from common.execution import ExecutionStore

NOW = datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "dashboard-v1.json"


class ApprovalTests(unittest.TestCase):
    def _task(self, root: Path) -> Path:
        task = root / ".trellis/tasks/task"
        task.mkdir(parents=True)
        (task / "task.json").write_text(json.dumps({
            "id": "task", "title": "Task", "status": "planning",
            "parent": None, "children": [],
        }), encoding="utf-8")
        (task / "prd.md").write_text("# PRD\n\n## Scope\n\n- src/\n", encoding="utf-8")
        (task / "design.md").write_text("# Design\n", encoding="utf-8")
        (task / "implement.md").write_text("# Plan\n\n- [ ] [W-001] Build.\n", encoding="utf-8")
        return task

    def test_artifact_hash_uses_exact_bytes_and_preserves_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task = self._task(Path(td))
            payload = b"# Plan\r\n\r\n- [ ] [W-001] Build.\r\n"
            (task / "implement.md").write_bytes(payload)
            review = artifact_review_package(task)
            artifact = review["artifacts"]["implement.md"]
            self.assertEqual(artifact["raw"], payload.decode("utf-8"))
            self.assertEqual(artifact["sha256"], hashlib.sha256(payload).hexdigest())

    def test_receipt_is_bound_to_identity_kind_scope_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = self._task(root)
            review = artifact_review_package(task)
            request = create_approval_request(
                root=root, context_key="pi_session", session_id="session-file",
                request_id="request-1", task_ref="task", kind="implementation",
                artifact_hashes=review["artifactHashes"], scope=[".trellis/scripts"],
                exclusions=["pi-app"], validation_commands=["python3 -m unittest"],
                created_at=NOW,
            )
            receipt = create_approval_receipt(
                request, decision="approve", receipt_id="receipt-1", responded_at=NOW,
            )
            valid = validate_approval_receipt(
                request=request, receipt=receipt, current_artifact_hashes=review["artifactHashes"],
                root=root, context_key="pi_session", task_ref="task", kind="implementation",
            )
            self.assertTrue(valid["authorized"])
            self.assertEqual(receipt["validationCommands"], request["validationCommands"])
            tampered = dict(receipt)
            tampered["validationCommands"] = ["python3 -c 'pass'"]
            mismatched = validate_approval_receipt(
                request=request, receipt=tampered, current_artifact_hashes=review["artifactHashes"],
                root=root, context_key="pi_session", task_ref="task", kind="implementation",
            )
            self.assertFalse(mismatched["authorized"])
            self.assertIn("approval_receipt_mismatch", [i["code"] for i in mismatched["issues"]])
            (task / "implement.md").write_text("# Plan\n\n- [ ] [W-001] Changed.\n", encoding="utf-8")
            changed = artifact_review_package(task)
            invalid = validate_approval_receipt(
                request=request, receipt=receipt, current_artifact_hashes=changed["artifactHashes"],
                root=root, context_key="pi_session", task_ref="task", kind="implementation",
            )
            self.assertFalse(invalid["authorized"])
            self.assertIn("artifact_hash_mismatch", [i["code"] for i in invalid["issues"]])


class DashboardFixtureTests(unittest.TestCase):
    def test_approval_requests_are_correlated_by_context_and_latest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / ".trellis/tasks/task"
            task.mkdir(parents=True)
            (task / "task.json").write_text(json.dumps({
                "id": "task", "title": "Task", "status": "planning",
                "parent": None, "children": [],
            }), encoding="utf-8")
            (task / "prd.md").write_text("# Task\n", encoding="utf-8")
            (task / "design.md").write_text("# Design\n", encoding="utf-8")
            plan = task / "implement.md"
            plan.write_text("- [ ] [W-001] First.\n", encoding="utf-8")

            older_review = artifact_review_package(task)
            older = create_approval_request(
                root=root, context_key="pi_z", session_id="session-z",
                request_id="shared-request", task_ref="task", kind="implementation",
                artifact_hashes=older_review["artifactHashes"], scope=["older"],
                exclusions=["commit"], validation_commands=["old-check"], created_at=NOW,
            )
            # Lexicographically later but chronologically older than the Zulu request below.
            older["createdAt"] = "2026-08-31T09:00:00+02:00"
            older_store = ExecutionStore(root, "pi_z", now=lambda: NOW)
            older_store.add_approval_request(older)
            older_store.add_approval_receipt(create_approval_receipt(
                older, decision="decline", receipt_id="receipt-z", responded_at=NOW,
            ))

            plan.write_text("- [ ] [W-001] Current.\n", encoding="utf-8")
            newer_review = artifact_review_package(task)
            newer = create_approval_request(
                root=root, context_key="pi_a", session_id="session-a",
                request_id="shared-request", task_ref="task", kind="implementation",
                artifact_hashes=newer_review["artifactHashes"], scope=["newer"],
                exclusions=["commit"], validation_commands=["new-check"],
                created_at=NOW + timedelta(seconds=1),
            )
            ExecutionStore(root, "pi_a", now=lambda: NOW).add_approval_request(newer)

            dashboard = build_dashboard(root, context_key="pi_a", now=NOW + timedelta(seconds=1))
            by_context = {request["contextKey"]: request for request in dashboard["approvalRequests"]}
            self.assertIsNone(by_context["pi_a"]["receipt"])
            self.assertEqual(by_context["pi_z"]["receipt"]["receiptId"], "receipt-z")
            self.assertTrue(by_context["pi_z"]["changes"]["changed"])
            self.assertEqual(
                by_context["pi_z"]["changes"]["artifacts"]["implement.md"],
                "modified",
            )
            self.assertFalse(by_context["pi_a"]["changes"]["changed"])
            self.assertFalse(dashboard["tasksByRef"]["task"]["review"]["changesSinceRequest"]["changed"])

    def test_unreadable_artifact_path_is_a_typed_dashboard_issue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = self._task(root)
            (task / "prd.md").unlink()
            (task / "prd.md").mkdir()
            dashboard = build_dashboard(root, now=NOW)
            self.assertFalse(dashboard["valid"])
            self.assertIn("unreadable_planning_artifact", [
                issue["code"] for issue in dashboard["issues"]
            ])

    def test_invalid_utf8_artifact_is_a_typed_dashboard_issue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = self._task(root)
            (task / "implement.md").write_bytes(b"# Plan\n\xff\n")
            dashboard = build_dashboard(root, now=NOW)
            self.assertFalse(dashboard["valid"])
            self.assertIn("unreadable_planning_artifact", [
                issue["code"] for issue in dashboard["issues"]
            ])

    def _task(self, root: Path) -> Path:
        task = root / ".trellis/tasks/task"
        task.mkdir(parents=True)
        (task / "task.json").write_text(json.dumps({
            "id": "task", "title": "Task", "status": "planning",
            "parent": None, "children": [],
        }), encoding="utf-8")
        (task / "prd.md").write_text("# Task\n", encoding="utf-8")
        (task / "implement.md").write_text("- [ ] [W-001] Work.\n", encoding="utf-8")
        return task

    def test_cross_context_primary_conflict_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / ".trellis/tasks/task"
            task.mkdir(parents=True)
            (task / "task.json").write_text(json.dumps({
                "id": "task", "title": "Task", "status": "in_progress",
                "parent": None, "children": [],
            }), encoding="utf-8")
            (task / "implement.md").write_text("- [ ] [W-001] Work.\n", encoding="utf-8")
            for context in ("pi_one", "pi_two"):
                ExecutionStore(root, context, now=lambda: NOW).select(
                    task_ref="task", work_item_ref="W-001", role="primary",
                    executor={"kind": "main", "sessionId": context, "agent": "main"},
                )
            dashboard = build_dashboard(root, context_key="pi_one", now=NOW)
            self.assertFalse(dashboard["valid"])
            self.assertIn("concurrent_primary_assignments", [i["code"] for i in dashboard["issues"]])

    def test_dashboard_v1_fixture(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for task_ref, task_data in fixture["tasks"].items():
                task = root / ".trellis/tasks" / task_ref
                task.mkdir(parents=True)
                (task / "task.json").write_text(json.dumps(task_data["taskJson"]), encoding="utf-8")
                for name, raw in task_data["artifacts"].items():
                    (task / name).write_text(raw, encoding="utf-8")
            store = ExecutionStore(root, "pi_fixture", now=lambda: NOW)
            assignment = store.select(
                task_ref="child", work_item_ref="W-010", role="primary",
                executor={"kind": "main", "sessionId": "session-file", "agent": "main"},
                assignment_id="a-fixture", next_action="run fixture test",
            )
            review = artifact_review_package(root / ".trellis/tasks/child")
            request = create_approval_request(
                root=root, context_key="pi_fixture", session_id="session-file",
                request_id="request-fixture", task_ref="child", kind="implementation",
                artifact_hashes=review["artifactHashes"], scope=[".trellis/scripts"],
                exclusions=["pi-app"], validation_commands=["python3 -m unittest"], created_at=NOW,
            )
            store.add_approval_request(request)
            sessions = root / ".trellis/.runtime/sessions"
            sessions.mkdir(parents=True)
            (sessions / "pi_fixture.json").write_text(
                json.dumps({"current_task": ".trellis/tasks/child"}), encoding="utf-8"
            )
            dashboard = build_dashboard(root, context_key="pi_fixture", now=NOW)
            approval_requests = dashboard["approvalRequests"]
            for projected in approval_requests:
                projected["rootFingerprint"] = "<ROOT_FINGERPRINT>"
            observed = {
                "schemaVersion": dashboard["schemaVersion"],
                "kind": dashboard["kind"],
                "selected": dashboard["selected"],
                "treeRoots": dashboard["taskTree"]["roots"],
                "childItems": dashboard["tasksByRef"]["child"]["plan"]["items"],
                "childReview": dashboard["tasksByRef"]["child"]["review"],
                "assignments": dashboard["execution"]["assignments"],
                "approvalRequests": approval_requests,
                "issues": dashboard["issues"],
            }
            self.assertEqual(observed, fixture["expected"])
            self.assertEqual(assignment["assignmentId"], "a-fixture")


if __name__ == "__main__":
    unittest.main()
