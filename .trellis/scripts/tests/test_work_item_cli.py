from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

TASK_PY = Path(__file__).resolve().parents[1] / "task.py"


class WorkItemCliTests(unittest.TestCase):
    def _repo(self, root: Path) -> None:
        task = root / ".trellis/tasks/task"
        task.mkdir(parents=True)
        (task / "task.json").write_text(json.dumps({
            "id": "task", "title": "Task", "status": "in_progress",
            "parent": None, "children": [],
        }), encoding="utf-8")
        (task / "prd.md").write_text("# Task\n", encoding="utf-8")
        (task / "implement.md").write_text("## Work\n- [ ] [W-001] Implement.\n", encoding="utf-8")
        sessions = root / ".trellis/.runtime/sessions"
        sessions.mkdir(parents=True)
        (sessions / "pi_cli.json").write_text(
            json.dumps({"current_task": ".trellis/tasks/task"}), encoding="utf-8"
        )

    def _run(self, root: Path, *args: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["python3", str(TASK_PY), *args], cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "TRELLIS_CONTEXT_ID": "pi_cli"},
        )
        if ok and result.returncode != 0:
            self.fail(f"command failed: {result.stderr}\n{result.stdout}")
        return result

    def test_dashboard_json_and_cursor_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            selected = json.loads(self._run(
                root, "work-item", "select", "--context", "pi_cli",
                "--task", "task", "--item", "W-001", "--assignment", "a-cli",
                "--role", "primary", "--executor-kind", "main",
                "--session-id", "session-file", "--agent", "main",
                "--next-action", "write code",
            ).stdout)
            self.assertEqual(selected["assignment"]["assignmentId"], "a-cli")
            self._run(root, "work-item", "update", "--context", "pi_cli",
                      "--assignment", "a-cli", "--state", "verifying",
                      "--next-action", "run tests")
            self._run(root, "work-item", "evidence", "--context", "pi_cli",
                      "--assignment", "a-cli", "--evidence-kind", "test",
                      "--ref", "unittest:cli", "--summary", "passed")
            dashboard = json.loads(self._run(
                root, "dashboard", "--json", "--context", "pi_cli"
            ).stdout)
            self.assertEqual(dashboard["schemaVersion"], 1)
            self.assertEqual(dashboard["selected"]["taskRef"], "task")
            self.assertEqual(dashboard["execution"]["assignments"][0]["declaredState"], "verifying")
            self._run(root, "work-item", "release", "--context", "pi_cli",
                      "--assignment", "a-cli")
            after = json.loads(self._run(
                root, "dashboard", "--json", "--context", "pi_cli"
            ).stdout)
            self.assertEqual(after["execution"]["assignments"], [])

    def test_block_rejects_non_blocking_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            self._run(
                root, "work-item", "select", "--context", "pi_cli",
                "--task", "task", "--item", "W-001", "--assignment", "a-cli",
                "--role", "primary", "--executor-kind", "main",
                "--session-id", "session-file", "--agent", "main",
            )
            result = self._run(
                root, "work-item", "block", "--context", "pi_cli",
                "--assignment", "a-cli", "--state", "working",
                "--blocker", "approval", ok=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("blocking state", result.stderr)

    def test_approval_cli_rejects_receipt_after_artifact_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            requested = json.loads(self._run(
                root, "work-item", "request-approval", "--context", "pi_cli",
                "--task", "task", "--request-id", "request-1",
                "--session-id", "session-file", "--approval-kind", "implementation",
                "--scope", ".trellis/scripts", "--exclusion", "pi-app",
                "--validation-command", "python3 -m unittest",
            ).stdout)
            self.assertEqual(requested["approvalRequest"]["kind"], "implementation")
            self._run(
                root, "work-item", "record-approval", "--context", "pi_cli",
                "--request-id", "request-1", "--receipt-id", "receipt-1",
                "--decision", "approve",
            )
            valid = json.loads(self._run(
                root, "work-item", "validate-approval", "--context", "pi_cli",
                "--request-id", "request-1", "--receipt-id", "receipt-1",
            ).stdout)
            self.assertTrue(valid["authorized"])
            plan = root / ".trellis/tasks/task/implement.md"
            plan.write_text(plan.read_text(encoding="utf-8") + "\n- [ ] changed\n", encoding="utf-8")
            invalid = json.loads(self._run(
                root, "work-item", "validate-approval", "--context", "pi_cli",
                "--request-id", "request-1", "--receipt-id", "receipt-1",
            ).stdout)
            self.assertFalse(invalid["authorized"])
            self.assertIn("artifact_hash_mismatch", [i["code"] for i in invalid["issues"]])

    def test_record_approval_is_terminal_and_rejects_a_second_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            self._run(
                root, "work-item", "request-approval", "--context", "pi_cli",
                "--task", "task", "--request-id", "request-terminal",
                "--session-id", "session-file", "--approval-kind", "implementation",
                "--scope", ".trellis/scripts", "--exclusion", "commit",
            )
            self._run(
                root, "work-item", "record-approval", "--context", "pi_cli",
                "--request-id", "request-terminal", "--receipt-id", "receipt-decline",
                "--decision", "decline",
            )

            rejected = self._run(
                root, "work-item", "record-approval", "--context", "pi_cli",
                "--request-id", "request-terminal", "--receipt-id", "receipt-late-approve",
                "--decision", "approve", ok=False,
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("already answered", rejected.stderr)
            runtime = json.loads((root / ".trellis/.runtime/execution/pi_cli.json").read_text(encoding="utf-8"))
            self.assertEqual([receipt["decision"] for receipt in runtime["approvalReceipts"]], ["decline"])
            self.assertEqual(runtime["approvalRequests"][0]["status"], "decline")

    def test_record_approval_rejects_changed_artifacts_without_persisting_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            self._run(
                root, "work-item", "request-approval", "--context", "pi_cli",
                "--task", "task", "--request-id", "request-stale",
                "--session-id", "session-file", "--approval-kind", "implementation",
                "--scope", ".trellis/scripts", "--exclusion", "commit",
            )
            plan = root / ".trellis/tasks/task/implement.md"
            plan.write_text(plan.read_text(encoding="utf-8") + "\n- [ ] [W-002] Changed.\n", encoding="utf-8")

            rejected = self._run(
                root, "work-item", "record-approval", "--context", "pi_cli",
                "--request-id", "request-stale", "--receipt-id", "receipt-stale",
                "--decision", "approve", ok=False,
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("artifact_hash_mismatch", rejected.stderr)
            runtime = json.loads((root / ".trellis/.runtime/execution/pi_cli.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime["approvalReceipts"], [])
            self.assertEqual(runtime["approvalRequests"][0]["status"], "pending")

    def test_dashboard_ignores_invalid_utf8_task_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            pointer = root / ".trellis/.runtime/sessions/pi_cli.json"
            pointer.write_bytes(b"{\xff}")
            dashboard = json.loads(self._run(
                root, "dashboard", "--json", "--context", "pi_cli"
            ).stdout)
            self.assertIsNone(dashboard["selected"]["taskRef"])

    def test_dashboard_does_not_follow_or_rebind_escaping_task_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            pointer = root / ".trellis/.runtime/sessions/pi_cli.json"
            pointer.write_text(json.dumps({"current_task": "../../outside/task"}), encoding="utf-8")
            dashboard = json.loads(self._run(
                root, "dashboard", "--json", "--context", "pi_cli"
            ).stdout)
            self.assertIsNone(dashboard["selected"]["taskRef"])

    def test_dashboard_rejects_context_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            result = self._run(
                root, "dashboard", "--json", "--context", "../escape", ok=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid context key", result.stderr)

    def test_malformed_plan_mutation_fails_without_overwriting_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._repo(root)
            plan = root / ".trellis/tasks/task/implement.md"
            plan.write_text("- [ ] [W-1] malformed\n", encoding="utf-8")
            result = self._run(
                root, "work-item", "select", "--context", "pi_cli",
                "--task", "task", "--item", "W-1", "--role", "primary",
                "--executor-kind", "main", "--session-id", "session-file",
                "--agent", "main", ok=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("plan is invalid", result.stderr)
            self.assertFalse((root / ".trellis/.runtime/execution/pi_cli.json").exists())


if __name__ == "__main__":
    unittest.main()
