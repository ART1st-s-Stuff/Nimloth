from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from common.work_items import build_task_tree, parse_implement_markdown


class PlanParserTests(unittest.TestCase):
    def test_explicit_and_legacy_identity_are_deterministic(self) -> None:
        source = """# Plan

## Build

- [ ] [W-001] Build parser.
- [x] Legacy   item
"""
        plan = parse_implement_markdown("08-31-example", source)
        self.assertTrue(plan["valid"])
        self.assertEqual([i["ref"] for i in plan["items"][:1]], ["W-001"])
        legacy = plan["items"][1]
        payload = json.dumps(
            ["08-31-example", ["Plan", "Build"], "Legacy item"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertEqual(
            legacy["ref"],
            "legacy-" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        self.assertFalse(legacy["stable"])
        self.assertEqual(legacy["planState"], "done")
        build_section = next(section for section in plan["sections"] if section["title"] == "Build")
        self.assertEqual(build_section["headingPath"], ["Plan", "Build"])

    def test_duplicate_and_malformed_ids_fail_closed(self) -> None:
        source = """## Work
- [ ] [W-001] first
- [ ] [W-001] second
- [ ] [W-1] malformed
- [ ] [w-002] lowercase malformed
- [ ] [W_003] underscore malformed
- [ ] [W] missing suffix
"""
        plan = parse_implement_markdown("task", source)
        self.assertFalse(plan["valid"])
        self.assertEqual(
            [issue["code"] for issue in plan["issues"]],
            [
                "duplicate_work_item_id",
                "malformed_work_item_id",
                "malformed_work_item_id",
                "malformed_work_item_id",
                "malformed_work_item_id",
            ],
        )
        malformed = plan["items"][2]
        self.assertIsNone(malformed["ref"])
        self.assertTrue(malformed["malformed"])
        self.assertTrue(all(item["ref"] is None for item in plan["items"][2:]))

    def test_code_fences_are_not_plan_items(self) -> None:
        source = """## Real
````markdown
```example
- [ ] [W-999] example only
```
````
- [ ] actual
"""
        plan = parse_implement_markdown("task", source)
        self.assertEqual(len(plan["items"]), 1)
        self.assertEqual(plan["items"][0]["text"], "actual")


class TaskTreeTests(unittest.TestCase):
    def _task(self, root: Path, name: str, *, parent=None, children=(), status="in_progress") -> None:
        task_dir = root / ".trellis" / "tasks" / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text(
            json.dumps({
                "id": name,
                "title": name,
                "status": status,
                "parent": parent,
                "children": list(children),
            }),
            encoding="utf-8",
        )

    def test_tree_includes_archived_children(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._task(root, "parent", children=("child",))
            child = root / ".trellis" / "tasks" / "archive" / "2026-08" / "child"
            child.mkdir(parents=True)
            (child / "task.json").write_text(json.dumps({
                "id": "child", "title": "child", "status": "completed",
                "parent": "parent", "children": [],
            }), encoding="utf-8")
            tree = build_task_tree(root)
            self.assertTrue(tree["valid"])
            self.assertEqual(tree["roots"], ["parent"])
            self.assertEqual(tree["nodes"]["child"]["location"], "archive/2026-08")

    def test_invalid_utf8_task_metadata_is_visible_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / ".trellis/tasks/bad"
            task.mkdir(parents=True)
            (task / "task.json").write_bytes(b"{\xff}")
            tree = build_task_tree(root)
            self.assertFalse(tree["valid"])
            self.assertIn("malformed_task", [issue["code"] for issue in tree["issues"]])

    def test_missing_child_and_cycle_are_visible_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._task(root, "a", parent="b", children=("b", "missing"))
            self._task(root, "b", parent="a", children=("a",))
            tree = build_task_tree(root)
            self.assertFalse(tree["valid"])
            self.assertEqual(
                {issue["code"] for issue in tree["issues"]},
                {"missing_child", "task_tree_cycle"},
            )


if __name__ == "__main__":
    unittest.main()
