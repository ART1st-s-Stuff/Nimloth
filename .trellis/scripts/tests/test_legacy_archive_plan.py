from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / ".trellis/tasks/09-03-archive-legacy-trellis-knowledge"
TOOL_PATH = TASK / "tools/legacy_archive_plan.py"
ROLLBACK_PATH = TASK / "tools/rollback-legacy-archive.py"
MOVE_SCRIPT = TASK / "tools/propose-legacy-archive-moves.sh"


def load_tool():
    spec = importlib.util.spec_from_file_location("legacy_archive_plan", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rollback():
    spec = importlib.util.spec_from_file_location("rollback_legacy_archive", ROLLBACK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ROLLBACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyArchivePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()
        cls.rollback = load_rollback()

    def test_known_error_audit_matches_bytes_union_and_exact_contract_text(self) -> None:
        rows = self.tool.load_jsonl(TASK / "research/known-error-audit.jsonl")
        self.assertEqual([], self.tool.validate_audit(ROOT, rows, mode="post"))
        self.assertEqual(156, len(rows))
        self.assertEqual(33, sum(row["disposition"] == "retain-contract" for row in rows))
        self.assertEqual(117, sum(row["disposition"] == "archive-only" for row in rows))
        self.assertEqual(4, sum(row["disposition"] == "needs-human-decision" for row in rows))
        duplicate_prefixes = sorted(
            prefix for prefix in {row["id_prefix"] for row in rows}
            if sum(row["id_prefix"] == prefix for row in rows) > 1
        )
        self.assertEqual(["E0025", "E0026", "E0043", "E0044", "E0094"], duplicate_prefixes)
        retained = [row for row in rows if row["disposition"] == "retain-contract"]
        self.assertTrue(all(row["contract_mappings"] for row in retained))
        self.assertTrue(all(not row["contract_mappings"] for row in rows if row not in retained))

    def test_move_manifest_is_complete_deterministic_and_collision_free(self) -> None:
        rows = self.tool.load_jsonl(TASK / "research/legacy-move-manifest.jsonl")
        self.assertEqual([], self.tool.validate_move_manifest(ROOT, rows, mode="post"))
        self.assertEqual(254, len(rows))
        self.assertEqual(157, sum(row["group"] == "known-errors" for row in rows))
        self.assertEqual(95, sum(row["group"] == "ai-tasks" for row in rows))
        self.assertEqual(2, sum(row["group"] == "root-history" for row in rows))

    def test_rewrite_patch_covers_active_context_eval_and_slurm_rewrites(self) -> None:
        patch = self.tool.load_rewrite_patch()
        self.assertEqual([], self.tool.validate_rewrite_patch(ROOT, patch, mode="post"))
        replacements = [replacement for entry in patch["files"] for replacement in entry["replacements"]]
        self.assertEqual(18, len(patch["files"]))
        self.assertEqual(19, sum(row["kind"] == "active-context" for row in replacements))
        self.assertEqual(4, sum(row["kind"] == "eval-provenance" for row in replacements))
        self.assertEqual(5, sum(row["kind"] == "slurm-provenance" for row in replacements))
        slurm = [row for row in replacements if row["kind"] == "slurm-provenance"]
        self.assertTrue(all("archive/pre-trellis/sft1_exp.md" in row["new"] for row in slurm))
        self.assertTrue(all("no actor/critic update" in row["new"] for row in slurm))

    def test_rewrite_patch_accepts_simulated_post_move_files(self) -> None:
        patch = self.tool.load_rewrite_patch()
        with tempfile.TemporaryDirectory() as repo_dir:
            root = Path(repo_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            for entry in patch["files"]:
                source = ROOT / entry["path"]
                target = root / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                text = target.read_text(encoding="utf-8")
                for replacement in entry["replacements"]:
                    text = text.replace(replacement["old"], replacement["new"], 1)
                target.write_text(text, encoding="utf-8")
                if any(row["kind"] == "active-context" for row in entry["replacements"]):
                    source_task = source.parent / "task.json"
                    target_task = target.parent / "task.json"
                    shutil.copyfile(source_task, target_task)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            self.assertEqual([], self.tool.validate_rewrite_patch(root, patch, mode="post"))

    def test_destination_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(repo_dir)
            (root / "archive").symlink_to(outside_dir, target_is_directory=True)
            errors = self.tool.validate_destination_path(
                root, "archive/escaped/file.md", require_absent=True
            )
            self.assertTrue(any("symlink" in error for error in errors), errors)

    def test_destination_non_directory_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir:
            root = Path(repo_dir)
            (root / "archive").write_text("not a directory", encoding="utf-8")
            errors = self.tool.validate_destination_path(
                root, "archive/child/file.md", require_absent=True
            )
            self.assertTrue(any("not a directory" in error for error in errors), errors)

    def test_manifest_validator_supports_simulated_pre_and_post_move_repo(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir:
            root = Path(repo_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            # Split legacy literals so the production active-reference scanner does
            # not mistake this isolated negative fixture for a live repository ref.
            payloads = {
                "ai_rules/" + "known_errors/README.md": "index\n",
                "ai_" + "tasks/task.md": "task\n",
                "AI_branch_" + "progress.md": "progress\n",
                "AI_" + "issues.md": "issues\n",
            }
            for relative, payload in payloads.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            rows = self.tool.build_move_manifest(root)
            self.assertEqual([], self.tool.validate_move_manifest(root, rows, mode="pre"))
            self.tool.prepare_destination_directories(root, rows)
            for row in rows:
                self.assertEqual([], self.tool.validate_destination_path(
                    root, row["destination"], require_absent=True
                ))
                subprocess.run(
                    ["git", "mv", "--", row["source"], row["destination"]],
                    cwd=root, check=True,
                )
            self.assertEqual([], self.tool.validate_move_manifest(root, rows, mode="post"))

    def test_execute_gate_rejects_each_self_consistent_changed_artifact_before_validator(self) -> None:
        for changed_artifact in ("manifest", "patch"):
            with self.subTest(changed_artifact=changed_artifact), tempfile.TemporaryDirectory() as repo_dir:
                root = Path(repo_dir)
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                subprocess.run(
                    ["git", "checkout", "-qb", "task/archive-legacy-trellis-knowledge"],
                    cwd=root, check=True,
                )
                fixture_task = root / ".trellis/tasks/09-03-archive-legacy-trellis-knowledge"
                tools = fixture_task / "tools"
                research = fixture_task / "research"
                tools.mkdir(parents=True)
                research.mkdir(parents=True)
                shutil.copyfile(MOVE_SCRIPT, tools / MOVE_SCRIPT.name)
                manifest = research / "legacy-move-manifest.jsonl"
                if changed_artifact == "manifest":
                    manifest.write_text(
                        '{"source":"changed","destination":"also-changed"}\n',
                        encoding="utf-8",
                    )
                else:
                    shutil.copyfile(TASK / "research/legacy-move-manifest.jsonl", manifest)
                patch = research / "approved-rewrite-patch.json"
                patch.write_text(json.dumps({
                    "schema_version": 1,
                    "move_manifest_sha256": file_sha256(manifest),
                    "files": [],
                }), encoding="utf-8")
                validator = tools / "legacy_archive_plan.py"
                validator.write_text(
                    "from pathlib import Path\nPath('VALIDATOR_RAN').write_text('mutation')\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    ["bash", str(tools / MOVE_SCRIPT.name), "--execute-approved-moves"],
                    cwd=root, check=False, capture_output=True, text=True,
                )
                self.assertNotEqual(0, completed.returncode)
                self.assertIn(f"{changed_artifact} differs from reviewed SHA-256", completed.stderr)
                self.assertFalse((root / "VALIDATOR_RAN").exists())

    def test_rollback_late_rewrite_drift_causes_zero_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir:
            root = Path(repo_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            valid = root / "valid.txt"
            drifted = root / "drifted.txt"
            valid.write_text("valid-new\n", encoding="utf-8")
            drifted.write_text("unexpected\n", encoding="utf-8")
            patch = {"files": [
                {
                    "path": "drifted.txt",
                    "pre_sha256": hashlib.sha256(b"drift-old\n").hexdigest(),
                    "post_sha256": hashlib.sha256(b"drift-new\n").hexdigest(),
                    "replacements": [{"old": "drift-old", "new": "drift-new"}],
                },
                {
                    "path": "valid.txt",
                    "pre_sha256": hashlib.sha256(b"valid-old\n").hexdigest(),
                    "post_sha256": file_sha256(valid),
                    "replacements": [{"old": "valid-old", "new": "valid-new"}],
                },
            ]}
            before = valid.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "neither reviewed hash"):
                self.rollback.execute_rollback(root, [], patch)
            self.assertEqual(before, valid.read_bytes())

    def test_rollback_symlinked_ancestor_causes_zero_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(repo_dir)
            outside = Path(outside_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            payload = outside / "file.md"
            payload.write_text("payload\n", encoding="utf-8")
            (root / "restore").symlink_to(outside, target_is_directory=True)
            rewrite = root / "rewrite.txt"
            rewrite.write_text("new\n", encoding="utf-8")
            blob = subprocess.run(
                ["git", "hash-object", "--", "restore/file.md"], cwd=root,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            rows = [{
                "source": "restore/file.md",
                "destination": "archive/file.md",
                "sha256": file_sha256(payload),
                "blob": blob,
            }]
            patch = {"files": [{
                "path": "rewrite.txt",
                "pre_sha256": hashlib.sha256(b"old\n").hexdigest(),
                "post_sha256": file_sha256(rewrite),
                "replacements": [{"old": "old", "new": "new"}],
            }]}
            before = rewrite.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                self.rollback.execute_rollback(root, rows, patch)
            self.assertEqual(before, rewrite.read_bytes())
            self.assertEqual(b"payload\n", payload.read_bytes())

    def test_cli_post_check_reports_all_fingerprints(self) -> None:
        completed = subprocess.run(
            ["python3", str(TOOL_PATH), "--check", "--mode", "post"],
            cwd=ROOT, check=False, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(156, summary["known_error_records"])
        self.assertEqual(254, summary["move_records"])
        self.assertEqual(28, summary["rewrite_replacements"])
        self.assertRegex(summary["move_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["rewrite_patch_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
