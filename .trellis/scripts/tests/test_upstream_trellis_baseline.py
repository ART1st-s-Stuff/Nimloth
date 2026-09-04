from __future__ import annotations

import hashlib
import json
import shutil
import stat
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / ".trellis/scripts/tests/fixtures/upstream-trellis-0.6.16-manifest.json"
PROJECT_OWNED_EXCEPTIONS = {".trellis/config.yaml"}


def _trellis_package_root() -> Path:
    executable = shutil.which("trellis")
    if executable is None:
        raise RuntimeError("trellis CLI is not installed")
    package_root = Path(executable).resolve().parents[1]
    package_json = package_root / "package.json"
    if not package_json.is_file():
        raise RuntimeError(f"cannot resolve Trellis package from {executable}")
    version = json.loads(package_json.read_text())["version"]
    if version != "0.6.16":
        raise RuntimeError(f"expected Trellis 0.6.16, found {version}")
    return package_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpstreamTrellisBaselineTest(unittest.TestCase):
    def test_every_managed_file_matches_fixed_release_manifest(self) -> None:
        _trellis_package_root()
        manifest = json.loads(FIXTURE.read_text())
        self.assertEqual(manifest["release"], "@mindfoldhq/trellis@0.6.16")
        self.assertEqual(set(manifest["projectOwnedExceptions"]), PROJECT_OWNED_EXCEPTIONS)

        registry = json.loads((ROOT / ".trellis/.template-hashes.json").read_text())["hashes"]
        expected_paths = set(registry) - PROJECT_OWNED_EXCEPTIONS
        managed = manifest["managedFiles"]
        self.assertGreaterEqual(len(managed), 140)
        self.assertEqual(set(managed), expected_paths)

        failures: list[str] = []
        for relative, expected in managed.items():
            actual = ROOT / relative
            if not actual.is_file():
                failures.append(f"{relative}: missing")
                continue
            actual_hash = _sha256(actual)
            if actual_hash != expected["sha256"]:
                failures.append(f"{relative}: content {actual_hash}")
            executable = bool(actual.stat().st_mode & stat.S_IXUSR)
            if executable != expected["executable"]:
                failures.append(f"{relative}: executable={executable}")

        self.assertEqual(failures, [], "managed Trellis drift:\n" + "\n".join(failures))

    def test_direct_release_sources_and_version_match_0_6_16(self) -> None:
        package_root = _trellis_package_root()
        templates = package_root / "dist" / "templates"
        direct_pairs = {
            ROOT / ".trellis/workflow.md": templates / "trellis/workflow.md",
            ROOT / ".trellis/scripts/task.py": templates / "trellis/scripts/task.py",
            ROOT / ".pi/extensions/trellis/index.ts": templates / "pi/extensions/trellis/index.ts.txt",
        }
        mismatches = [
            actual.relative_to(ROOT).as_posix()
            for actual, expected in direct_pairs.items()
            if actual.read_bytes() != expected.read_bytes()
        ]
        self.assertEqual(mismatches, [])
        self.assertEqual((ROOT / ".trellis/.version").read_text().strip(), "0.6.16")


if __name__ == "__main__":
    unittest.main()
