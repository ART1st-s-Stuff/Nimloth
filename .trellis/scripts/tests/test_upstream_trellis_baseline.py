from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


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


class UpstreamTrellisBaselineTest(unittest.TestCase):
    def test_upstream_owned_files_match_release_0_6_16(self) -> None:
        package_root = _trellis_package_root()
        templates = package_root / "dist" / "templates"
        direct_pairs = {
            ROOT / ".trellis/workflow.md": templates / "trellis/workflow.md",
            ROOT / ".trellis/scripts/task.py": templates / "trellis/scripts/task.py",
            ROOT / ".pi/extensions/trellis/index.ts": templates / "pi/extensions/trellis/index.ts.txt",
        }
        registry = json.loads((ROOT / ".trellis/.template-hashes.json").read_text())["hashes"]
        generated_paths = (
            ".pi/agents/trellis-check.md",
            ".pi/agents/trellis-implement.md",
            ".pi/agents/trellis-research.md",
            ".pi/prompts/trellis-start.md",
            ".pi/prompts/trellis-continue.md",
            ".pi/prompts/trellis-finish-work.md",
        )

        mismatches = []
        for actual, expected in direct_pairs.items():
            if actual.read_bytes() != expected.read_bytes():
                mismatches.append(actual.relative_to(ROOT).as_posix())
        for relative in generated_paths:
            actual = (ROOT / relative).read_bytes()
            actual_hash = hashlib.sha256(actual).hexdigest()
            if actual_hash != registry.get(relative):
                mismatches.append(relative)

        self.assertEqual(
            mismatches,
            [],
            "upstream-owned Trellis files diverge from rendered 0.6.16: " + ", ".join(mismatches),
        )


if __name__ == "__main__":
    unittest.main()
