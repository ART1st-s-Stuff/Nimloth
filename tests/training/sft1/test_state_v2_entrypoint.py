from __future__ import annotations

import inspect
import json
from pathlib import Path

from nimloth.training.sft1 import entrypoint
from nimloth.training.sft1.entrypoint import validate_sft1_v2_canary_inputs
from tests.training.sft1._state_v2_fixtures import manifest_raw


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "training" / "sft1" / "state_interface_v2_code_canary.yaml"


def test_code_canary_entrypoint_validates_without_launching_services(
    tmp_path,
    capsys,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_raw()), encoding="utf-8")

    summary = validate_sft1_v2_canary_inputs(
        config_path=CONFIG,
        manifest_path=manifest,
        repo_root=ROOT,
    )
    assert summary["status"] == "code_canary_preflight_only"
    assert summary["launch_authorized"] is False
    assert summary["query_count"] == 16
    assert len(str(summary["config_identity"])) == 64

    assert entrypoint.main(
        [
            "--config",
            str(CONFIG),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(ROOT),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["manifest_identity"] == summary["manifest_identity"]

    source = inspect.getsource(entrypoint)
    assert "import ray" not in source
    assert "wandb" not in source.lower()
    assert "sbatch" not in source.lower()
