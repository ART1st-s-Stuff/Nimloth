import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "training"
    / "sft1"
    / "validate_rollout_train120_dump.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_train120", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dump(path: Path, *, corrupt_index: int | None = None) -> None:
    rows = []
    for source, eval_set in (
        ("navigation_base_train_resolution_probe", "base_train"),
        ("navigation_common_train_resolution_probe", "common_sense_train"),
    ):
        for seed in range(1, 61):
            image_path = path.parent / f"{len(rows)}.png"
            Image.new("RGB", (255, 255)).save(image_path)
            rows.append(
                {
                    "data_source": source,
                    "env_seed": seed,
                    "eval_set": eval_set,
                    "uid": f"{source}:{seed}:{eval_set}",
                    "config_id": f"NavigationEnvConfig(eval_set={eval_set},x=1)",
                    "metrics": {"success": False},
                    "image_paths": [str(image_path)],
                }
            )
    if corrupt_index is not None:
        rows[corrupt_index]["eval_set"] = "common_sense_train"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_train120_requires_exact_stable_metadata_and_pngs(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "0.jsonl"
    _write_dump(path)

    result = module.validate_dump(path, expected_png_size=255)

    assert result == {
        "records": 120,
        "unique_keys": 120,
        "metadata_mismatches": 0,
        "image_references": 120,
        "image_size": "255x255",
    }


def test_validate_train120_rejects_metadata_mismatch(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "0.jsonl"
    _write_dump(path, corrupt_index=0)

    with pytest.raises(ValueError, match="stable metadata mismatch"):
        module.validate_dump(path, expected_png_size=255)
