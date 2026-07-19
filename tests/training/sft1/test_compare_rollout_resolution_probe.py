import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "training"
    / "sft1"
    / "compare_rollout_resolution_probe.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_rollout_resolution_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, successes: list[int], size: int) -> Path:
    shard = root / "validation" / "train120" / "shard_001_060"
    shard.mkdir(parents=True)
    rows = []
    for seed, success in enumerate(successes, start=1):
        rows.append(
            {
                "data_source": "navigation_base_train_resolution_probe",
                "env_seed": seed,
                "traj_success": success,
            }
        )
        image_dir = shard / "image_0" / f"images_{seed - 1}"
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (size, size)).save(image_dir / "0.png")
    path = shard / "0.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_compare_reports_paired_success_and_image_sizes(tmp_path: Path):
    module = _load_module()
    old_path = _write_run(tmp_path / "old", [1, 1, 0, 0], 512)
    new_path = _write_run(tmp_path / "new", [1, 0, 1, 0], 255)

    result = module.compare_runs(old_path, new_path)

    assert result["records"] == 4
    assert result["old"]["success"] == 2
    assert result["new"]["success"] == 2
    assert result["success_rate_delta"] == 0.0
    assert result["paired"] == {
        "both_success": 1,
        "old_only_success": 1,
        "new_only_success": 1,
        "both_failure": 1,
    }
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert result["old"]["png_size_counts"] == {"512x512": 4}
    assert result["new"]["png_size_counts"] == {"255x255": 4}
