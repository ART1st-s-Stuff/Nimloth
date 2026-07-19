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


def test_compare_reads_raw_rollout_metrics_success(tmp_path: Path):
    module = _load_module()
    old_path = _write_run(tmp_path / "old", [0, 0], 512)
    new_path = _write_run(tmp_path / "new", [0, 0], 255)

    for path, successes in ((old_path, [True, False]), (new_path, [True, True])):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row, success in zip(rows, successes, strict=True):
            row.pop("traj_success")
            row["metrics"] = {"success": success}
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    result = module.compare_runs(old_path, new_path)

    assert result["old"]["success"] == 1
    assert result["new"]["success"] == 2
    assert result["paired"] == {
        "both_success": 1,
        "old_only_success": 0,
        "new_only_success": 1,
        "both_failure": 0,
    }


def test_compare_rejects_mismatched_runtime_metadata(tmp_path: Path):
    module = _load_module()
    path = _write_run(tmp_path / "run", [1], 255)
    row = json.loads(path.read_text())
    row["eval_set"] = "base_train"
    row["config_id"] = "NavigationEnvConfig(eval_set=common_sense_train,x=1)"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        module._load_rows(path)
    except ValueError as error:
        assert "mismatched runtime config_id" in str(error)
    else:
        raise AssertionError("mismatched runtime metadata must be rejected")
