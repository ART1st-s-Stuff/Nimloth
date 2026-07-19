import importlib.util
import json
from pathlib import Path

import pandas as pd
from PIL import Image


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "training"
    / "sft1"
    / "recover_rollout_resolution_pairs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("recover_rollout_pairs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_control(path: Path) -> None:
    rows = []
    for seed in range(1, 3):
        rows.append(
            {
                "data_source": "navigation_base_train_resolution_probe",
                "prompt": [{"role": "user", "content": ""}],
                "extra_info": {
                    "seed": seed,
                    "env_config": {"eval_set": "base_train"},
                },
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def _write_run(
    root: Path,
    *,
    size: int,
    order: list[int],
    reported_seeds: list[int],
    successes: dict[int, bool],
) -> Path:
    root.mkdir(parents=True)
    colors = {1: (255, 0, 0), 2: (0, 255, 0)}
    rows = []
    for record_index, (task, reported_seed) in enumerate(
        zip(order, reported_seeds, strict=True)
    ):
        image_path = root / f"{record_index}.png"
        Image.new("RGB", (size, size), colors[task]).save(image_path)
        rows.append(
            {
                "env_id": f"val{task}",
                "config_id": "NavigationEnvConfig(eval_set=base_train,x=1)",
                "output_str": (
                    "Human Instruction: duplicate instruction\n"
                    "Decide your next action(s)."
                ),
                "metrics": {"success": successes[task]},
                "data_source": "navigation_base_train_resolution_probe",
                "env_seed": reported_seed,
                "eval_set": "base_train",
                "image_paths": [str(image_path)],
            }
        )
    path = root / "0.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_recovers_pairs_from_batch_instruction_and_initial_frame(tmp_path: Path):
    module = _load_module()
    old_control = tmp_path / "old.parquet"
    new_control = tmp_path / "new.parquet"
    _write_control(old_control)
    _write_control(new_control)
    old_jsonl = _write_run(
        tmp_path / "old",
        size=512,
        order=[1, 2],
        reported_seeds=[2, 1],
        successes={1: True, 2: False},
    )
    new_jsonl = _write_run(
        tmp_path / "new",
        size=255,
        order=[2, 1],
        reported_seeds=[1, 2],
        successes={1: True, 2: True},
    )

    result = module.compare_runtime_pairs(
        old_jsonl,
        new_jsonl,
        old_control,
        new_control,
        batch_size=2,
    )

    assert result["records"] == 2
    assert result["old_success"] == 1
    assert result["new_success"] == 2
    assert result["paired"] == {
        "both_success": 1,
        "old_only_success": 0,
        "new_only_success": 1,
        "both_failure": 0,
    }
    assert result["matching"]["multi_record_groups"] == 1
    assert result["matching"]["max_assigned_initial_frame_rmse"] == 0.0
    assert result["matching"]["min_unassigned_initial_frame_rmse_in_multi_groups"] > 0
