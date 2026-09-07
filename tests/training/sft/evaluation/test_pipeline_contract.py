"""Static/CPU checks for the Slurm-owned SFT1 -> SFT2 experiment pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
PIPELINE = ROOT / "experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh"
CONTRACT = ROOT / "experiments/training/sft/evaluation/pipeline_contract.py"


def load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "sft_eval_pipeline_contract", CONTRACT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_encodes_sequential_training_merge_and_eval_contract():
    text = PIPELINE.read_text(encoding="utf-8")
    stage1 = text.index("-m nimloth.training.sft.stage1")
    first_merge = text.index("-m nimloth.training.sft.stage1.checkpoint_export", stage1)
    stage2 = text.index("-m nimloth.training.sft.stage2", first_merge)
    second_merge = text.index(
        "-m nimloth.training.sft.stage1.checkpoint_export", stage2
    )
    evaluation = text.index("-m nimloth.training.sft.evaluation", second_merge)
    assert stage1 < first_merge < stage2 < second_merge < evaluation
    assert "--latent-token-count 1 --latent-query-mode generate" in text
    assert "--latent-token-count 16 --latent-query-mode inject" in text
    assert '--model "${STAGE1_MERGED}"' in text
    assert '--base-model "${STAGE1_MERGED}"' in text
    assert text.count("--lora-r 64 --lora-alpha 128") == 2
    assert text.count("--no-wandb") == 2
    assert text.count("--lr 1e-6 --embedding-lr 5e-6") == 2
    assert "--max-val-batches" not in text
    assert "--no-cache --no-wandb" in text
    assert "--episodes-per-eval-set 60 --seed-offset 1 --max-steps 20" in text
    assert (
        "--temperature 0 --top-p 1 --max-response-tokens 512 --tensor-parallel-size 1"
        in text
    )


def test_launcher_owns_resources_and_fails_closed():
    text = PIPELINE.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --account=peilab",
        "#SBATCH --partition=normal",
        "#SBATCH --qos=normal_qos",
        "#SBATCH --nodes=1",
        "#SBATCH --gres=gpu:8",
        "#SBATCH --time=06:00:00",
    ):
        assert directive in text
    for required in (
        "EXPECTED_COMMIT",
        "EXPECTED_VAGEN_COMMIT",
        "EXPECTED_VERL_COMMIT",
        "EXPECTED_LEWM_COMMIT",
    ):
        assert f"${{{required}:?" in text
    assert "status --porcelain --untracked-files=all" in text
    assert '[[ ! -e "${RUN_ROOT}" ]]' in text
    assert "rank-map" in text
    assert "direct_render_probe" in (
        ROOT / "experiments/training/sft/evaluation/environment_arm.sh"
    ).read_text(encoding="utf-8")
    assert "trap cleanup EXIT" in text
    assert "terminate_group" in text
    assert "WANDB_MODE=disabled" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "--standalone" not in text
    arm_text = (
        ROOT / "experiments/training/sft/evaluation/environment_arm.sh"
    ).read_text(encoding="utf-8")
    assert (
        'source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"'
        not in arm_text
    )
    assert "timeout --signal=TERM --kill-after=10s 150s" in arm_text
    assert "export HOME=" not in arm_text
    assert arm_text.count('env HOME="${OWNED_HOME}"') == 2
    assert "set -x" in text


def test_jsonl_preflight_counts_answers_images_and_rejects_missing(
    tmp_path, monkeypatch
):
    module = load_contract_module()
    image = tmp_path / "observation.png"
    image.write_bytes(b"image")
    data = tmp_path / "data.jsonl"
    rows = [
        {
            "id": "episode-1",
            "image_paths": [str(image)],
            "messages": [
                {"role": "user", "content": "<image> go"},
                {"role": "assistant", "content": "<think>x</think> action"},
            ],
        }
    ]
    data.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    monkeypatch.setitem(
        module.EXPECTED_DATA, "train", {"records": 1, "assistant_turns": 1}
    )
    summary, images = module.inspect_jsonl(data, "train")
    assert summary["records"] == summary["assistant_turns"] == 1
    assert images == {str(image.resolve())}
    rows[0]["image_paths"] = [str(tmp_path / "missing.png")]
    data.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="recorded image"):
        module.inspect_jsonl(data, "train")


def _write_eval_arm(path: Path, checkpoint: Path) -> None:
    path.mkdir()
    argv = [
        "--backend",
        "vllm",
        "--model",
        str(checkpoint),
        "--num-episodes",
        "120",
        "--split",
        "test",
        "--seed-offset",
        "1",
        "--seed-per-eval-set",
        "--max-steps",
        "20",
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--max-response-tokens",
        "512",
        "--tensor-parallel-size",
        "1",
    ]
    (path / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "evaluation": "sft_eval_direct_v1",
                "policy_fingerprint": {"sha256": path.name},
                "rollout_argv": argv,
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "overall": {"success_rate": 0.5, "avg_reward": 0.5, "avg_steps": 10.0},
        "by_eval_set": {
            "base": {"success_rate": 0.5, "avg_reward": 0.5, "avg_steps": 10.0},
            "common_sense": {"success_rate": 0.5, "avg_reward": 0.5, "avg_steps": 10.0},
        },
    }
    (path / "rollout_summary.json").write_text(
        json.dumps(
            {
                "status": "ALL_OK",
                "num_trajectories": 120,
                "num_transitions": 1200,
                "eval_sets": ["base", "common_sense"],
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    (path / "trajectories.jsonl").write_text(
        "".join(
            json.dumps({"id": f"{path.name}-{index}"}) + "\n" for index in range(120)
        ),
        encoding="utf-8",
    )


def test_finalize_requires_both_complete_120_episode_arms(tmp_path):
    module = load_contract_module()
    module._load_validated_trajectories = lambda path: [
        SimpleNamespace(record_id=json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage1_eval = tmp_path / "stage1_eval"
    stage2_eval = tmp_path / "stage2_eval"
    _write_eval_arm(stage1_eval, tmp_path / "stage1_model")
    _write_eval_arm(stage2_eval, tmp_path / "stage2_model")
    stage1_merge = tmp_path / "stage1_merge.json"
    stage2_merge = tmp_path / "stage2_merge.json"
    stage1_merge.write_text('{"stage":"format"}', encoding="utf-8")
    stage2_merge.write_text('{"stage":"query"}', encoding="utf-8")
    output = tmp_path / "final.json"
    args = module.build_parser().parse_args(
        [
            "finalize",
            "--stage1-eval",
            str(stage1_eval),
            "--stage2-eval",
            str(stage2_eval),
            "--stage1-merge",
            str(stage1_merge),
            "--stage2-merge",
            str(stage2_merge),
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 0
    assert json.loads(output.read_text())["status"] == "passed"
    rows = (stage2_eval / "trajectories.jsonl").read_text().splitlines()
    (stage2_eval / "trajectories.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        module.finalize(args)


def test_cleanup_failure_overrides_previously_passed_status(tmp_path):
    module = load_contract_module()
    output = tmp_path / "final_status.json"
    output.write_text('{"schema":"test","status":"passed"}\n', encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "record-exit",
            "--exit-code",
            "92",
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "cleanup_failed"
    assert payload["pipeline_status_before_cleanup"] == "passed"
    assert payload["exit_code"] == 92
