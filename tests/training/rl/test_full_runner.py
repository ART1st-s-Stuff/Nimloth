from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_full.sh"
CONTINUATION_CONFIG = (
    REPO_ROOT / "configs/training/rl/planner_greedy_h2_continuation_gate.yaml"
)


def _write_fake_iteration_runner(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path


def append_step(path: Path, iteration: int) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["global_step"])
        if write_header:
            writer.writeheader()
        writer.writerow({"global_step": iteration})


iteration = int(os.environ["ITERATION"])
total = int(os.environ["TOTAL_ITERATIONS"])
initial_global_step = int(os.environ.get("RUN_INITIAL_GLOBAL_STEP", "0"))
failure_mode = os.environ.get("FAKE_FAILURE_MODE", "")
root = Path(os.environ["RUN_OUT"])
train = root / "train"
latest = train / "latest"
tag = f"iter_{iteration:04d}"
manifest = root / "rollouts" / tag / "fresh_policy_manifest.json"
failure_marker = root.parent / f"fake_failure_{failure_mode}.done"

with (root.parent / "fake_calls.txt").open("a", encoding="utf-8") as stream:
    stream.write(f"{iteration}\\n")

if iteration == initial_global_step + 1:
    assert Path(os.environ["MODEL"]) == Path(os.environ["INITIAL_MODEL"])
    assert os.environ["RESUME_CHECKPOINT"] == os.environ.get(
        "INITIAL_RESUME_CHECKPOINT", ""
    )
    (root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("running\\n", encoding="utf-8")
else:
    snapshot = train / "policy_inputs" / tag
    assert Path(os.environ["MODEL"]) == snapshot
    assert Path(os.environ["WM_CKPT"]) == snapshot
    assert Path(os.environ["RESUME_CHECKPOINT"]) == snapshot
    previous = (
        root
        / "rollouts"
        / f"iter_{iteration - 1:04d}"
        / "fresh_policy_manifest.json.consumption.json"
    )
    assert Path(json.loads(previous.read_text())["checkpoint_path"]) == snapshot

if (
    iteration == 2
    and failure_mode == "before_outputs"
    and not failure_marker.exists()
):
    failure_marker.touch()
    sys.exit(23)

manifest.parent.mkdir(parents=True, exist_ok=False)
manifest.write_text("{}\\n", encoding="utf-8")
consumption = Path(str(manifest) + ".consumption.json")
consumption.write_text(
    json.dumps(
        {
            "state": "in_progress",
            "starting_global_step": iteration - 1,
        }
    )
    + "\\n",
    encoding="utf-8",
)

latest.mkdir(parents=True)
(latest / "rl_state.pt").write_bytes(f"step={iteration}".encode())
append_step(train / "train_step_log.csv", iteration)

if (
    iteration == 2
    and failure_mode == "after_speculative_checkpoint"
    and not failure_marker.exists()
):
    failure_marker.touch()
    (root / "reference" / tag).mkdir(parents=True)
    (Path(str(root) + ".ray") / tag).mkdir(parents=True)
    sys.exit(23)

consumption.write_text(
    json.dumps(
        {
            "state": "committed",
            "starting_global_step": iteration - 1,
            "committed_global_step": iteration,
            "checkpoint_path": str(latest.resolve()),
        }
    )
    + "\\n",
    encoding="utf-8",
)
if (
    iteration == total
    and failure_mode == "after_committed_before_final"
    and not failure_marker.exists()
):
    failure_marker.touch()
    sys.exit(23)
if iteration == total:
    final = train / "final"
    final.mkdir()
    (final / "rl_state.pt").write_bytes(f"step={iteration}".encode())
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_evaluation_runner(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path

assert os.environ["PIPELINE_MODE"] == "eval"
iteration = int(os.environ["ITERATION"])
root = Path(os.environ["RUN_OUT"])
assert iteration == 10
assert Path(os.environ["MODEL"]) == root / "train/latest"
target = root / "evaluation" / f"iter_{iteration:04d}"
target.mkdir(parents=True)
(target / "eval_done.flag").write_text("ALL_OK\\n", encoding="utf-8")
with (root.parent / "fake_eval_calls.txt").open("a", encoding="utf-8") as stream:
    stream.write(f"{iteration}\\n")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _runner_environment(tmp_path: Path, *, failure_mode: str = "") -> dict[str, str]:
    initial_model = tmp_path / "initial_model"
    initial_model.mkdir()
    (initial_model / "config.json").write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "formal"
    fake_runner = tmp_path / "fake_iteration.py"
    _write_fake_iteration_runner(fake_runner)
    return {
        **os.environ,
        "HOLD_JOB": "test-hold",
        "REPO": str(REPO_ROOT),
        "ENV_REPO": str(tmp_path / "environment"),
        "PYTHON": sys.executable,
        "RL_CONFIG": str(CONTINUATION_CONFIG),
        "RUN_OUT": str(output_root / "run"),
        "FORMAL_OUTPUT_ROOT": str(output_root),
        "ITERATION_RUNNER": str(fake_runner),
        "INITIAL_MODEL": str(initial_model),
        "INITIAL_WM_CKPT": str(initial_model),
        "REFERENCE_MODEL": str(initial_model),
        "WANDB_RUN_NAME": "test-continuation",
        "FAKE_FAILURE_MODE": failure_mode,
    }


def _assert_completed_run(environment: dict[str, str]) -> None:
    run_output = Path(environment["RUN_OUT"])
    snapshot = run_output / "train/policy_inputs/iter_0002/rl_state.pt"
    assert snapshot.read_bytes() == b"step=1"
    assert (run_output / "train/latest/rl_state.pt").read_bytes() == b"step=2"
    assert (run_output / "train/final/rl_state.pt").read_bytes() == b"step=2"
    with (run_output / "train/train_step_log.csv").open(encoding="utf-8") as stream:
        assert [row["global_step"] for row in csv.DictReader(stream)] == ["1", "2"]


def test_full_runner_relocates_checkpoint_and_resumes_next_policy(
    tmp_path: Path,
) -> None:
    environment = _runner_environment(tmp_path)
    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    _assert_completed_run(environment)
    calls = tmp_path / "formal/fake_calls.txt"
    assert calls.read_text(encoding="utf-8") == "1\n2\n"

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)
    assert calls.read_text(encoding="utf-8") == "1\n2\n"


def test_full_runner_uses_its_batch_allocation_when_hold_job_is_absent(
    tmp_path: Path,
) -> None:
    environment = _runner_environment(tmp_path)
    environment.pop("HOLD_JOB")
    environment["SLURM_JOB_ID"] = "test-batch-allocation"

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    _assert_completed_run(environment)


@pytest.mark.parametrize(
    "failure_mode",
    ["before_outputs", "after_speculative_checkpoint"],
)
def test_full_runner_recovers_an_interrupted_iteration(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    environment = _runner_environment(tmp_path, failure_mode=failure_mode)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    run_output = Path(environment["RUN_OUT"])
    latest = run_output / "train/latest/rl_state.pt"
    assert latest.is_file() == (failure_mode == "after_speculative_checkpoint")
    assert (run_output / "train/policy_inputs/iter_0002/rl_state.pt").is_file()

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)
    _assert_completed_run(environment)
    calls = tmp_path / "formal/fake_calls.txt"
    assert calls.read_text(encoding="utf-8") == "1\n2\n2\n"

    if failure_mode == "after_speculative_checkpoint":
        recovery_root = Path(str(run_output) + ".recovery")
        attempts = list(recovery_root.glob("iter_0002_attempt_*"))
        assert len(attempts) == 1
        assert (attempts[0] / "uncommitted_latest/rl_state.pt").is_file()
        assert (attempts[0] / "rollout/fresh_policy_manifest.json").is_file()
        assert (attempts[0] / "train_step_log.csv.before_recovery").is_file()


def test_full_runner_recovers_a_committed_iteration_missing_only_final_alias(
    tmp_path: Path,
) -> None:
    environment = _runner_environment(
        tmp_path,
        failure_mode="after_committed_before_final",
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    run_output = Path(environment["RUN_OUT"])
    assert (run_output / "train/latest/rl_state.pt").is_file()
    assert not (run_output / "train/final").exists()

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)
    _assert_completed_run(environment)
    calls = tmp_path / "formal/fake_calls.txt"
    assert calls.read_text(encoding="utf-8") == "1\n2\n"


def test_full_runner_can_continue_optimizer_state_in_a_new_output(tmp_path: Path) -> None:
    environment = _runner_environment(tmp_path)
    initial_resume = tmp_path / "initial_resume"
    initial_resume.mkdir()
    torch.save(
        {"iteration": 1, "global_step": 1},
        initial_resume / "rl_state.pt",
    )
    environment["INITIAL_RESUME_CHECKPOINT"] = str(initial_resume)

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    run_output = Path(environment["RUN_OUT"])
    assert (run_output / "train/final/rl_state.pt").read_bytes() == b"step=2"
    with (run_output / "train/train_step_log.csv").open(encoding="utf-8") as stream:
        assert [row["global_step"] for row in csv.DictReader(stream)] == ["2"]
    assert (tmp_path / "formal/fake_calls.txt").read_text(encoding="utf-8") == "2\n"
    consumption = json.loads(
        (
            run_output
            / "rollouts/iter_0002/fresh_policy_manifest.json.consumption.json"
        ).read_text(encoding="utf-8")
    )
    assert consumption["starting_global_step"] == 1
    assert consumption["committed_global_step"] == 2


def test_full_runner_runs_external_eval_once_at_iteration_ten(tmp_path: Path) -> None:
    config_path = tmp_path / "external_eval.yaml"
    config_text = CONTINUATION_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace("iterations: 2", "iterations: 10")
    config_text = config_text.replace(
        "  temperature: 0.7",
        "  eval_datasets:\n    - base\n    - common_sense\n  temperature: 0.7",
    )
    config_text = config_text.replace(
        "validation:\n  enabled: false\n  interval: 2\n  envs: 0",
        "validation:\n  enabled: false\n  external: true\n  interval: 10\n  envs: 120",
    )
    config_path.write_text(config_text, encoding="utf-8")
    evaluation_runner = tmp_path / "fake_evaluation.py"
    _write_fake_evaluation_runner(evaluation_runner)
    environment = _runner_environment(tmp_path)
    environment["RL_CONFIG"] = str(config_path)
    environment["EVALUATION_RUNNER"] = str(evaluation_runner)

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)
    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    assert (tmp_path / "formal/fake_eval_calls.txt").read_text(
        encoding="utf-8"
    ) == "10\n"
    assert (
        Path(environment["RUN_OUT"]) / "evaluation/iter_0010/eval_done.flag"
    ).is_file()
