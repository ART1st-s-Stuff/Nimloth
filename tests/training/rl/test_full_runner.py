from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_full.sh"
CONTINUATION_CONFIG = (
    REPO_ROOT / "configs/training/rl/planner_greedy_h2_continuation_gate.yaml"
)


def test_full_runner_relocates_checkpoint_and_resumes_next_policy(
    tmp_path: Path,
) -> None:
    initial_model = tmp_path / "initial_model"
    initial_model.mkdir()
    (initial_model / "config.json").write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "formal"
    run_output = output_root / "run"
    fake_runner = tmp_path / "fake_iteration.py"
    fake_runner.write_text(
        """#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

iteration = int(os.environ["ITERATION"])
total = int(os.environ["TOTAL_ITERATIONS"])
root = Path(os.environ["RUN_OUT"])
train = root / "train"
latest = train / "latest"
tag = f"iter_{iteration:04d}"
manifest = root / "rollouts" / tag / "fresh_policy_manifest.json"
manifest.parent.mkdir(parents=True, exist_ok=False)

if iteration == 1:
    assert Path(os.environ["MODEL"]) == Path(os.environ["INITIAL_MODEL"])
    assert os.environ["RESUME_CHECKPOINT"] == ""
    (root / "README.md").write_text("running\\n", encoding="utf-8")
else:
    snapshot = train / "policy_inputs" / tag
    assert Path(os.environ["MODEL"]) == snapshot
    assert Path(os.environ["WM_CKPT"]) == snapshot
    assert Path(os.environ["RESUME_CHECKPOINT"]) == snapshot
    previous = root / "rollouts" / f"iter_{iteration - 1:04d}" / "fresh_policy_manifest.json.consumption.json"
    assert Path(json.loads(previous.read_text())["checkpoint_path"]) == snapshot

manifest.write_text("{}\\n", encoding="utf-8")
latest.mkdir(parents=True)
(latest / "rl_state.pt").write_bytes(f"step={iteration}".encode())

step_log = train / "train_step_log.csv"
write_header = not step_log.exists()
with step_log.open("a", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["global_step"])
    if write_header:
        writer.writeheader()
    writer.writerow({"global_step": iteration})

Path(str(manifest) + ".consumption.json").write_text(
    json.dumps({"state": "committed", "checkpoint_path": str(latest)}) + "\\n",
    encoding="utf-8",
)
if iteration == total:
    final = train / "final"
    final.mkdir()
    (final / "rl_state.pt").write_bytes(f"step={iteration}".encode())

with (root.parent / "fake_calls.txt").open("a", encoding="utf-8") as stream:
    stream.write(f"{iteration}\\n")
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)

    environment = {
        **os.environ,
        "HOLD_JOB": "test-hold",
        "REPO": str(REPO_ROOT),
        "ENV_REPO": str(tmp_path / "environment"),
        "PYTHON": sys.executable,
        "RL_CONFIG": str(CONTINUATION_CONFIG),
        "RUN_OUT": str(run_output),
        "FORMAL_OUTPUT_ROOT": str(output_root),
        "ITERATION_RUNNER": str(fake_runner),
        "INITIAL_MODEL": str(initial_model),
        "INITIAL_WM_CKPT": str(initial_model),
        "REFERENCE_MODEL": str(initial_model),
        "WANDB_RUN_NAME": "test-continuation",
    }
    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)

    snapshot = run_output / "train/policy_inputs/iter_0002/rl_state.pt"
    assert snapshot.read_bytes() == b"step=1"
    assert (run_output / "train/latest/rl_state.pt").read_bytes() == b"step=2"
    assert (run_output / "train/final/rl_state.pt").read_bytes() == b"step=2"
    assert (output_root / "fake_calls.txt").read_text(encoding="utf-8") == "1\n2\n"

    subprocess.run([str(FULL_RUNNER)], check=True, env=environment)
    assert (output_root / "fake_calls.txt").read_text(encoding="utf-8") == "1\n2\n"
