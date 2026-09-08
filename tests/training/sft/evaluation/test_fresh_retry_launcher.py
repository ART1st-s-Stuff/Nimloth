"""Check stage isolation and fail-closed guards without submitting jobs."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "experiments/training/sft/evaluation"


def test_only_stage1_learning_rate_changes():
    pipeline = (SCRIPTS / "run_step79_stage1_stage2_eval.sh").read_text()
    stage1 = pipeline.split("-m nimloth.training.sft.stage1", 1)[1].split("2>&1", 1)[0]
    stage2 = pipeline.split("-m nimloth.training.sft.stage2", 1)[1].split("2>&1", 1)[0]
    assert "--lr 2e-4 --embedding-lr 5e-4" in stage1
    assert "--lr 1e-6 --embedding-lr 5e-6" in stage2
    continuation = (SCRIPTS / "run_stage1_continuation_dgx56.sh").read_text()
    assert "--lr 2e-4 --embedding-lr 5e-4" in continuation


def test_fresh_run_has_no_resume_or_stage2():
    script = (SCRIPTS / "run_stage1_fresh_retry.sh").read_text()
    assert "--resume " not in script
    assert "--new-scheduler-segment-from" not in script
    assert "-m nimloth.training.sft.stage2" not in script
    assert "--epochs 1" in script
    assert '--grad-accum "$((32 / WORLD_SIZE))"' in script
    assert '--adapter "${RUN_ROOT}/stage1/epoch_001"' in script
    assert "--resume-save-steps 5 --keep-resume-checkpoints 2" in script


def test_invalid_world_fails_before_any_slurm_call():
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_stage1_fresh_retry.sh")],
        env={
            **os.environ,
            "WORLD_SIZE": "3",
            "EXPECTED_NODE": "test",
            "ALLOCATION_JOB_ID": "123",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "WORLD_SIZE must divide" in result.stderr


def test_existing_output_is_rejected_before_loading(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/sh\ncase "$*" in *rev-parse*) echo expected;; esac\n')
    git.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_stage1_fresh_retry.sh")],
        env={
            **os.environ,
            "PATH": str(fake_bin) + ":" + os.environ["PATH"],
            "WORLD_SIZE": "4",
            "EXPECTED_NODE": "dgx-22",
            "ALLOCATION_JOB_ID": "123",
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NODELIST": "dgx-22",
            "NIMLOTH_ALLOCATION_STEP": "1",
            "REPO": str(tmp_path),
            "EXPECTED_COMMIT": "expected",
            "RUN_ROOT": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "refuses existing RUN_ROOT" in result.stderr


def test_embedded_python_compiles():
    script = (SCRIPTS / "run_stage1_fresh_retry.sh").read_text()
    blocks = script.split("<<'PYEND'\n")[1:]
    assert len(blocks) == 2
    for block in blocks:
        compile(block.split("\nPYEND", 1)[0], "inline.py", "exec")
