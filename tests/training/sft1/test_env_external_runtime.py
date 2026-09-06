"""Exercise launcher routing only; fake executables do not validate GPU imports."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("use_reconstruction", [False, True])
def test_env_launcher_preserves_runtime_after_credentials(tmp_path, use_reconstruction):
    repo = tmp_path / "repo"
    scripts = repo / "experiments/training/sft1"
    scripts.mkdir(parents=True)
    credentials = tmp_path / "credentials.env"
    credentials.write_text(
        'export PYTHON_ENV="/wrong/credential/venv"\n'
        'export VAGEN_DIR="/wrong/credential/vagen"\n'
        'export PYTHONPATH="/wrong/credential/imports"\n'
        'export PATH="/wrong/credential/bin:$PATH"\n'
    )
    for name in ("env_external_4gpu.slurm", "common_env.sh"):
        source = (ROOT / "experiments/training/sft1" / name).read_text()
        # Isolate the real credential-file source behind a temporary test file.
        source = source.replace("/project/peilab/atst/flower/.env", str(credentials))
        (scripts / name).write_text(source)
    baseline = repo / "experiments/training/baseline"
    baseline.mkdir()
    (baseline / "setup_ai2thor_env.sh").write_text(
        (ROOT / "experiments/training/baseline/setup_ai2thor_env.sh").read_text()
    )
    vagen = repo / ("reconstructed VAGEN" if use_reconstruction else "external/VAGEN")
    (vagen / "verl").mkdir(parents=True)
    venv = repo / "venv"
    (venv / "bin").mkdir(parents=True)
    record = tmp_path / "calls.jsonl"
    python = venv / "bin/python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_RECORD'], 'a') as out:\n"
        "    out.write(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(), "
        "'python_env': os.environ['PYTHON_ENV'], 'virtual_env': os.environ['VIRTUAL_ENV'], "
        "'vagen_dir': os.environ['VAGEN_DIR'], 'pythonpath': os.environ['PYTHONPATH']}) + '\\n')\n"
    )
    python.chmod(0o755)
    curl = venv / "bin/curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    hostname = venv / "bin/hostname"
    hostname.write_text('#!/bin/sh\nif [ "$#" -gt 0 ]; then echo 127.0.0.1; else echo fixture; fi\n')
    hostname.chmod(0o755)
    vulkan = tmp_path / "vulkan"
    lib = vulkan / "extracted/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
    lib.parent.mkdir(parents=True)
    lib.touch()
    job_id = f"test-{tmp_path.parent.name}-{tmp_path.name}"
    env = dict(os.environ)
    env.pop("VAGEN_DIR", None)
    env.update(
        REPO=str(repo), PYTHON_ENV=str(venv), PYTHONPATH="/caller/imports",
        ROLLOUT_RUN_DIR=str(tmp_path / "run"), ROLLOUT_RUN_NAME="routing-test",
        CUDA_VISIBLE_DEVICES="0", ENV_SERVICE_COUNT="1", SLURM_JOB_ID=job_id,
        VULKAN_ROOT=str(vulkan), AI2THOR_HOME_ROOT=str(tmp_path / "thor"),
        CALL_RECORD=str(record), USER="launcher-test",
    )
    if use_reconstruction:
        env["VAGEN_DIR"] = str(vagen)
    result = subprocess.run(
        ["bash", str(scripts / "env_external_4gpu.slurm")], env=env,
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == 3  # setup library check, render smoke, environment server
    for call in calls:
        assert call["argv"][0] == str(python)
        assert call["python_env"] == call["virtual_env"] == str(venv)
        assert call["vagen_dir"] == str(vagen)
        assert call["pythonpath"] == f"{vagen}:{vagen}/verl:/caller/imports"
    assert calls[-1]["argv"][1:3] == ["-m", "vagen.server.server"]
    assert calls[-1]["cwd"] == str(vagen)
    assert (tmp_path / "run/external_env_4gpu/ready").is_file()
    Path(f"/tmp/vagen_env_sft1v79_{job_id}_pids").unlink()
