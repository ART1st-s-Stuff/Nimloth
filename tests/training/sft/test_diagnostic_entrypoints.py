"""诊断脚本从任意工作目录执行时必须加载自身 checkout 的实验包。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = [
    "probe_kv_trajectory.py",
    "probe_kv_incremental.py",
    "probe_kv_incremental_trajectory.py",
    "probe_trajectory_once_equiv.py",
    "debug_trajectory_once.py",
    "diagnose_trajectory_equiv.py",
    "smoke_speedup.py",
    "validate_per_image_vision_cache.py",
    "validate_trajectory_once_2step.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_diagnostic_script_help_outside_checkout(tmp_path, script):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/training/sft2/diagnosis" / script), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
