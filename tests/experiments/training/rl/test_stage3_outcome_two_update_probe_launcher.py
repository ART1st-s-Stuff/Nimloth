from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = ROOT / "experiments/training/rl/run_stage3_outcome_two_update_probe.sh"


def _acceptance_program() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    programs = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(programs) == 2
    return programs[-1]


def _run_acceptance(
    tmp_path: Path,
    *,
    actor_enabled: bool,
    credit_assignment: str,
    policy_tokens: float = 12.0,
    gradient_l2: float = 1.0,
    qwen_gradient_l2: float = 1.0,
    mean_abs_advantage: float = 0.5,
) -> subprocess.CompletedProcess[str]:
    run = tmp_path / "run"
    train = run / "train"
    train.mkdir(parents=True)
    fields = [
        "global_step",
        "wm_mse",
        "dino_grid_mse",
        "value_loss",
        "outcome_bce",
        "total_loss",
        "outcome_count",
        "loss_finite",
        "gradient_finite",
        "optimizer_updates",
        "actor_loss",
        "policy_tokens",
        "mean_ratio",
        "clip_fraction",
        "mean_advantage",
        "mean_abs_advantage",
        "entropy",
        "gradient_l2",
        "gradient_parameter_count",
        "gradient_qwen_l2",
        "gradient_qwen_parameter_count",
    ]
    with (train / "train_step_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in (1, 2):
            writer.writerow(
                {
                    "global_step": step,
                    "wm_mse": 0.4,
                    "dino_grid_mse": 0.6,
                    "value_loss": 0.2,
                    "outcome_bce": 0.5,
                    "total_loss": 2.0,
                    "outcome_count": 8,
                    "loss_finite": 1,
                    "gradient_finite": 1,
                    "optimizer_updates": 1,
                    # A valid first PPO update may have a near-zero scalar loss.
                    "actor_loss": 0.0,
                    "policy_tokens": policy_tokens,
                    "mean_ratio": 1.0,
                    "clip_fraction": 0.0,
                    "mean_advantage": 0.0,
                    "mean_abs_advantage": mean_abs_advantage,
                    "entropy": 0.7,
                    "gradient_l2": gradient_l2,
                    "gradient_parameter_count": 128,
                    "gradient_qwen_l2": qwen_gradient_l2,
                    "gradient_qwen_parameter_count": 64,
                }
            )
    program = tmp_path / "acceptance.py"
    program.write_text(_acceptance_program(), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(program),
            str(run),
            str(actor_enabled).lower(),
            credit_assignment,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_accepts_direct_ppo_with_zero_scalar_loss_and_real_gradient(
    tmp_path: Path,
) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=True,
        credit_assignment="turn",
    )

    assert result.returncode == 0, result.stderr
    result_payload = json.loads(
        (tmp_path / "run/two_update_probe_complete.json").read_text()
    )
    assert result_payload["actor_enabled"] is True


def test_launcher_rejects_direct_ppo_without_policy_tokens(tmp_path: Path) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=True,
        credit_assignment="turn",
        policy_tokens=0.0,
    )

    assert result.returncode != 0
    assert "no policy tokens" in result.stderr


def test_launcher_rejects_direct_ppo_without_qwen_gradient(tmp_path: Path) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=True,
        credit_assignment="turn",
        qwen_gradient_l2=0.0,
    )

    assert result.returncode != 0
    assert "zero Qwen gradient norm" in result.stderr


def test_launcher_rejects_direct_ppo_without_nonzero_advantages(
    tmp_path: Path,
) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=True,
        credit_assignment="turn",
        mean_abs_advantage=0.0,
    )

    assert result.returncode != 0
    assert "no nonzero advantages" in result.stderr


def test_launcher_keeps_non_actor_planner_probe_compatible(tmp_path: Path) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=False,
        credit_assignment="action",
        policy_tokens=0.0,
        gradient_l2=0.0,
    )

    assert result.returncode == 0, result.stderr


def test_launcher_requires_turn_credit_for_direct_qwen_ppo(tmp_path: Path) -> None:
    result = _run_acceptance(
        tmp_path,
        actor_enabled=True,
        credit_assignment="action",
    )

    assert result.returncode != 0
    assert "credit_assignment=turn" in result.stderr


def test_launcher_shell_and_embedded_python_parse() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    programs = re.findall(
        r"<<'PY'\n(.*?)\nPY",
        LAUNCHER.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert len(programs) == 2
    for program in programs:
        compile(program, str(LAUNCHER), "exec")
