from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_beta_calibration.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_beta_calibration_on_hold.sh"
HOLD = ROOT / "experiments/training/rl/hold_eight_gpu_60m_k4_calibration.slurm"
ENTRYPOINT = ROOT / "external/VAGEN/vagen/k4_beta_calibration.py"


def test_k4_calibration_shells_parse_and_bind_target_allocation() -> None:
    for path in (RUNNER, LAUNCHER, HOLD):
        subprocess.run(["bash", "-n", str(path)], check=True)
    hold = HOLD.read_text()
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "#SBATCH --partition=normal" in hold
    assert "#SBATCH --gres=gpu:8" in hold
    assert "#SBATCH --cpus-per-task=64" in hold
    assert "#SBATCH --mem=256G" in hold
    assert "#SBATCH --time=01:00:00" in hold
    assert "dgx-13,dgx-23,dgx-32,dgx-37,dgx-51" in hold
    assert "nimloth-id173-k4-calibration" in hold
    assert "sleep infinity" not in hold
    assert "launch_vagen_k4_beta_calibration_on_hold.sh" in hold
    assert '"${SLURM_JOB_ID}"' in hold
    assert "ReqTRES=[^ ]*mem=256G" in launcher
    assert "MinMemoryNode=256G" in launcher
    assert "AllocTRES=[^ ]*mem=256G" not in launcher
    assert "ReqTRES=[^ ]*mem=256G" in runner
    assert "MinMemoryNode=256G" in runner
    assert "AllocTRES=[^ ]*mem=256G" not in runner
    assert "--gres=gpu:8" in launcher
    assert "TimeLimit=01:00:00" in launcher


def test_k4_calibration_embedded_python_compiles() -> None:
    blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY", RUNNER.read_text(), flags=re.DOTALL)
    assert len(blocks) == 6
    for index, block in enumerate(blocks):
        compile(block, f"{RUNNER}:inline-{index}", "exec")


def test_k4_calibration_pins_code_data_checkpoint_and_clean_worktrees() -> None:
    source = RUNNER.read_text()
    for name in (
        "EXPECTED_PARENT_COMMIT",
        "EXPECTED_VAGEN_COMMIT",
        "EXPECTED_VERL_COMMIT",
    ):
        assert name in source
    assert "494f264494b2525f2c13595f63ac4912963e6d2f" in source
    assert "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "checkpoint_preflight.json" in source
    assert "source_hashes.json" in source
    assert "eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9" in source
    assert "dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b" in source
    assert "27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1" in source
    assert "rsync" not in source
    assert "scp " not in source
    assert 'COMMAND=(' in source and '"${COMMAND[@]}"' in source
    assert "RUNTIME_ROOT=/tmp/i173-${SLURM_JOB_ID}" in source
    assert "RAY_TMPDIR=${RUNTIME_ROOT}" in source
    worst_case_socket = (
        "/tmp/i173-99999999/ray/session_2026-08-15_00-40-07_"
        "477970_1313372/sockets/plasma_store"
    )
    assert len(worst_case_socket.encode()) <= 107


def test_k4_calibration_values_are_explicit_and_optimizer_free() -> None:
    source = RUNNER.read_text()
    exact = (
        "--joint-run-seed 172001",
        "--joint-alpha 1.0",
        "--joint-beta 0.0",
        "--joint-prior-temperature 1.0",
        "--joint-score-dtype float32",
        "--planning-horizon 4",
        "--mcts-num-simulations 100",
        "--mcts-exploration-constant 1.0",
        "--trajectory-count 24",
        "--seeds-per-split 8",
        "--seed-start 0",
        "--max-turns 20",
        "--response-length 512",
        "--temperature 0.7",
        "--top-p 0.95",
        "--per-turn-format-reward 0.01",
        "--format-reward 0.0",
        "--success-reward 1.0",
        "--tensor-parallel-size 8",
        "--max-num-seqs 24",
        "--agent-loop-num-workers 24",
        "--minimum-median-planner-spread 1e-8",
    )
    for value in exact:
        assert value in source
    assert "WANDB_MODE=disabled" in source
    assert "No optimizer, backward, parameter update, checkpoint, resume" in source
    assert "canary_started':False" in source
    entrypoint = ENTRYPOINT.read_text()
    assert '"optimizer": None' in entrypoint
    assert '"checkpoint_output": None' in entrypoint
    assert "calibrated_beta_requires_human_approval" in entrypoint
    assert "median_prior / median_planner" in entrypoint
    assert "torch.optim" not in entrypoint
