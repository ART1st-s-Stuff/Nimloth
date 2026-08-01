"""Contracts for the on-hold 4+4+4+2+1+1 world16 launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCH = ROOT / "experiments/training/sft2/launch_dino_grid_ws16_444211_on_hold.sh"
NODE = ROOT / "experiments/training/sft2/run_dino_grid_ws16_single_gpu_agent.sh"
PREFLIGHT = ROOT / "experiments/training/sft2/preflight_dino_grid_ws16_444211.slurm"
VALIDATOR = ROOT / "experiments/training/sft2/validate_ws16_444211_allocation.py"


def test_ws16_444211_contract() -> None:
    launch = LAUNCH.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert "--het-group=0" in launch
    assert "--het-group=1" in launch
    assert "--het-group=2" in launch
    assert "--ntasks=12" in launch
    assert "--ntasks=2" in launch
    assert "--gpus-per-task=1" in launch
    assert "run_agents probe" in launch
    assert "run_agents train" in launch
    assert "NPROC_PER_NODE=1" in node
    assert "NNODES=16" in node
    assert "RESUME=0" in node
    assert "assert ranks == set(range(16))" in validator
    assert "assert len(all_uuids) == 16" in validator


def test_ws16_444211_preflight_contract() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "--world-size 16" in text
    assert "--gpus-per-node-list 4,4,4,2,1,1" in text
    assert "--agent-count 16" in text
    assert "--gpus-per-agent 1" in text


def test_ws16_444211_scripts_expand_runtime_variables() -> None:
    for path in (LAUNCH, NODE, PREFLIGHT):
        assert r"\${" not in path.read_text(encoding="utf-8")
