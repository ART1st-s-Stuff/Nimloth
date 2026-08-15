from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id182_gate_phase.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id182_gate_on_hold.sh"
SLURM = ROOT / "experiments/training/rl/id182_k4_single_update_restore_gate.slurm"
VAGEN = ROOT / "external/VAGEN"
CONFIG = VAGEN / "vagen/configs/joint_id182_gate.yaml"
TRAIN = VAGEN / "examples/train/navigation/train_navigation_joint_id182.yaml"
VAL = VAGEN / "examples/train/navigation/val_navigation_joint_id182.yaml"


def test_id182_shells_parse_and_bind_exact_allocation() -> None:
    for path in (RUNNER, LAUNCHER, SLURM):
        subprocess.run(["bash", "-n", str(path)], check=True)
    slurm = SLURM.read_text()
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    for value in (
        "#SBATCH --partition=normal",
        "#SBATCH --gres=gpu:8",
        "#SBATCH --cpus-per-task=64",
        "#SBATCH --mem=256G",
        "#SBATCH --time=02:00:00",
        "dgx-13,dgx-23,dgx-32,dgx-37,dgx-51",
        "module load slurm",
    ):
        assert value in slurm
    for name in (
        "REPO",
        "EXPECTED_PARENT_COMMIT",
        "EXPECTED_VAGEN_COMMIT",
        "EXPECTED_VERL_COMMIT",
    ):
        assert name in slurm and name in launcher and name in runner
    assert "ReqTRES=[^ ]*mem=256G" in launcher
    assert "ReqTRES=[^ ]*mem=256G" in runner
    assert "MinMemoryNode=256G" in launcher and "MinMemoryNode=256G" in runner
    assert "AllocTRES=[^ ]*mem=256G" not in launcher + runner
    assert "TimeLimit=02:00:00" in launcher + runner


def test_id182_embedded_python_compiles() -> None:
    blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY", RUNNER.read_text(), flags=re.DOTALL)
    assert len(blocks) >= 5
    for index, block in enumerate(blocks):
        compile(block, f"{RUNNER}:inline-{index}", "exec")


def test_id182_pins_code_assets_and_two_separate_checkpoint_roots() -> None:
    source = RUNNER.read_text()
    for value in (
        "494f264494b2525f2c13595f63ac4912963e6d2f",
        "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
        "eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9",
        "dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b",
        "27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1",
        "63c933b6ebadae3ee64a4663b5bd1ec71676f64629faf2cda6c15393e534e563",
        "fcfec9497bc08d1faeb91c07e954b8a9638a1dfa7882f7c3f8b6824d269e2d51",
        "37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1",
        "cce9c81d6257e0f61dbedf6e075f9d873756f10ad0a98b72ea240788027c0e5e",
    ):
        assert value in source
    assert "ACTOR_MODEL=${REPAIR_ROOT}/checkpoint" in source
    assert "PLANNING_MODEL=${ROOT}/outputs/experiments" in source
    assert "ID182_ACTOR_MODEL=${ACTOR_MODEL}" in source
    assert "ID182_PLANNING_CHECKPOINT=${PLANNING_MODEL}" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "rsync" not in source and "scp " not in source


def test_id182_contract_is_one_update_then_restore_only() -> None:
    source = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    config = CONFIG.read_text()
    assert "PHASE=update_1" in launcher
    assert "PHASE=restore_only" in launcher
    assert "resume_update_2" not in source + launcher + config
    assert "trainer.total_training_steps=2" not in source + launcher
    assert "EXPECTED_STEP=1" in source
    assert "EXPECTED_SOURCE=777" in source
    assert "not (root/'global_step_2').exists()" in source
    assert "initial.snapshot_id!=updated.snapshot_id" in source
    assert "planning_optimizer_state']['state']" in source
    assert "Setting global step to 1" in source
    assert "ID182_K4_FRESH_RESTORE_ONLY_ALL_OK global_step=1" in source
    assert "/tmp/i182-" in source
    assert "/tmp/i181-" not in source
    trainer = (VAGEN / "vagen/ray_trainer.py").read_text()
    assert 'phase == "restore_only"' in trainer
    assert "restore-only gate did not load its complete target step" in trainer
    assert 'f"ID{self.joint_integration_gate.experiment_id}_"' in trainer
    assert "no canary, validation rollout, second update, or long training" in source
    assert "SIGREG_CUDA_FORWARD_BACKWARD_OK" in source
    assert "all(buffer.device==device for buffer in module.buffers())" in source
    assert "ID181 reached actor update before the rank-local empty failure" in source
    assert "rank-local empty K4 shard fix" in source


def test_id182_formal_values_and_three_split_data_are_explicit() -> None:
    config = CONFIG.read_text()
    train = TRAIN.read_text()
    for value in (
        "beta: 85.78297006578457",
        "planning_horizon: 4",
        "mcts_num_simulations: 100",
        "mcts_exploration_constant: 1.0",
        "gamma: 1.0",
        "gae_lambda: 0.95",
        "ppo_clip_ratio: 0.2",
        "token_kl_coefficient: 0.01",
        "guided_entropy_coefficient: 0.01",
        "lr: 1.0e-7",
        "projector_lr: 1.0e-4",
        "predictor_lr: 1.0e-4",
        "value_head_lr: 1.0e-4",
        "state_mse_weight: 1.0",
        "dino_grid_weight: 0.5",
        "sigreg_weight: 0.1",
        "temperature: 0.7",
        "top_p: 0.95",
        "max_response_length: 512",
        "train_batch_size: 24",
        "ppo_epochs: 1",
    ):
        assert value in config
    assert train.count("n_envs: 8") == 3
    assert train.count("seed: [0, 8]") == 3
    runner = RUNNER.read_text()
    assert "8 deterministic instances per split" in runner
    assert "inclusive seed directive [0,8]" in runner
    assert "seeds 0..7 each" not in runner
    assert "dataset_manifest.json" in runner
    assert "len(rows)==24" in runner
    assert "len({row['rollout_sample_id'] for row in rows})==24" in runner
    assert "sampled deterministically with replacement" in runner
    assert train.count("max_turns: 20") == 3
    assert train.count("per_turn_format_reward: 0.01") == 3
    assert train.count("\n      format_reward: 0.0") == 3
    assert train.count("success_reward: 1.0") == 3
    for split in ("base_train", "common_sense_train", "long_horizon_train"):
        assert f"eval_set: {split}" in train
    assert "validation is disabled" in VAL.read_text()


def test_general_production_gate_remains_closed() -> None:
    trainer = (VAGEN / "vagen/ray_trainer.py").read_text()
    gate = (VAGEN / "vagen/joint_policy/integration_gate.py").read_text()
    assert "refusing production training" in trainer
    assert "id182_k4_single_update_restore_gate_v1" in gate
    assert "restricted to experiment {experiment_id}" in gate
