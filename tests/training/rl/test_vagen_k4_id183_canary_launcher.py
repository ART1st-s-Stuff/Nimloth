from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[3]
VAGEN = ROOT / "external/VAGEN"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id183_canary_phase.sh"
PHASE1_LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id183_canary_phase1_on_hold.sh"
PHASE2_LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id183_canary_phase2_on_hold.sh"
COMMON_LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id183_canary_multinode_on_hold.sh"
PHASE1_SLURM = ROOT / "experiments/training/rl/id183_k4_canary_phase1.slurm"
PHASE2_SLURM = ROOT / "experiments/training/rl/id183_k4_canary_phase2.slurm"
CONFIG = VAGEN / "vagen/configs/joint_id183_canary.yaml"
TRAIN = VAGEN / "examples/train/navigation/train_navigation_joint_id183.yaml"
VAL = VAGEN / "examples/train/navigation/val_navigation_joint_id183.yaml"


def test_shells_parse_and_neither_launcher_can_run_both_phases() -> None:
    for path in (
        RUNNER,
        PHASE1_LAUNCHER,
        PHASE2_LAUNCHER,
        COMMON_LAUNCHER,
        PHASE1_SLURM,
        PHASE2_SLURM,
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)
    first = PHASE1_LAUNCHER.read_text()
    second = PHASE2_LAUNCHER.read_text()
    assert "PHASE=train_to_5" in first
    assert "PHASE=resume_to_10" not in first
    assert "PHASE=resume_to_10" in second
    assert "PHASE=train_to_5" not in second
    for slurm in (PHASE1_SLURM.read_text(), PHASE2_SLURM.read_text()):
        for value in (
            "#SBATCH --partition=normal",
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks=2",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --gres=gpu:4",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=128G",
            "#SBATCH --time=05:00:00",
            "dgx-13,dgx-23,dgx-32,dgx-37,dgx-51",
            "module load slurm",
        ):
            assert value in slurm


def test_launchers_build_exact_two_by_four_ray_cluster() -> None:
    source = COMMON_LAUNCHER.read_text()
    for value in (
            "NumNodes=2",
            "gres/gpu=8",
            "cpu=64",
            "mem=256G",
            "MinMemoryNode=128G",
            "nimloth_load_slurm_gpu_counts",
            "RAY_EXPECTED_NODE_IPS",
            "NCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME",
            "VLLM_HOST_IP",
            "SLURM_BIN_DIR=/cm/shared/apps/slurm/current/bin",
            "SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf",
            'PATH="${SLURM_BIN_DIR}:${ROOT}/.venv-vagen-main/bin:/usr/bin:/bin"',
            "HF_HOME=/project/peilab/atst/.cache/huggingface",
            "TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface",
            "TORCH_HOME=/project/peilab/atst/flower/.cache/torch",
            "VLLM_WORKER_MULTIPROC_METHOD=spawn",
            'source /project/peilab/atst/flower/.env',
            "wandb_api_key_present",
            "ID183_TRAIN_CONFIG=\"${PHASE_OUT}/train_navigation_joint_id183.yaml\"",
            "ray.scripts.scripts start --block --head",
            '--address="${RAY_ADDRESS}"',
            "ray.init(address=os.environ['RAY_ADDRESS'])",
            "counts != [4.0, 4.0]",
            "ID183_EXPECTED_NNODES=2",
            "ID183_EXPECTED_GPUS_PER_NODE=4",
        ):
        assert value in source
    assert "NumNodes=1" not in source
    assert "--gres=gpu:8" not in source
    assert "counts != [4.0, 4.0]" in source
    assert 'JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")' in source
    assert 'scontrol show job -dd "${HOLD_JOB}" -o' not in source
    assert "flock -n 9" in source
    assert "duplicate ID183 launcher" in source
    assert "--overlap" in source
    assert "timeout 120s srun" in source
    assert "for _ in $(seq 1 20)" in source
    assert '"${cleanup_empty}" != true && "${status}" -eq 0' in source
    runner = RUNNER.read_text()
    assert "ENV_URL=http://${ID183_HEAD_IP}:${ENV_PORT}" in runner
    assert '--host="${ID183_HEAD_IP}"' in runner
    assert '[[ "${ID183_EXPECTED_NNODES}" == 2 ]]' in runner
    assert '[[ "${ID183_EXPECTED_GPUS_PER_NODE}" == 4 ]]' in runner
    assert "SLURM_MEM_PER_NODE" not in runner
    assert "MinMemoryNode=128G" in runner
    assert "ID183_RAY_2X4_OK" in runner
    assert '--ntasks-per-node=1 --gres=gpu:4 --label' in runner


def test_runner_has_exact_resume_and_checkpoint_boundaries() -> None:
    source = RUNNER.read_text()
    for value in (
        "phase1_train_to_5",
        "phase2_fresh_resume_to_10",
        "global_step_5/joint_checkpoint_complete.json",
        "load_complete_joint_checkpoint(root/f'global_step_{value}')",
        "EXPECTED_STEP=5",
        "EXPECTED_STEP=10",
        "EXPECTED_SOURCE=781",
        "EXPECTED_SOURCE=786",
        "trainer.total_training_steps=10",
        "trainer.total_epochs=10",
        "trainer.resume_mode=auto",
        "trainer.val_before_train=false",
        "trainer.test_freq=10",
        "validation/0.jsonl",
        "validation/10.jsonl",
        "summarize_canary_validation_rows",
    ):
        assert value in source
    for forbidden_step in (1, 2, 3, 4, 6, 7, 8, 9):
        assert not re.search(
            rf"global_step_{forbidden_step}(?:/|['\"]|$)",
            source,
        )
    assert "global_step_5" in source and "global_step_10" in source
    assert "5x60" in source
    assert "must not start" in source


def test_phase_walltime_contains_all_hard_timeout_budgets() -> None:
    source = RUNNER.read_text()
    # Conservative bound: 13,200s train, 30s timeout kill, 150s render,
    # 90*(5s curl + 2s sleep) health, eight 300s+10s-kill prewarms,
    # 127s phase cleanup, 130s W&B convergence, and a conservative 1000s
    # external-Ray fabric/bootstrap/probes/cluster-cleanup allowance.
    assert "TimeLimit=05:00:00" in source
    assert "PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS:-13200}" in source
    assert 5 * 3600 >= 13200 + 30 + 150 + 630 + 8 * 310 + 127 + 130 + 1000


def test_wandb_identity_is_new_then_must_resume() -> None:
    source = RUNNER.read_text()
    assert "WANDB_RUN_ID=nimloth-id183-k4-10update-canary" in source
    assert "WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology" in source
    assert "ID183 W&B identity already exists" in source
    assert "ID183 phase2 W&B preflight mismatch" in source
    assert "steps!=list(range(6))" in source
    assert "steps==list(range(max_step+1))" in source
    assert "export WANDB_RESUME=never" in source
    assert "export WANDB_RESUME=must" in source
    assert source.index("WANDB_RESUME=never") < source.index("WANDB_RESUME=must")


def test_cleanup_tracks_runtime_processes_by_proc_environ() -> None:
    source = RUNNER.read_text()
    assert "runtime_process_ids()" in source
    assert "Path('/proc').iterdir()" in source
    assert "root in environ or root in cmdline" in source
    assert 'pgrep -f "${RUNTIME_ROOT}"' not in source
    assert "terminate_group \"${TRAIN_PID}\"" in source
    assert "owned_processes_after.log" in source


def test_embedded_python_compiles_and_hashes_all_assets() -> None:
    source = RUNNER.read_text()
    blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        compile(block, f"{RUNNER}:inline-{index}", "exec")
    common_source = COMMON_LAUNCHER.read_text()
    common_blocks = re.findall(
        r"<<'PY'.*?\n(.*?)\nPY",
        common_source,
        flags=re.DOTALL,
    )
    assert len(common_blocks) >= 4
    for index, block in enumerate(common_blocks):
        compile(block, f"{COMMON_LAUNCHER}:inline-{index}", "exec")
    expected_hashes = {
        "base_train": "eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9",
        "common_sense_train": "dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b",
        "long_horizon_train": "27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1",
        "base": "6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a",
        "common_sense": "3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf",
        "long_horizon": "ff23dcb171ff8008721a8a74ee7c677f6535f84c25407715326a8b313771bdaf",
        "complex_instruction": "767730e5b83812a199a27be41477da98def2becfd5cc8bd3e45d8cfdce260b9b",
        "visual_appearance": "e66bc8aab0141c662761ef1a1d857aa6297972c6a0890526b008990eded8ddc1",
    }
    for digest in expected_hashes.values():
        assert digest in source


def test_data_contract_is_train_three_by_eight_and_heldout_five_by_eight() -> None:
    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert len(train) == 3
    assert len(val) == 5
    assert {row["config"]["eval_set"] for row in train} == {
        "base_train",
        "common_sense_train",
        "long_horizon_train",
    }
    assert {row["config"]["eval_set"] for row in val} == {
        "base",
        "common_sense",
        "long_horizon",
        "complex_instruction",
        "visual_appearance",
    }
    for row in [*train, *val]:
        assert row["n_envs"] == 8
        assert row["max_turns"] == 20
        assert row["config"]["per_turn_format_reward"] == 0.01
        assert row["config"]["format_reward"] == 0.0
        assert row["config"]["success_reward"] == 1.0
    for row in val:
        assert row["seed_list"][:8] == list(range(8))
        assert len(row["seed_list"]) > row["n_envs"]


def test_config_is_ten_update_canary_and_stops_before_full_evaluation() -> None:
    source = CONFIG.read_text()
    for value in (
        "id183_k4_10update_canary_v1",
        "beta: 85.78297006578457",
        "checkpoint_frequency: 5",
        "train_batch_size: 24",
        "val_batch_size: 40",
        "total_training_steps: 5",
        "total_epochs: 5",
        "val_before_train: true",
        "test_freq: -1",
        "save_freq: 5",
        "max_actor_ckpt_to_keep: 2",
        "temperature: 0.7",
        "top_p: 0.95",
        "nnodes: 2",
        "n_gpus_per_node: 4",
        "address: auto",
    ):
        assert value in source
    assert "production" not in source.lower()
    runner = RUNNER.read_text()
    assert "second update" not in runner
    assert "long training" in runner
    assert "5x60 held-out evaluation" in runner


def test_heldout_scenes_are_disjoint_from_training_scenes() -> None:
    asset_root = VAGEN / "vagen/envs/navigation/assets"
    train_scenes = set()
    for name in ("base_train", "common_sense_train", "long_horizon_train"):
        tasks = json.loads((asset_root / f"{name}.json").read_text())["tasks"]
        assert len(tasks) == 1200
        train_scenes.update(item["scene"] for item in tasks)
    for name in (
        "base",
        "common_sense",
        "long_horizon",
        "complex_instruction",
        "visual_appearance",
    ):
        tasks = json.loads((asset_root / f"{name}.json").read_text())["tasks"]
        scenes = {item["scene"] for item in tasks}
        assert len(tasks) == 60
        assert len(scenes) == 60
        assert scenes.isdisjoint(train_scenes)
