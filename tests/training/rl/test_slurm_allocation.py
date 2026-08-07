from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "experiments/training/rl/slurm_allocation.sh"
CONTROLLER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_slurm.sh"
PIPELINE = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_smoke.sh"
FULL_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_online_ppo_full.sh"
WAIT_LAUNCHER = (
    REPO_ROOT / "experiments/training/rl/wait_for_1x8_hold_and_launch.sh"
)
PARALLEL_CONTROLLER = (
    REPO_ROOT
    / "experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh"
)
SHARD_RUNNER = REPO_ROOT / "experiments/training/rl/run_vllm_rollout_shard.sh"
CONTINUATION = REPO_ROOT / "src/nimloth/training/rl/continuation.py"
PPO_VALUE_GPU_GATE = (
    REPO_ROOT / "experiments/training/rl/gpu_gate_ppo_value_critic.slurm"
)
PPO_VALUE_STAGED_RUNNER = (
    REPO_ROOT
    / "experiments/training/rl/run_ppo_value_gc_gate_then_train_on_hold.sh"
)
PLANNER_POLICY_GPU_GATE_4X4 = (
    REPO_ROOT
    / "experiments/training/rl/run_planner_policy_gpu_gate_4x4_on_hold.sh"
)
HETERO_32_CONFIG = (
    REPO_ROOT / "configs/training/rl/planner_greedy_h1_full_32gpu_88844.yaml"
)
HETERO_32_BATCH = REPO_ROOT / "experiments/training/rl/train_true32_88844.slurm"
EIGHT_GPU_CONFIG = (
    REPO_ROOT / "configs/training/rl/planner_greedy_h1_full_8gpu_44.yaml"
)
EIGHT_GPU_BATCH = REPO_ROOT / "experiments/training/rl/train_8gpu_44.slurm"
ONE_NODE_EIGHT_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml"
)
ONE_NODE_EIGHT_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_8gpu_1x8.slurm"
)
HETERO_EIGHT_GPU_CONFIG = (
    REPO_ROOT / "configs/training/rl/planner_greedy_h1_full_8gpu_422.yaml"
)
SIXTEEN_ROLLOUT_HETERO_EIGHT_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_422.yaml"
)
HETERO_EIGHT_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_8gpu_422.slurm"
)
SIXTEEN_ROLLOUT_22_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_22gpu_8662.yaml"
)
SIXTEEN_ROLLOUT_22_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_22gpu_8662.slurm"
)
SIXTEEN_ROLLOUT_20_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_20gpu_8642.yaml"
)
SIXTEEN_ROLLOUT_20_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_20gpu_8642.slurm"
)
SIXTEEN_ROLLOUT_12_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_642.yaml"
)
SIXTEEN_ROLLOUT_12_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_12gpu_642.slurm"
)
SIXTEEN_ROLLOUT_12_GPU_6222_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_6222.yaml"
)
SIXTEEN_ROLLOUT_12_GPU_6222_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_12gpu_6222.slurm"
)
SIXTEEN_ROLLOUT_8_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_44.yaml"
)
SIXTEEN_ROLLOUT_24_GPU_CONFIG = (
    REPO_ROOT
    / "configs/training/rl/planner_greedy_h1_full_16rollout_24gpu_66642.yaml"
)
SIXTEEN_ROLLOUT_24_GPU_BATCH = (
    REPO_ROOT / "experiments/training/rl/train_24gpu_66642.slurm"
)


def _load_counts(job_details: str) -> list[str]:
    script = f"""
scontrol() {{
  [[ "$1 $2" == "show hostnames" ]] || return 1
  case "$3" in
    'dgx-[40,48]') printf '%s\\n' dgx-40 dgx-48 ;;
    dgx-10) printf '%s\\n' dgx-10 ;;
    dgx-12) printf '%s\\n' dgx-12 ;;
    *) return 1 ;;
  esac
}}
source {shlex.quote(str(HELPER))}
declare -A counts
nimloth_load_slurm_gpu_counts {shlex.quote(job_details)} counts
for node in "${{!counts[@]}}"; do
  printf '%s=%s\\n' "$node" "${{counts[$node]}}"
done | sort
"""
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_loads_gpu_counts_from_compressed_node_expression() -> None:
    details = """
JOB_GRES=gpu:8
  Nodes=dgx-[40,48] CPU_IDs=128-143 Mem=0 GRES=gpu:4(IDX:3,5-7)
"""

    assert _load_counts(details) == ["dgx-40=4", "dgx-48=4"]


def test_wait_launcher_gates_and_detaches_one_existing_1x8_hold() -> None:
    launcher = WAIT_LAUNCHER.read_text(encoding="utf-8")

    assert 'PENDING) sleep "${POLL_SECONDS}"' in launcher
    assert '[[ "${gpu_counts[${node}]:-}" == 8 ]]' in launcher
    assert 'for slot in 0 4; do' in launcher
    assert 'grep -Fq "\\"status\\": \\"AI2THOR_RENDER_OK\\""' in launcher
    assert "NODE_CLEAN_OK" in launcher
    assert 'nohup timeout --signal=TERM --kill-after=30s' in launcher
    assert 'srun --jobid="${HOLD_JOB}" --overlap' in launcher
    assert 'bash "${REPO}/experiments/training/rl/train_8gpu_1x8.slurm"' in launcher
    assert "scancel" not in launcher

    assert launcher.index("WANDB_IDENTITIES_OK") < launcher.index(
        "LAUNCH_PREFLIGHT_OK"
    )
    assert launcher.index("AI2THOR_RENDER_OK") < launcher.index(
        "LAUNCH_PREFLIGHT_OK"
    )
    assert launcher.index("NODE_CLEAN_OK") < launcher.index(
        "DETACHED_SRUN_LAUNCHED"
    )


def test_4plus4_batch_gates_each_rollout_node_renderer_before_training() -> None:
    batch = EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert ': "${EXPECTED_VAGEN_COMMIT:?}"' in batch
    assert ': "${EXPECTED_LEWM_COMMIT:?}"' in batch
    assert 'git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD' in batch
    assert 'git -C "${ENV_REPO}/external/le-wm" rev-parse HEAD' in batch
    assert "ENV_DEPENDENCIES_OK" in batch
    assert 'for node in "${ALLOCATED_NODES[@]}"; do' in batch
    assert "SLURM_RESTART_COUNT" in batch
    assert '(( ${#allocated_gpus[@]} == 4 ))' in batch
    assert 'export CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}"' in batch
    assert "nimloth.environment.navigation.direct_render_probe" in batch
    assert "grep -Fq '\"status\": \"AI2THOR_RENDER_OK\"'" in batch
    assert batch.index("ENV_DEPENDENCIES_OK") < batch.index(
        "WANDB_IDENTITIES_OK"
    )
    assert batch.index("ENV_DEPENDENCIES_OK") < batch.index(
        "RENDER_PREFLIGHT_ALL_OK"
    )
    assert batch.index("RENDER_PREFLIGHT_ALL_OK") < batch.index(
        "run_vllm_online_ppo_full.sh"
    )


def test_ppo_value_gpu_gate_requires_real_long_prefixes() -> None:
    gate = PPO_VALUE_GPU_GATE.read_text(encoding="utf-8")

    assert "MINIMUM_STATE_TOKENS=${MINIMUM_STATE_TOKENS:-14000}" in gate
    assert gate.count("--select-longest-final-transition") == 2
    assert gate.count('--minimum-state-tokens "${MINIMUM_STATE_TOKENS}"') == 2
    assert "transition_selection=global-qualified-longest-final" in gate


def test_planner_policy_gpu_gate_uses_every_gpu_in_4plus4_hold() -> None:
    runner = PLANNER_POLICY_GPU_GATE_4X4.read_text(encoding="utf-8")

    assert '[[ "${#NODES[@]}" == 2 ]]' in runner
    assert '[[ "${GPU_COUNTS[${node}]:-}" == 4 ]]' in runner
    assert '"2 4 2 8 4"' in runner
    assert "MINIMUM_STATE_TOKENS=${MINIMUM_STATE_TOKENS:-1}" in runner
    assert "nimloth.environment.navigation.direct_render_probe" in runner
    assert '"status": "AI2THOR_RENDER_OK"' in runner
    assert (
        'ROLLOUT_WORKERS=2 PIPELINE_MODE=train PIPELINE_PHASE=rollout'
        in runner
    )
    assert 'WORLD_SIZE=4 bash -lc' in runner
    assert 'for local_rank in 0 1; do' in runner
    assert '--gpus-per-rank 2' in runner
    assert 'for rank in range(4)' in runner
    assert 'FRESH_ROLLOUT_SOURCE=${FRESH_ROLLOUT_SOURCE:-}' in runner
    assert "reused fresh rollout already has consumption state" in runner
    assert "reused fresh rollout must be outside the new gate output" in runner
    assert "stage=rollout status=reused" in runner
    assert "validating all complete rank results" in runner
    assert '"controller_srun_exit": srun_exit' in runner
    assert "stage=ddp_step status=passed_with_srun_warning" in runner
    assert runner.index('stage=rollout status=starting') < runner.index(
        'stage=single_grad status=starting'
    )
    assert runner.index('stage=single_grad status=starting') < runner.index(
        'stage=ddp_step status=starting'
    )


def test_resumed_staged_pipeline_creates_a_new_first_iteration_output() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "FIRST_ITERATION=$((RUN_INITIAL_GLOBAL_STEP + 1))" in pipeline
    assert pipeline.count("ITERATION == FIRST_ITERATION") == 2
    assert "ITERATION == 1" not in pipeline


def test_parallel_controller_can_gate_between_rollout_and_training() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")

    assert "PIPELINE_PHASE=${PIPELINE_PHASE:-all}" in controller
    assert "rollout) RUN_ROLLOUT=true; RUN_TRAIN=false ;;" in controller
    assert "train) RUN_ROLLOUT=false; RUN_TRAIN=true ;;" in controller
    assert '[[ -s "${MANIFEST}" ]]' in controller
    assert '[[ -s "${TRAJECTORY_JSONL}" ]]' in controller
    assert '[[ ! -e "${CONSUMPTION}" ]]' in controller
    assert '[[ -s "${RESUME_CHECKPOINT}/rl_state.pt" ]]' in controller
    assert 'if [[ "${RUN_TRAIN}" == true ]]; then' in controller
    assert "ROLLOUT_STAGE_OK manifest=${MANIFEST}" in controller


def test_ppo_value_staged_runner_gates_before_consuming_fresh_rollout() -> None:
    runner = PPO_VALUE_STAGED_RUNNER.read_text(encoding="utf-8")

    rollout = runner.index("run_parallel_phase rollout disabled")
    gate = runner.index('bash "${GPU_GATE}"')
    train = runner.index("run_parallel_phase train online")
    assert rollout < gate < train
    assert (
        'TRAJECTORY_JSONL="${GATE_DIAGNOSTIC_TRAJECTORY_JSONL}"' in runner
    )
    assert 'FRESH_ROLLOUT_MANIFEST="${GATE_DIAGNOSTIC_MANIFEST}"' in runner
    assert (
        ': "${GATE_DIAGNOSTIC_TRAJECTORY_JSONL:?set the fixed long-prefix gate corpus}"'
        in runner
    )
    assert (
        ': "${GATE_DIAGNOSTIC_MANIFEST:?set the fixed gate corpus manifest}"'
        in runner
    )
    assert "gate diagnostic trajectory must be outside the formal RUN_OUT" in runner
    assert "gate diagnostic manifest must be outside the formal RUN_OUT" in runner
    assert "formal_train_trajectory=%s" in runner
    assert "MINIMUM_STATE_TOKENS=${MINIMUM_STATE_TOKENS:-14000}" in runner
    assert runner.count("stage=gpu_gate status=passed") == 1
    assert '[[ ! -e "${RUN_OUT}" ]]' in runner
    assert '[[ ! -e "${STAGE_LOG}" ]]' in runner


def test_loads_heterogeneous_per_node_gpu_counts() -> None:
    details = """
JOB_GRES=gpu:5
  Nodes=dgx-10 CPU_IDs=0-3 Mem=0 GRES=gpu:1(IDX:7)
  Nodes=dgx-12 CPU_IDs=0-15 Mem=0 GRES=gpu:4(IDX:0-3)
"""

    assert _load_counts(details) == ["dgx-10=1", "dgx-12=4"]


def test_ray_workers_receive_repo_pythonpath_and_are_import_probed() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")

    assert controller.count('env PYTHONPATH="${RAY_PYTHONPATH}"') == 2
    assert "def import_nimloth()" in controller
    assert "import_nimloth.options(resources={resource: 0.001})" in controller


def test_controller_pins_slurm_client_for_non_login_shells() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")

    assert "/cm/shared/apps/slurm/current/bin" in controller
    assert "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf" in controller
    assert 'export PATH="${SLURM_BIN_DIR}:${PATH}"' in controller
    assert controller.index('export PATH="${SLURM_BIN_DIR}:${PATH}"') < (
        controller.index("squeue -h")
    )


def test_pair_parallel_topology_is_config_driven_and_node_local() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "config.total_gpus" in controller
    assert "count % CONFIG_GPUS_PER_RANK" in controller
    assert "node_gpus % TRAIN_GPUS_PER_RANK" in pipeline
    assert "node_ranks=$((node_gpus / TRAIN_GPUS_PER_RANK))" in pipeline
    assert "local_rank<NIMLOTH_NODE_RANKS" in pipeline
    assert 'NIMLOTH_DDP_GPU_STRIDE="${TRAIN_GPUS_PER_RANK}"' in pipeline
    assert '--nproc_per_node="${TRAIN_WORLD_SIZE}"' in pipeline


def test_full_runner_uses_one_fresh_manifest_per_resumed_update() -> None:
    runner = FULL_RUNNER.read_text(encoding="utf-8")
    continuation = CONTINUATION.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "planner_greedy_h2_full.yaml" in runner
    assert (
        "for ((iteration=START_ITERATION; "
        "iteration<=TOTAL_ITERATIONS; iteration++))" in runner
    )
    assert 'prepare-policy "${RUN_OUT}" "${iteration}"' in runner
    assert "latest.rename(snapshot)" in continuation
    assert "relocate_consumption_checkpoint(" in continuation
    assert "validate_committed_iteration(" in continuation
    assert 'RESUME_CHECKPOINT="${resume_checkpoint}"' in runner
    assert 'SEED_OFFSET="${seed_offset}"' in runner
    assert '"${ITERATION_RUNNER}"' in runner
    assert controller.count('ITERATION="${ITERATION}" TOTAL_ITERATIONS="${TOTAL_ITERATIONS}"') == 3
    assert '--eval-sets "${TRAIN_DATASETS[@]}"' in pipeline
    assert '--rl-iterations "${ITERATION}"' in pipeline
    assert 'TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")' in pipeline
    assert 'TRAIN_ARGS+=(--defer-final-checkpoint)' in pipeline
    assert 'PREFLIGHT_OK commit=${COMMIT}' in pipeline


def test_full_runner_creates_new_date_parent_before_first_progress_write(
    tmp_path,
) -> None:
    formal_root = tmp_path / "formal"
    run_out = formal_root / "2026-08-06" / "id134"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-c" ]]; then
  printf '%s\n' '60 16 1 2 false true 10 120 base,common_sense'
  exit 0
fi
if [[ "$1 $2 $3" == "-m nimloth.training.rl.continuation prepare-run" ]]; then
  printf '%s\n' '0 1 0 -'
  exit 0
fi
exit 91
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    iteration_runner = tmp_path / "iteration-runner.sh"
    iteration_runner.write_text(
        "#!/usr/bin/env bash\nexit 42\n",
        encoding="utf-8",
    )
    iteration_runner.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "HOLD_JOB": "123",
        "REPO": str(REPO_ROOT),
        "ENV_REPO": str(REPO_ROOT),
        "PYTHON": str(fake_python),
        "RL_CONFIG": str(config),
        "RUN_OUT": str(run_out),
        "FORMAL_OUTPUT_ROOT": str(formal_root),
        "ITERATION_RUNNER": str(iteration_runner),
        "INITIAL_MODEL": str(model),
        "INITIAL_WM_CKPT": str(model),
        "INITIAL_RESUME_CHECKPOINT": "",
        "INITIAL_GLOBAL_STEP": "0",
        "TOTAL_ITERATIONS": "60",
        "REFERENCE_MODEL": str(model),
        "WANDB_PROJECT": "nimloth-rl",
        "WANDB_RUN_NAME": "id134",
    })

    result = subprocess.run(
        ["bash", str(FULL_RUNNER)],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42, result.stderr
    assert not run_out.exists()
    progress = Path(f"{run_out}.iteration_progress.log")
    assert progress.is_file()
    assert "iteration=1 status=starting" in progress.read_text(encoding="utf-8")
    assert "iteration=1 status=controller_failed exit=42" in progress.read_text(
        encoding="utf-8"
    )


def test_current_vllm_pipeline_preserves_checkpoint_processor_resolution() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "--max-pixels" not in pipeline


def test_parallel_controller_derives_tp4_workers_and_world_from_config() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")
    shard_runner = SHARD_RUNNER.read_text(encoding="utf-8")

    assert "CONFIG_TOTAL_GPUS == CONFIG_WORLD_SIZE * CONFIG_GPUS_PER_RANK" in controller
    assert "ROLLOUT_WORKERS=$((CONFIG_TOTAL_GPUS / TP_SIZE))" in controller
    assert 'workers_per_node=$((node_gpus / TP_SIZE))' in controller
    assert 'global_worker=$((NIMLOTH_WORKER_OFFSET + local_worker))' in controller
    assert 'NODE_SPECS+=("${node}:${node_gpus}:${het_group}")' in controller
    assert 'NIMLOTH_TRAIN_NODE_SPECS="${NIMLOTH_TRAIN_NODE_SPECS}"' in controller
    assert 'SHARD_GPU_VISIBLE="${shard_visible}"' in controller
    assert 'SHARD_SEED="${shard_seed}"' in controller
    assert 'SHARD_EVAL_SETS="${NIMLOTH_DATASETS}"' in controller
    assert 'SHARD_NUM_EPISODES="${NIMLOTH_EPISODES_PER_WORKER}"' in controller
    assert 'SHARD_SEED_PER_EVAL_SET=true' in controller
    assert 'PIPELINE_MODE}" == eval' in controller
    assert 'SHARD_NAVIGATION_PROFILE=vagen_eval' in controller
    assert 'SHARD_TEMPERATURE=0 SHARD_TOP_P=1' in controller
    assert 'merge_rollout_shards.py' in controller
    assert 'PIPELINE_PHASE=train' in controller
    assert '--vllm-distributed-executor-backend mp' in shard_runner
    assert 'export VLLM_WORKER_MULTIPROC_METHOD=spawn' in shard_runner
    assert '--num-episodes "${SHARD_NUM_EPISODES}"' in shard_runner
    assert '--navigation-profile "${SHARD_NAVIGATION_PROFILE}"' in shard_runner


def test_true32_heterogeneous_topology_is_explicitly_routed_by_het_group() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    config = HETERO_32_CONFIG.read_text(encoding="utf-8")
    batch = HETERO_32_BATCH.read_text(encoding="utf-8")

    assert 'SLURM_JOB_NODELIST_HET_GROUP_${het_group}' in controller
    assert 'SRUN_ARGS+=(--het-group="${het_group}")' in controller
    assert 'NIMLOTH_HET_GPUS_PER_NODE' in controller
    assert 'TRAIN_HET_GROUPS[${node}]=${het_group}' in pipeline
    assert 'RDZV_SRUN_ARGS+=(--het-group="${head_het_group}")' in pipeline
    assert 'TRAIN_SRUN_ARGS+=(--het-group="${het_group}")' in pipeline
    assert "nodes: 5" in config
    assert "world_size: 16" in config
    assert "gpus_per_rank: 2" in config
    assert "rollout_tensor_parallel_size: 4" in config
    assert 'test "${SLURM_HET_SIZE}" -eq 2' in batch
    assert 'test "${#HET_NODES_8[@]}" -eq 3' in batch
    assert 'test "${#HET_NODES_4[@]}" -eq 2' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=8,4" in batch
    assert 'exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"' in batch


def test_eight_gpu_422_routes_one_tp4_worker_and_four_training_ranks() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")
    config = HETERO_EIGHT_GPU_CONFIG.read_text(encoding="utf-8")
    batch = HETERO_EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert "nodes: 3" in config
    assert "world_size: 4" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_4[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 2 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=4,2" in batch
    assert "export ROLLOUT_WORKERS=1" in batch
    assert "--ignore-submodules=untracked" in batch
    assert "node_gpus % CONFIG_GPUS_PER_RANK" in controller
    assert 'EVAL_ALL_DATASETS_PER_WORKER=true' in controller
    assert 'dataset=${NIMLOTH_EVAL_DATASETS}' in controller


def test_eight_gpu_44_batch_and_config_preserve_parallel_contract() -> None:
    config = EIGHT_GPU_CONFIG.read_text(encoding="utf-8")
    batch = EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert "nodes: 2" in config
    assert "world_size: 4" in config
    assert "gpus_per_rank: 2" in config
    assert "rollout_tensor_parallel_size: 4" in config
    assert '[[ "${#ALLOCATED_NODES[@]}" == 2 ]]' in batch
    assert '[[ "${GPU_COUNTS[${node}]:-}" == 4 ]]' in batch
    assert "export ROLLOUT_WORKERS=2" in batch
    assert 'exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"' in batch


def test_16rollout_8gpu_422_routes_one_tp4_worker_and_four_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_HETERO_EIGHT_GPU_CONFIG.read_text(encoding="utf-8")
    batch = HETERO_EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "max_state_tokens: 16384" in config
    assert "max_episode_attempts: 3" in config
    assert "nodes: 3" in config
    assert "world_size: 4" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_4[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 2 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=4,2" in batch
    assert "export ROLLOUT_WORKERS=1" in batch


def test_16rollout_8gpu_44_routes_two_tp4_workers_and_four_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_8_GPU_CONFIG.read_text(encoding="utf-8")
    batch = EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "max_state_tokens: 16384" in config
    assert "max_episode_attempts: 3" in config
    assert "nodes: 2" in config
    assert "world_size: 4" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#ALLOCATED_NODES[@]}" == 2 ]]' in batch
    assert '[[ "${GPU_COUNTS[${node}]:-}" == 4 ]]' in batch
    assert "export ROLLOUT_WORKERS=2" in batch


def test_16rollout_8gpu_1x8_routes_two_tp4_workers_and_four_training_ranks() -> None:
    config = ONE_NODE_EIGHT_GPU_CONFIG.read_text(encoding="utf-8")
    batch = ONE_NODE_EIGHT_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "max_state_tokens: 16384" in config
    assert "nodes: 1" in config
    assert "world_size: 4" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#ALLOCATED_NODES[@]}" == 1 ]]' in batch
    assert '[[ "${GPU_COUNTS[${node}]:-}" == 8 ]]' in batch
    assert "export ROLLOUT_WORKERS=2" in batch
    assert "run_vllm_online_ppo_parallel_slurm.sh" in batch
    assert "requires the two-TP4 parallel runner" in batch
    assert "max_episode_attempts: 3" in config
    assert '--max-episode-attempts "${MAX_EPISODE_ATTEMPTS}"' in (
        SHARD_RUNNER.read_text(encoding="utf-8")
    )


def test_formal_batches_allow_empty_fresh_resume_checkpoint() -> None:
    batches = []
    for batch_path in sorted(
        (REPO_ROOT / "experiments/training/rl").glob("*.slurm")
    ):
        if "INITIAL_RESUME_CHECKPOINT" in batch_path.read_text(encoding="utf-8"):
            batches.append(batch_path)

    assert batches
    for batch_path in batches:
        batch = batch_path.read_text(encoding="utf-8")
        assert ': "${INITIAL_RESUME_CHECKPOINT?}"' in batch, batch_path.name
        assert ': "${INITIAL_RESUME_CHECKPOINT:?}"' not in batch, batch_path.name

        gate = batch.split("export SLURM_CONF", maxsplit=1)[0]
        required_names = re.findall(r'^: "\$\{([A-Z0-9_]+)[^}]*\}"$', gate, re.MULTILINE)
        environment = os.environ.copy()
        environment.update({name: "set" for name in required_names})
        environment["INITIAL_RESUME_CHECKPOINT"] = ""
        result = subprocess.run(
            ["bash"],
            input=gate,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (batch_path.name, result.stderr)


def test_22gpu_8662_routes_four_tp4_workers_and_eleven_training_ranks() -> None:
    controller = PARALLEL_CONTROLLER.read_text(encoding="utf-8")
    config = SIXTEEN_ROLLOUT_22_GPU_CONFIG.read_text(encoding="utf-8")
    batch = SIXTEEN_ROLLOUT_22_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "nodes: 4" in config
    assert "world_size: 11" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_8[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_6[@]}" == 2 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 1 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=8,6,2" in batch
    assert "export ROLLOUT_WORKERS=4" in batch
    assert "--ignore-submodules=untracked" in batch
    assert "node_gpus >= TP_SIZE" not in controller
    assert "allocation_workers=$((allocation_workers + node_gpus / TP_SIZE))" in controller


def test_20gpu_8642_routes_four_tp4_workers_and_ten_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_20_GPU_CONFIG.read_text(encoding="utf-8")
    batch = SIXTEEN_ROLLOUT_20_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "nodes: 4" in config
    assert "world_size: 10" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_8[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_6[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_4[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 1 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=8,6,4,2" in batch
    assert "export ROLLOUT_WORKERS=4" in batch
    assert "--ignore-submodules=untracked" in batch


def test_12gpu_642_routes_two_tp4_workers_and_six_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_12_GPU_CONFIG.read_text(encoding="utf-8")
    batch = SIXTEEN_ROLLOUT_12_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "nodes: 3" in config
    assert "world_size: 6" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_6[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_4[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 1 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=6,4,2" in batch
    assert "export ROLLOUT_WORKERS=2" in batch
    assert "--ignore-submodules=untracked" in batch


def test_12gpu_6222_routes_one_tp4_worker_and_six_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_12_GPU_6222_CONFIG.read_text(encoding="utf-8")
    batch = SIXTEEN_ROLLOUT_12_GPU_6222_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "max_state_tokens: 16384" in config
    assert "nodes: 4" in config
    assert "world_size: 6" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_6[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 3 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=6,2" in batch
    assert "export ROLLOUT_WORKERS=1" in batch
    assert "--ignore-submodules=untracked" in batch


def test_24gpu_66642_routes_four_tp4_workers_and_twelve_training_ranks() -> None:
    config = SIXTEEN_ROLLOUT_24_GPU_CONFIG.read_text(encoding="utf-8")
    batch = SIXTEEN_ROLLOUT_24_GPU_BATCH.read_text(encoding="utf-8")

    assert "envs_per_iteration: 16" in config
    assert "batch_size: 16" in config
    assert "nodes: 5" in config
    assert "world_size: 12" in config
    assert "gpus_per_rank: 2" in config
    assert '[[ "${#HET_NODES_6[@]}" == 3 ]]' in batch
    assert '[[ "${#HET_NODES_4[@]}" == 1 ]]' in batch
    assert '[[ "${#HET_NODES_2[@]}" == 1 ]]' in batch
    assert "export NIMLOTH_HET_GPUS_PER_NODE=6,4,2" in batch
    assert "export ROLLOUT_WORKERS=4" in batch
    assert "--ignore-submodules=untracked" in batch
