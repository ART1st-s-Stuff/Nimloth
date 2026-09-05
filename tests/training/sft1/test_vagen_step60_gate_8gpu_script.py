from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = (
    ROOT
    / "experiments"
    / "training"
    / "sft1"
    / "vagen_step60_gate_8gpu_preempt.slurm"
)


def _load_state_module():
    name = "vagen_step60_shard_state_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "experiments" / "training" / "sft1" / "vagen_step60_shard_state.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, fail_smoke: bool = False) -> tuple[dict[str, str], Path]:
    tools = tmp_path / "tools"
    tools.mkdir(parents=True)
    events = tmp_path / "events"
    run_root = tmp_path / "run"
    for name in ("model", "vagen", "nimloth"):
        (tmp_path / name).mkdir()
    subprocess.run(["git", "init", "-q", str(tmp_path / "nimloth")], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path / "nimloth"), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path / "nimloth"), "config", "user.name", "Test"],
        check=True,
    )
    (tmp_path / "nimloth" / "tracked").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path / "nimloth"), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(tmp_path / "nimloth"), "commit", "-qm", "fixture"], check=True)
    nimloth_commit = subprocess.check_output(
        ["git", "-C", str(tmp_path / "nimloth"), "rev-parse", "HEAD"], text=True
    ).strip()
    for name in ("partition.json", "contract.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    python = tmp_path / "python-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(python, "#!/usr/bin/env bash\nexec \"$@\"\n")
    _write_executable(tools / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        tools / "env-service",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        mkdir -p "$ROLLOUT_RUN_DIR/external_env_4gpu"
        : > "$ROLLOUT_RUN_DIR/external_env_4gpu/env_urls.txt"
        for port in 8400 8401 8402 8403; do
          echo "http://127.0.0.1:$port" >> "$ROLLOUT_RUN_DIR/external_env_4gpu/env_urls.txt"
        done
        printf 'env:%s:%s:%s\n' "$CUDA_VISIBLE_DEVICES" "$VAGEN_DIR" "$PYTHON_ENV" >> "$EVENTS"
        trap 'printf "env-stopped\\n" >> "$EVENTS"; exit 0' TERM INT
        touch "$ROLLOUT_RUN_DIR/external_env_4gpu/ready"
        while true; do sleep 1; done
        """,
    )
    # It is invoked through bash; executable mode is intentionally absent.
    (tools / "env-service").chmod(0o644)
    _write_executable(
        tools / "collector",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        output= shard= source= resume=0 env_url= handoff= handoff_sha=
        while (($#)); do
          case "$1" in
            --output-dir) output=$2; shift 2;;
            --shard-index) shard=$2; shift 2;;
            --source-index) source=$2; shift 2;;
            --env-url) env_url=$2; shift 2;;
            --inspection-handoff) handoff=$2; shift 2;;
            --expected-inspection-handoff-sha256) handoff_sha=$2; shift 2;;
            --resume) resume=1; shift;;
            *) shift;;
          esac
        done
        label=${source:-shard-$shard}
        [[ -f "$handoff" && "$(sha256sum "$handoff" | awk '{print $1}')" == "$handoff_sha" ]] || exit 27
        printf 'collector:%s:%s:%s:%s:%s\n' "$label" "$CUDA_VISIBLE_DEVICES" "$env_url" "$resume" "$output" >> "$EVENTS"
        if [[ -n "$source" && "${FAIL_SMOKE:-0}" == 1 ]]; then exit 19; fi
        if [[ -n "$source" && "${SLOW_SMOKE:-0}" == 1 ]]; then
          trap 'printf "cancelled:%s\\n" "$label" >> "$EVENTS"; exit 143' TERM INT
          sleep 10
        fi
        if [[ -n "${FAIL_GATE_SHARD:-}" && "${FAIL_GATE_SHARD}" == "$shard" ]]; then
          if [[ "${SPAWN_ORPHAN_ON_FAIL:-0}" == 1 ]]; then
            (trap 'printf "orphan-stopped:%s\\n" "$label" >> "$EVENTS"; exit 0' TERM INT; while true; do sleep 1; done) &
          fi
          exit 23
        fi
        if [[ -z "$source" && "${SLOW_GATE:-0}" == 1 ]]; then
          trap 'printf "cancelled:%s\\n" "$label" >> "$EVENTS"; exit 143' TERM INT
          sleep 10
        fi
        rm -rf "${output}.inprogress"
        mkdir -p "$output"
        touch "$output/COMPLETE"
        """,
    )
    _write_executable(
        tools / "validator",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        mode=$1; shift
        printf 'helper:%s\n' "$mode" >> "$EVENTS"
        if [[ "$mode" == initialize-run ]]; then
          root=$1; shift
          run_mode= expected= identity=
          while (($#)); do
            case "$1" in
              --run-mode) run_mode=$2; shift 2;;
              --expected-nimloth-commit) expected=$2; shift 2;;
              --identity-field) identity+="$2"$'\n'; shift 2;;
              *) shift;;
            esac
          done
          [[ -n "$expected" && "$(git -C "$NIMLOTH_ROOT" rev-parse HEAD)" == "$expected" ]] || exit 30
          [[ -z "$(git -C "$NIMLOTH_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || exit 30
          if [[ "$run_mode" == fresh ]]; then
            mkdir "$root" || exit 31
            mkdir "$root/attempts" "$root/smoke" "$root/gate"
            printf '%s' "$identity" > "$root/RUN_IDENTITY"
          else
            [[ ! -e "$root/NON_RESUMABLE.json" ]] || exit 38
            [[ -f "$root/RUN_IDENTITY" && "$(cat "$root/RUN_IDENTITY")"$'\n' == "$identity" ]] || exit 32
          fi
          exit 0
        fi
        if [[ "$mode" == finalize-run ]]; then
          root=$1; shift
          exit_code= attempt= orchestrator_signal=
          while (($#)); do
            case "$1" in
              --exit-code) exit_code=$2; shift 2;;
              --attempt-id) attempt=$2; shift 2;;
              --orchestrator-signal) orchestrator_signal=$2; shift 2;;
              *) shift;;
            esac
          done
          [[ "${FAIL_FINALIZE:-0}" == 1 ]] && exit 41
          if [[ "$exit_code" == 143 && "$orchestrator_signal" =~ ^(TERM|INT)$ ]]; then exit 0; fi
          printf '{"attempt":"%s","exit_code":%s,"orchestrator_signal":"%s"}\n' "$attempt" "$exit_code" "$orchestrator_signal" > "$root/NON_RESUMABLE.json"
          exit 0
        fi
        if [[ "$mode" == verify-nimloth ]]; then
          root= expected=
          while (($#)); do
            case "$1" in
              --nimloth-root) root=$2; shift 2;;
              --expected-nimloth-commit) expected=$2; shift 2;;
              *) shift;;
            esac
          done
          [[ "$(git -C "$root" rev-parse HEAD)" == "$expected" ]] || exit 36
          [[ -z "$(git -C "$root" status --porcelain=v1 --untracked-files=all)" ]] || exit 36
          exit 0
        fi
        items=() handoff= expected_handoff_sha=
        while (($#)); do
          case "$1" in
            --item) items+=("$2"); shift 2;;
            --inspection-handoff) handoff=$2; shift 2;;
            --expected-inspection-handoff-sha256) expected_handoff_sha=$2; shift 2;;
            *) shift;;
          esac
        done
        if [[ "$mode" == classify-batch ]]; then
          printf 'expensive-common-inspection\n' >> "$EVENTS"
          [[ "${FAIL_CLASSIFY_PRODUCER:-0}" == 1 ]] && { printf 'smoke\tfresh\n'; exit 42; }
        else
          [[ -f "$handoff" && "$(sha256sum "$handoff" | awk '{print $1}')" == "$expected_handoff_sha" ]] || exit 37
        fi
        for item in "${items[@]}"; do
          IFS='|' read -r label output run_id kind index policy concurrency <<< "$item"
          printf 'state:%s:%s:%s:%s\n' "$mode" "$label" "$policy" "$concurrency" >> "$EVENTS"
          if [[ "$mode" == classify-batch ]]; then
            if [[ -e "$output" || -L "$output" ]]; then
              [[ -d "$output" && -f "$output/COMPLETE" && ! -e "${output}.inprogress" ]] || exit 33
              state=complete
            elif [[ -d "${output}.inprogress" && -f "${output}.inprogress/MATCH" ]]; then
              state=resume
            elif [[ ! -e "${output}.inprogress" && ! -L "${output}.inprogress" ]]; then state=fresh
            else exit 34; fi
            printf '%s\t%s\n' "$label" "$state"
          elif [[ "$mode" == validate-batch ]]; then
            [[ -d "$output" && -f "$output/COMPLETE" ]] || exit 35
            printf 'validated:%s\n' "$output" >> "$EVENTS"
          fi
        done
        if [[ "$mode" == classify-batch ]]; then
          [[ "${FAIL_HANDOFF_PUBLICATION:-0}" == 1 ]] && exit 43
          printf '{"format":"fake-bound-handoff-v1"}\n' > "$handoff"
        fi
        """,
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NODELIST": "fake-node",
        "NIMLOTH_ROOT": str(tmp_path / "nimloth"),
        "EXPECTED_NIMLOTH_COMMIT": nimloth_commit,
        "PYTHON_EXECUTABLE": str(python),
        "VAGEN_RUNTIME_ROOT": str(tmp_path / "vagen"),
        "MODEL_PATH": str(tmp_path / "model"),
        "PARTITION_MANIFEST": str(tmp_path / "partition.json"),
        "SOURCE_RUNTIME_CONTRACT": str(tmp_path / "contract.json"),
        "RUN_ROOT": str(run_root),
        "RUN_ID_PREFIX": "test-gate",
        "RUN_MODE": "fresh",
        "ENV_SERVICE_ENTRYPOINT": str(tools / "env-service"),
        "COLLECTOR_ENTRYPOINT": str(tools / "collector"),
        "SHARD_STATE_ENTRYPOINT": str(tools / "validator"),
        "EXPECTED_RECONSTRUCTION_HEAD": "h" * 40,
        "EXPECTED_RECONSTRUCTION_TREE": "t" * 40,
        "EXPECTED_RECONSTRUCTION_DIFF_SHA256": "d" * 64,
        "EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256": "c" * 64,
        "SMOKE_FORMAT_FAILURE_POLICY": "fail_shard",
        "SMOKE_COLLECTOR_CONCURRENCY": "1",
        "GATE_FORMAT_FAILURE_POLICY": "exclude_trajectory",
        "GATE_COLLECTOR_CONCURRENCY": "4",
        "GPU_MEMORY_UTILIZATION": "0.8",
        "ENGINE_SEED": "42",
        "EVENTS": str(events),
        "FAIL_SMOKE": "1" if fail_smoke else "0",
        "ENV_READY_POLL_SECONDS": "0.01",
        "ENV_READY_MAX_POLLS": "100",
        "PATH": f"{tools}:{os.environ['PATH']}",
    }
    return env, events


def test_static_resource_topology_and_gate_order() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=preempt",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=224",
        "#SBATCH --gres=gpu:8",
        "#SBATCH --mem=480G",
    ):
        assert directive in text
    assert 'ENV_GPUS=("${ALLOCATED_GPUS[@]:0:4}")' in text
    assert 'POLICY_GPUS=("${ALLOCATED_GPUS[@]:4:4}")' in text
    assert "--tensor-parallel-size 1" in text
    assert 'setsid bash "${ENV_SERVICE_ENTRYPOINT}"' in text
    assert '-x "${ENV_SERVICE_ENTRYPOINT}"' not in text
    assert "EXPECTED_NIMLOTH_COMMIT" in text
    assert "SMOKE_FORMAT_FAILURE_POLICY" in text
    assert "GATE_FORMAT_FAILURE_POLICY" in text
    assert 'export PYTHON_ENV="$(dirname "$(dirname "${PYTHON_EXECUTABLE}")")"' in text
    assert 'CLASSIFICATION_OUTPUT="${CONTROL_DIR}/classification.tsv"' in text
    assert 'set -o noclobber' in text
    assert 'done < "${CLASSIFICATION_OUTPUT}"' in text
    assert "done < <(" not in text
    assert text.index("verify-handoff") < text.index('setsid bash "${ENV_SERVICE_ENTRYPOINT}"')
    smoke_launch = text.index("setsid bash -c 'run_collector \"$@\"' _ smoke")
    assert smoke_launch < text.index("launch_gate_shards")
    assert smoke_launch < text.index('CHILD_PIDS+=("${SMOKE_PID}")')
    assert text.index('remove_owned_child "${SMOKE_PID}"') < text.index("launch_gate_shards")
    assert text.index('--item "${SMOKE_ITEM}" >> "${LOG_DIR}/smoke-validation.log"') < text.index("launch_gate_shards")


def test_environment_entrypoint_rejects_symlink_but_not_missing_execute_bit(
    tmp_path: Path,
) -> None:
    env, events = _fixture(tmp_path)
    original = Path(env["ENV_SERVICE_ENTRYPOINT"])
    symlink = original.with_name("env-service-link")
    symlink.symlink_to(original)
    env["ENV_SERVICE_ENTRYPOINT"] = str(symlink)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists()


def test_smoke_and_gate_collection_contracts_are_not_interchangeable(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    env["SMOKE_COLLECTOR_CONCURRENCY"] = "4"
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists()

    env, events = _fixture(tmp_path / "gate")
    env["GATE_FORMAT_FAILURE_POLICY"] = "fail_shard"
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists()


def test_env_entrypoint_preserves_caller_python_env_across_credentials(tmp_path: Path) -> None:
    caller = tmp_path / "caller-env"
    caller.mkdir()
    common_env = ROOT / "experiments" / "training" / "sft1" / "common_env.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n%s\\n" "$PYTHON_ENV" "$VIRTUAL_ENV" "$PATH"',
            "_",
            str(common_env),
        ],
        env={**os.environ, "REPO": str(tmp_path / "repo"), "PYTHON_ENV": str(caller)},
        text=True,
        capture_output=True,
        check=True,
    )
    python_env, virtual_env, path = result.stdout.splitlines()
    assert python_env == str(caller)
    assert virtual_env == str(caller)
    assert path.split(":", 1)[0] == str(caller / "bin")

    env_script = (
        ROOT / "experiments" / "training" / "sft1" / "env_external_4gpu.slurm"
    ).read_text(encoding="utf-8")
    assert env_script.count('source "${SCRIPTDIR}/common_env.sh"') == 1
    assert "/project/peilab/atst/flower/.env" not in env_script
    assert "/project/peilab/atst/.env" not in env_script


def test_classification_failures_propagate_before_environment_start(tmp_path: Path) -> None:
    for variable, expected_rc in (
        ("FAIL_CLASSIFY_PRODUCER", 42),
        ("FAIL_HANDOFF_PUBLICATION", 43),
    ):
        env, events = _fixture(tmp_path / variable.lower())
        env[variable] = "1"
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
        )
        assert result.returncode == expected_rc
        assert not events.exists() or "env:" not in events.read_text(encoding="utf-8")
        attempt = next((Path(env["RUN_ROOT"]) / "attempts").iterdir())
        classification = attempt / "control" / "classification.tsv"
        assert classification.is_file()
        assert classification.read_text(encoding="utf-8")


def test_smoke_failure_blocks_every_gate_shard(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path, fail_smoke=True)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    lines = events.read_text(encoding="utf-8").splitlines()
    first_env = next(i for i, line in enumerate(lines) if line.startswith("env:"))
    before_env = lines[:first_env]
    assert sum(line == "helper:classify-batch" for line in before_env) == 1
    assert [line.split(":")[2] for line in before_env if line.startswith("state:classify-batch:")] == [
        "smoke", "shard-0", "shard-1", "shard-2", "shard-3"
    ]
    assert "state:classify-batch:smoke:fail_shard:1" in before_env
    assert all(
        f"state:classify-batch:shard-{i}:exclude_trajectory:4" in before_env
        for i in range(4)
    )
    assert any(line.startswith("collector:0:4:") for line in lines)
    assert not any(line.startswith("collector:shard-") for line in lines)


def test_fresh_gate_launches_four_disjoint_collectors(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr + result.stdout
    collectors = [
        line
        for line in events.read_text(encoding="utf-8").splitlines()
        if line.startswith("collector:")
    ]
    assert len(collectors) == 5  # smoke, then four gate collectors
    assert events.read_text(encoding="utf-8").splitlines().count(
        "expensive-common-inspection"
    ) == 1
    gate = [line for line in collectors if "collector:shard-" in line]
    assert {line.split(":")[2] for line in gate} == {"4", "5", "6", "7"}
    assert {line.split(":")[1] for line in gate} == {
        "shard-0",
        "shard-1",
        "shard-2",
        "shard-3",
    }


def test_complete_and_resume_states_are_fail_closed(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    first = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0
    events.write_text("", encoding="utf-8")
    env["RUN_MODE"] = "resume"
    env["FAIL_SMOKE"] = "0"
    env["SLURM_JOB_ID"] = "124"
    run_root = Path(env["RUN_ROOT"])
    for stale in (
        run_root / "gate" / "shard-002",
        run_root / "gate" / "shard-003",
        run_root / "gate" / "shard-004",
    ):
        for child in stale.iterdir():
            child.unlink()
        stale.rmdir()
    resume = run_root / "gate" / "shard-002.inprogress"
    resume.mkdir(parents=True)
    (resume / "MATCH").touch()

    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr + result.stdout
    lines = events.read_text(encoding="utf-8").splitlines()
    collectors = [line for line in lines if line.startswith("collector:")]
    assert len(collectors) == 3  # one complete shard is strictly skipped
    assert {line.split(":")[1] for line in collectors} == {
        "shard-1",
        "shard-2",
        "shard-3",
    }
    assert {line.split(":")[2] for line in collectors} == {"5", "6", "7"}
    assert any(
        line.startswith("collector:shard-1:5:http://127.0.0.1:8401:1:")
        for line in collectors
    )
    env_line = next(line for line in lines if line.startswith("env:"))
    assert env_line == (
        f"env:0,1,2,3:{env['VAGEN_RUNTIME_ROOT']}:"
        f"{Path(env['PYTHON_EXECUTABLE']).parent.parent}"
    )
    assert all(f"/gate/shard-{i:03d}" in "\n".join(lines) for i in range(1, 5))


def test_child_failure_propagates_and_cancels_siblings(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    env["FAIL_GATE_SHARD"] = "1"
    env["SLOW_GATE"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 23
    lines = events.read_text(encoding="utf-8").splitlines()
    attempt = next((Path(env["RUN_ROOT"]) / "attempts").iterdir())
    pid_log = (attempt / "control" / "children.tsv").read_text(encoding="utf-8")
    assert sum(line.startswith("shard-") for line in pid_log.splitlines()) == 4
    assert any(line.startswith("cancelled:shard-") for line in lines)
    assert "env-stopped" in lines


def test_invalid_inprogress_blocks_all_runtime_activation(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path, fail_smoke=True)
    first = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert first.returncode != 0
    events.write_text("", encoding="utf-8")
    run_root = Path(env["RUN_ROOT"])
    env["RUN_MODE"] = "resume"
    env["FAIL_SMOKE"] = "0"
    env["SLURM_JOB_ID"] = "124"
    smoke = run_root / "smoke" / "source-index-00000"
    smoke.mkdir(parents=True)
    (smoke / "COMPLETE").touch()
    (run_root / "gate" / "shard-001.inprogress").mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    lines = events.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("env:") for line in lines)
    assert not any("collector:shard-" in line for line in lines)


def test_fresh_and_resume_run_root_ownership_is_hash_bound(tmp_path: Path) -> None:
    env, _ = _fixture(tmp_path)
    run_root = Path(env["RUN_ROOT"])
    first = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr + first.stdout
    identity = (run_root / "RUN_IDENTITY").read_bytes()

    duplicate_fresh = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert duplicate_fresh.returncode != 0
    assert (run_root / "RUN_IDENTITY").read_bytes() == identity

    resumed = dict(env, RUN_MODE="resume", SLURM_JOB_ID="124")
    second = subprocess.run(
        ["bash", str(SCRIPT)], env=resumed, text=True, capture_output=True, check=False
    )
    assert second.returncode == 0, second.stderr + second.stdout
    controls = sorted((run_root / "attempts").iterdir())
    assert len(controls) == 2
    assert controls[0].name != controls[1].name
    assert all((attempt / "status.tsv").is_file() for attempt in controls)


def test_nimloth_commit_or_cleanliness_drift_fails_before_environment_start(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    env["EXPECTED_NIMLOTH_COMMIT"] = "0" * 40
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists() or "env:" not in events.read_text(encoding="utf-8")

    env, events = _fixture(tmp_path / "dirty")
    (Path(env["NIMLOTH_ROOT"]) / "tracked").write_text("dirty\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists() or "env:" not in events.read_text(encoding="utf-8")


def test_batch_helper_inspects_common_inputs_once_for_all_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    state = _load_state_module()

    calls = {"common": 0, "partition": 0, "outputs": []}

    class Collector:
        def inspect_output_state(self, specs, *, output_dir):
            calls["outputs"].append((specs, output_dir))
            return "fresh"

    collect = types.ModuleType("experiments.training.sft1.vagen_step60_collect")
    collect.EpisodeSpec = lambda **kwargs: types.SimpleNamespace(**kwargs)
    collect.prepare_collection_inspection_context = lambda **kwargs: (
        calls.__setitem__("common", calls["common"] + 1) or {}
    )
    spec = lambda index: types.SimpleNamespace(
        source_index=index,
        eval_set="base",
        seed=index,
        dataset_split="train",
        source_key=f"base:{index}",
    )
    collect.batch1_smoke_spec_from_manifest = lambda manifest, source_index: spec(source_index)
    collect.batch1_shard_specs_from_manifest = (
        lambda manifest, shard_index, shard_size: [spec(shard_index)]
    )
    collect.build_inspected_collector = lambda context, **kwargs: Collector()
    data = types.ModuleType("experiments.training.sft1.vagen_step60_data")
    data.load_published_partition_manifest = lambda path: (
        calls.__setitem__("partition", calls["partition"] + 1) or {"rows": []}
    )
    monkeypatch.setitem(sys.modules, collect.__name__, collect)
    monkeypatch.setitem(sys.modules, data.__name__, data)
    common = {
        "command": "classify-batch",
        "partition_manifest": tmp_path / "partition.json",
        "shard_size": 100,
        "source_runtime_root": tmp_path / "vagen",
        "source_runtime_contract": tmp_path / "contract.json",
        "model_path": tmp_path / "model",
        "expected_runtime_contract_payload_sha256": "c" * 64,
        "expected_reconstruction_head": "h" * 40,
        "expected_reconstruction_tree": "t" * 40,
        "expected_reconstruction_diff_sha256": "d" * 64,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "engine_seed": 42,
        "inspection_handoff": tmp_path / "handoff.json",
        "expected_inspection_handoff_sha256": None,
    }
    common["partition_manifest"].write_text("partition\n", encoding="utf-8")
    common["source_runtime_contract"].write_text("contract\n", encoding="utf-8")
    common["item"] = [
        f"smoke|{tmp_path / 'smoke'}|smoke|source-index|0|fail_shard|1",
        *[
            f"shard-{i}|{tmp_path / f'shard-{i}'}|gate-{i}|shard-index|{i}|exclude_trajectory|4"
            for i in range(4)
        ],
    ]
    state._run_batch(argparse.Namespace(**common))
    assert calls["common"] == 1
    assert calls["partition"] == 1
    assert len(calls["outputs"]) == 5


def test_ordinary_failure_is_non_resumable_and_finalize_error_preserves_exit(
    tmp_path: Path,
) -> None:
    env, _ = _fixture(tmp_path, fail_smoke=True)
    failed = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert failed.returncode == 19
    marker = Path(env["RUN_ROOT"]) / "NON_RESUMABLE.json"
    assert marker.is_file()

    resumed = dict(env, RUN_MODE="resume", FAIL_SMOKE="0", SLURM_JOB_ID="124")
    rejected = subprocess.run(
        ["bash", str(SCRIPT)], env=resumed, text=True, capture_output=True, check=False
    )
    assert rejected.returncode != 0

    env, _ = _fixture(tmp_path / "finalize-error", fail_smoke=True)
    env["FAIL_FINALIZE"] = "1"
    failed = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert failed.returncode == 19


def test_cleanup_terminates_group_after_failed_leader_exits(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    env.update(FAIL_GATE_SHARD="1", SLOW_GATE="1", SPAWN_ORPHAN_ON_FAIL="1")
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 23
    assert any(
        line.startswith("orphan-stopped:")
        for line in events.read_text(encoding="utf-8").splitlines()
    )


def test_python_run_root_initialization_fsyncs_identity_layout_and_parent(
    tmp_path: Path, monkeypatch
) -> None:
    state = _load_state_module()

    fsynced: list[Path] = []
    real_fsync = state.os.fsync

    def recording_fsync(fd: int) -> None:
        fsynced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr(state.os, "fsync", recording_fsync)
    run_root = tmp_path / "run"
    state.initialize_run_root(run_root, run_mode="fresh", identity=b"bound\n")

    assert (run_root / "RUN_IDENTITY").read_bytes() == b"bound\n"
    assert all((run_root / name).is_dir() for name in ("attempts", "smoke", "gate"))
    assert run_root / "RUN_IDENTITY" in fsynced
    assert run_root in fsynced
    assert tmp_path in fsynced
    with pytest.raises(FileExistsError):
        state.initialize_run_root(run_root, run_mode="fresh", identity=b"bound\n")


def test_term_while_smoke_is_running_cancels_smoke_before_any_gate_launch(
    tmp_path: Path,
) -> None:
    env, events = _fixture(tmp_path)
    env["SLOW_SMOKE"] = "1"
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if events.exists() and "collector:0:" in events.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("smoke collector did not start")
    process.terminate()
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 143, stderr + stdout
    lines = events.read_text(encoding="utf-8").splitlines()
    assert "cancelled:0" in lines
    assert not any(line.startswith("collector:shard-") for line in lines)
    assert "env-stopped" in lines
    run_root = Path(env["RUN_ROOT"])
    assert not (run_root / "NON_RESUMABLE.json").exists()
    attempt = next((run_root / "attempts").iterdir())
    pid_lines = (attempt / "control" / "children.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert sum(line.startswith("smoke\t") for line in pid_lines) == 1
    assert not any(line.startswith("shard-") for line in pid_lines)
    assert any(
        line.split("\t")[0] == "orchestrator" and line.endswith("\t143\tTERM")
        for line in (attempt / "status.tsv").read_text(encoding="utf-8").splitlines()
    )


def test_orchestrator_term_is_recorded_and_keeps_run_resumable(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    env["SLOW_GATE"] = "1"
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if events.exists() and "collector:shard-" in events.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        process.kill()
        raise AssertionError("gate collectors did not start")
    process.terminate()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 143, stderr + stdout
    run_root = Path(env["RUN_ROOT"])
    assert not (run_root / "NON_RESUMABLE.json").exists()
    attempt = next((run_root / "attempts").iterdir())
    assert any(
        line.split("\t")[0] == "orchestrator" and line.endswith("\t143\tTERM")
        for line in (attempt / "status.tsv").read_text(encoding="utf-8").splitlines()
    )


def test_child_rc143_is_non_resumable_without_orchestrator_signal(tmp_path: Path) -> None:
    env, _ = _fixture(tmp_path)
    env.update(FAIL_GATE_SHARD="1", FAIL_GATE_RC="143")
    collector = Path(env["COLLECTOR_ENTRYPOINT"])
    collector.write_text(
        collector.read_text(encoding="utf-8").replace("exit 23", 'exit "${FAIL_GATE_RC:-23}"'),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 143
    marker = Path(env["RUN_ROOT"]) / "NON_RESUMABLE.json"
    assert marker.is_file()
    assert '"orchestrator_signal":""' in marker.read_text(encoding="utf-8")


def test_python_non_resumable_marker_is_durable_but_interruption_stays_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    state = _load_state_module()
    run_root = tmp_path / "run"
    state.initialize_run_root(run_root, run_mode="fresh", identity=b"bound\n")
    fsynced: list[Path] = []
    real_fsync = state.os.fsync

    def recording_fsync(fd: int) -> None:
        fsynced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr(state.os, "fsync", recording_fsync)
    state.mark_run_non_resumable(run_root, exit_code=7, attempt_id="attempt-1")
    marker = run_root / "NON_RESUMABLE.json"
    assert marker.is_file()
    assert any(path.name.startswith(".NON_RESUMABLE.json.tmp-") for path in fsynced)
    assert run_root in fsynced
    with pytest.raises(ValueError, match="non-resumable"):
        state.initialize_run_root(run_root, run_mode="resume", identity=b"bound\n")

    interrupted = tmp_path / "interrupted"
    state.initialize_run_root(interrupted, run_mode="fresh", identity=b"bound\n")
    state.mark_run_non_resumable(
        interrupted,
        exit_code=143,
        attempt_id="attempt-2",
        orchestrator_signal="TERM",
    )
    assert not (interrupted / "NON_RESUMABLE.json").exists()
    state.initialize_run_root(interrupted, run_mode="resume", identity=b"bound\n")

    ordinary_143 = tmp_path / "ordinary-143"
    state.initialize_run_root(ordinary_143, run_mode="fresh", identity=b"bound\n")
    state.mark_run_non_resumable(
        ordinary_143,
        exit_code=143,
        attempt_id="attempt-3",
        orchestrator_signal=None,
    )
    assert (ordinary_143 / "NON_RESUMABLE.json").is_file()


def test_resume_identity_drift_fails_before_environment_start(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path)
    first = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr + first.stdout
    events.write_text("", encoding="utf-8")
    resumed = dict(env, RUN_MODE="resume", SLURM_JOB_ID="124", ENGINE_SEED="43")
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=resumed, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not events.exists() or "env:" not in events.read_text(encoding="utf-8")
