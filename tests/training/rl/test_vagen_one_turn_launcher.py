import re
from pathlib import Path


def _runner_source() -> str:
    return (
        Path(__file__).resolve().parents[3]
        / "experiments/training/rl/run_vagen_one_turn_smoke.sh"
    ).read_text(encoding="utf-8")


def test_one_turn_launcher_validates_capture_v2_generation_identity() -> None:
    source = _runner_source()
    assert "state['schema']=='nimloth_policy_state_v2'" in source
    assert "state['generation_id']!=state['request_id']" in source
    assert "'generation_id':state['generation_id']" in source
    assert "nimloth_policy_state_v1" not in source


def test_one_turn_launcher_embedded_python_compiles() -> None:
    programs = re.findall(
        r"<<'PY'[^\n]*\n(.*?)\nPY",
        _runner_source(),
        flags=re.DOTALL,
    )
    assert len(programs) >= 4
    for index, program in enumerate(programs):
        compile(program, f"run_vagen_one_turn_smoke.sh:heredoc-{index}", "exec")


def test_one_turn_launcher_has_strict_guided_tp8_contract() -> None:
    source = _runner_source()
    launch = (
        Path(__file__).resolve().parents[3]
        / "experiments/training/rl/launch_vagen_one_turn_smoke_on_hold.sh"
    ).read_text(encoding="utf-8")
    assert "guided_tp8_gate" in source
    assert '[[ "${EXPERIMENT_ID}" == "163" ]]' in source
    assert '"1:1:1:float32:42:776"' in source
    assert "--guided" in source
    assert 'SMOKE_COMMAND+=("${SMOKE_EXTRA_ARGS[@]}")' in source
    assert '"${SMOKE_COMMAND[@]}"' in source
    assert '--critic-qwen-hidden-dim 2048' in source
    assert '--critic-state-dim 1024' in source
    assert '--joint-snapshot-source-step "${JOINT_SNAPSHOT_SOURCE_STEP}"' in source
    assert "vagen_decision_ledger_v2_frozen_q_guided" in source
    assert "nimloth_frozen_q_scoring_v1" in source
    assert "vagen_frozen_q_guided_action_draw_v2" in source
    assert "vagen_guided_action_execution_v3" in source
    assert "key['run_seed']==42" in source
    assert "pin['snapshot_source_step']==776" in source
    assert ': "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"' in source
    assert "EXPECTED_VAGEN_COMMIT" in launch
    assert "JOINT_SNAPSHOT_SOURCE_STEP" in launch
