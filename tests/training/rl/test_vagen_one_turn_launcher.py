from pathlib import Path


def test_one_turn_launcher_validates_capture_v2_generation_identity() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "experiments/training/rl/run_vagen_one_turn_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "state['schema']=='nimloth_policy_state_v2'" in source
    assert "state['generation_id']!=state['request_id']" in source
    assert "'generation_id':state['generation_id']" in source
    assert "nimloth_policy_state_v1" not in source
