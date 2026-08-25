from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/training/rl/render_id185_rollout_visualization.py"
SPEC = importlib.util.spec_from_file_location("id185_visualization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_renderer_builds_interactive_step_view(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    payload = {
        "schema": "vagen_single_rollout_visualization_audit_v1",
        "rollout_sample_id": "sha256:test",
        "task": "navigate to the Toaster",
        "turn_count": 1,
        "success": False,
        "turns": [
            {
                "executed_action": "MoveAhead",
                "prior_action": "RotateLeft",
                "observation_image": "step_00_observation.png",
                "cot": "<think>actual reasoning</think>",
                "current_state_value": 0.2,
                "executed_action_direct_q": 0.1,
                "executed_action_predicted_value": 0.3,
                "env_turn_reward": 0.01,
                "planner_latency_seconds": 0.4,
                "action_ranking": [
                    {
                        "action": "MoveAhead",
                        "prior_probability": 0.1,
                        "guided_probability": 0.8,
                        "direct_q": 0.1,
                        "predicted_root_value": 0.3,
                        "root_visits": 80,
                        "is_executed_action": True,
                        "is_prior_action": False,
                    }
                ],
                "predicted_action_sequences": [
                    {
                        "actions": ["MoveAhead"] * 4,
                        "predicted_value": 0.4,
                        "visits": 80,
                    }
                ],
            }
        ],
    }
    (audit / "rollout_audit.json").write_text(json.dumps(payload))
    output = audit / "index.html"
    MODULE.render(audit, output)
    text = output.read_text()
    assert "ID185 K4 Scheme-B rollout" in text
    assert 'id="task"' in text
    match = re.search(r"JSON\.parse\(atob\('([^']+)'\)\)", text)
    assert match is not None
    embedded = json.loads(base64.b64decode(match.group(1)))
    assert embedded["task"] == "navigate to the Toaster"
    assert "Current state value" in text
    assert "Predicted action lists" in text
    assert "__AUDIT_B64__" not in text
