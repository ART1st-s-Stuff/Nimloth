import json
from pathlib import Path

import pytest

from nimloth.eval.rcdm_cache_rollout import validate_turn_window


def test_validate_turn_window_requires_exact_actions_and_both_turns() -> None:
    validate_turn_window([0, 4, 0, 5, 0], [0, 4, 0, 5, 0])
    with pytest.raises(ValueError, match="action mismatch"):
        validate_turn_window([0, 4, 0, 5, 0], [0, 5, 0, 4, 0])
    with pytest.raises(ValueError, match="both turn_right"):
        validate_turn_window([0, 4, 0, 4, 0], [0, 4, 0, 4, 0])


def test_committed_turn_selection_uses_five_actions_from_rollout_start() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/eval/reconstruction/rcdm_rollout5_turns_val.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["selections"]) == 6
    for selection in data["selections"]:
        actions = selection["expected_actions"]
        assert selection["start_step"] == 0
        validate_turn_window(actions, actions)
