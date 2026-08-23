import numpy as np
import pytest

from nimloth.eval.id189_wm_decoder_diagnostic import _noise_seed, state_diagnostic


def test_state_diagnostic_separates_wm_from_copy_and_ranks_action() -> None:
    current = np.zeros((16, 1024), dtype=np.float32)
    actual = np.ones((16, 1024), dtype=np.float32)
    executed = np.full((16, 1024), 0.75, dtype=np.float32)
    wrong = np.full((16, 1024), 0.1, dtype=np.float32)

    result = state_diagnostic(
        current=current,
        actual_next=actual,
        predicted_next=executed,
        depth1_states={0: wrong, 2: executed},
        executed_action=2,
    )

    assert result["state_copy_rmse"] == pytest.approx(1.0)
    assert result["state_predicted_rmse"] == pytest.approx(0.25)
    assert result["state_predicted_over_copy"] == pytest.approx(0.25)
    assert result["state_predicted_better_than_copy"] is True
    assert result["state_executed_action_rank"] == 1
    assert result["state_executed_action_top1"] is True
    assert result["state_depth1_action_count"] == 2


def test_noise_seed_is_identity_and_repeat_stable() -> None:
    first = _noise_seed(20260823, "sha256:" + "a" * 64, 0)
    assert first == _noise_seed(20260823, "sha256:" + "a" * 64, 0)
    assert first != _noise_seed(20260823, "sha256:" + "a" * 64, 1)
    assert first != _noise_seed(20260823, "sha256:" + "b" * 64, 0)
    assert 0 <= first < 2**63 - 1


def test_state_diagnostic_requires_executed_depth1_state() -> None:
    state = np.zeros((16, 1024), dtype=np.float32)
    with pytest.raises(ValueError, match="executed action"):
        state_diagnostic(
            current=state,
            actual_next=state,
            predicted_next=state,
            depth1_states={1: state},
            executed_action=0,
        )
