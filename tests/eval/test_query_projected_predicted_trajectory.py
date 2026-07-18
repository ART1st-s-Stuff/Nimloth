import torch

from nimloth.eval.query_projected_predicted_trajectory import prepare_rows


class _Predictor:
    def rollout_states(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return initial[:, None, :] + torch.arange(1, 6, device=initial.device)[None, :, None]


def _record(record_id: str, actions: list[int], shape: tuple[int, ...]) -> dict[int, dict]:
    return {
        step: {
            "id": f"{record_id}-{step}",
            "record_id": record_id,
            "step_index": step,
            "action_index": actions[min(step, 4)],
            "current_image_path": f"/{record_id}/{step}.png",
            "state_emb": torch.full(shape, float(step)),
        }
        for step in range(6)
    }


def test_prepare_rows_aligns_actual_and_autoregressive_predicted_states() -> None:
    actions = [0, 4, 0, 5, 0]
    selection = [{"run_index": 0, "record_id": "r", "expected_actions": actions}]
    rows, states = prepare_rows(
        selection,
        {"r": _record("r", actions, (8, 2048))},
        {"r": _record("r", actions, (8192,))},
        {"r": _record("r", actions, (16, 512))},
        _Predictor(),
        torch.device("cpu"),
    )
    assert len(rows) == 5
    assert rows[1]["action_name"] == "turn_right"
    assert rows[3]["action_name"] == "turn_left"
    assert states["query"].shape == (5, 8, 2048)
    assert states["projected"].shape == states["predicted"].shape == (5, 8192)
    assert states["positive"].shape == (5, 16, 512)
    torch.testing.assert_close(states["predicted"][:, 0], torch.arange(1, 6).float())
