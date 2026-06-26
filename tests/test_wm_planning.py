from __future__ import annotations

import torch

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.planning import (
    Planner,
    PlannerConfig,
    beam_search_action,
    greedy_action,
)
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.value_head import ValueHead


def _make_predictor_and_head(emb_dim: int = 64) -> tuple[LatentWMPredictor, ValueHead]:
    cfg = LeWMConfig(emb_dim=emb_dim, history_size=4)
    predictor = LatentWMPredictor.create(cfg)
    value_head = ValueHead(emb_dim=emb_dim)
    predictor.eval()
    value_head.eval()
    return predictor, value_head


# --- greedy ----------------------------------------------------------------


def test_greedy_single_state() -> None:
    _, value_head = _make_predictor_and_head()
    state = torch.randn(64)
    action = greedy_action(state, value_head)
    assert action.ndim == 0
    assert 0 <= action.item() < 8


def test_greedy_batch() -> None:
    _, value_head = _make_predictor_and_head()
    states = torch.randn(4, 64)
    actions = greedy_action(states, value_head)
    assert actions.shape == (4,)
    assert all(0 <= a.item() < 8 for a in actions)


def test_greedy_deterministic() -> None:
    _, value_head = _make_predictor_and_head()
    value_head.eval()
    state = torch.randn(4, 64)
    a1 = greedy_action(state, value_head)
    a2 = greedy_action(state, value_head)
    assert torch.equal(a1, a2)


# --- beam search -----------------------------------------------------------


def test_beam_search_single_state() -> None:
    predictor, value_head = _make_predictor_and_head()
    state = torch.randn(64)
    actions, scores = beam_search_action(state, predictor, value_head,
                                         beam_width=2, rollout_depth=3)
    assert actions.shape == (1,)
    assert scores.shape == (1,)
    assert 0 <= actions[0].item() < 8


def test_beam_search_batch() -> None:
    predictor, value_head = _make_predictor_and_head()
    states = torch.randn(3, 64)
    actions, scores = beam_search_action(states, predictor, value_head,
                                         beam_width=2, rollout_depth=3)
    assert actions.shape == (3,)
    assert scores.shape == (3,)


def test_beam_search_deterministic() -> None:
    predictor, value_head = _make_predictor_and_head()
    predictor.eval()
    value_head.eval()
    state = torch.randn(64)
    a1, s1 = beam_search_action(state, predictor, value_head, beam_width=2, rollout_depth=2)
    a2, s2 = beam_search_action(state, predictor, value_head, beam_width=2, rollout_depth=2)
    assert torch.equal(a1, a2)
    assert torch.allclose(s1, s2)


# --- Planner class ---------------------------------------------------------


def test_planner_greedy() -> None:
    _, value_head = _make_predictor_and_head()
    planner = Planner(PlannerConfig(algorithm="greedy"), value_head=value_head)
    state = torch.randn(64)
    action = planner.select_action(state)
    assert action.ndim == 0
    assert 0 <= action.item() < 8


def test_planner_beam_search() -> None:
    predictor, value_head = _make_predictor_and_head()
    planner = Planner(
        PlannerConfig(algorithm="beam_search", beam_width=2, rollout_depth=3),
        predictor=predictor, value_head=value_head,
    )
    state = torch.randn(64)
    action = planner.select_action(state)
    assert action.ndim == 0
    assert 0 <= action.item() < 8


def test_planner_config_from_dict() -> None:
    cfg = PlannerConfig.from_dict({"algorithm": "beam_search", "beam_width": 3})
    assert cfg.algorithm == "beam_search"
    assert cfg.beam_width == 3
    # defaults preserved
    assert cfg.rollout_depth == 4


def test_planner_from_dict_creates_greedy() -> None:
    _, value_head = _make_predictor_and_head()
    planner = Planner({"algorithm": "greedy"}, value_head=value_head)
    action = planner.select_action(torch.randn(64))
    assert 0 <= action.item() < 8


def test_planner_requires_value_head() -> None:
    try:
        Planner(PlannerConfig(algorithm="greedy"))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_planner_requires_predictor_for_beam() -> None:
    _, value_head = _make_predictor_and_head()
    try:
        Planner(PlannerConfig(algorithm="beam_search"), value_head=value_head)
        raised = False
    except ValueError:
        raised = True
    assert raised
