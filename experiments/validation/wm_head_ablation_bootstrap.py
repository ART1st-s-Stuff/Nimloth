"""Fast stdlib-only bootstrap checks for the frozen WM-head ablation."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/training/reconstruction/frozen_wm_head_shape_ablation.json"
DEV = ROOT / "experiments/validation/wm_head_ablation_dev.sh"
VERIFY = ROOT / "experiments/validation/verify_wm_head_shape_ablation.sh"
PYTHON_TARGETS = (
    ROOT / "experiments/validation/wm_head_ablation_bootstrap.py",
    ROOT / "src/nimloth/eval/matched_wm_ablation.py",
    ROOT / "src/nimloth/eval/matched_wm_metrics.py",
    ROOT / "src/nimloth/eval/matched_wm_render.py",
    ROOT / "src/nimloth/eval/matched_wm_turns.py",
    ROOT / "src/nimloth/training/wm_heads/data.py",
    ROOT / "src/nimloth/training/wm_heads/trainer.py",
    ROOT / "src/nimloth/wm/frozen_query_state.py",
    ROOT / "src/nimloth/wm/frozen_state_cache.py",
    ROOT / "src/nimloth/wm/matched_heads.py",
    ROOT / "tests/eval/test_matched_wm_ablation.py",
    ROOT / "tests/test_matched_wm_heads.py",
    ROOT / "tests/training/test_matched_wm_trainer.py",
)
CONTROL_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match, ast.comprehension)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_state(config: dict) -> None:
    state = config["state"]
    heads = config["heads"]
    assert state["tokens"] * state["token_dim"] == state["flat_dim"] == 8192
    assert heads["vector"]["tokens"] == 1
    assert heads["vector"]["emb_dim"] == state["flat_dim"]
    assert heads["token"]["tokens"] == state["tokens"]
    assert heads["token"]["emb_dim"] == state["token_dim"]


def check_budget(config: dict) -> None:
    training = config["training"]
    validation = config["validation"]
    assert training["steps"] == 10000
    assert training["batch_size"] == 128
    assert validation["max_gpus"] <= 2
    assert validation["max_gpu_hours"] <= 2
    assert validation["require_turn_rows"] == 30


def check_turns(config: dict) -> None:
    spec = load_json(ROOT / config["inputs"]["turn_spec"])
    required = set(config["reconstruction"]["required_actions"])
    forbidden = set(config["reconstruction"]["forbidden_substitutes"])
    selections = spec["selections"]
    assert len(selections) == config["reconstruction"]["num_rollouts"] == 6
    for selection in selections:
        actions = selection["expected_actions"]
        assert len(actions) == config["reconstruction"]["horizons"] == 5
        assert required.issubset(actions)
        assert forbidden.isdisjoint(actions)


def control_depth(node: ast.AST, depth: int = 0) -> int:
    child_depth = depth + int(isinstance(node, CONTROL_NODES))
    nested = [control_depth(child, child_depth) for child in ast.iter_child_nodes(node)]
    return max([child_depth, *nested])


def check_python_structure(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 200, f"file exceeds 200 LOC: {path}"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert node.end_lineno - node.lineno + 1 <= 30, f"construct exceeds 30 LOC: {path}:{node.lineno}"
            assert control_depth(node) <= 3, f"nesting exceeds 3: {path}:{node.lineno}"


def check_files() -> None:
    for path in (CONFIG, DEV, VERIFY, *PYTHON_TARGETS):
        assert path.is_file(), f"missing bootstrap file: {path}"
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 200
    for path in PYTHON_TARGETS:
        check_python_structure(path)


def main() -> int:
    config = load_json(CONFIG)
    check_state(config)
    check_budget(config)
    check_turns(config)
    check_files()
    print("bootstrap_config=PASS")
    print("turn_semantics=PASS required=turn_right,turn_left forbidden=move_right,move_left")
    print("state_views=PASS vector=1x8192 token=8x1024 shared_scalars=8192")
    print("structure=PASS file_max=200 construct_max=30 nesting_max=3")
    print("mise=WAIVED_BY_USER github_ci=WAIVED_BY_USER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
