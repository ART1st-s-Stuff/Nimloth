"""Real-environment evaluation, separate from offline SFT validation."""
from .config import EvaluationConfig


def eval_direct(config: EvaluationConfig) -> int:
    from .cli import eval_direct as run
    return run(config)


def eval_wm(config: EvaluationConfig) -> int:
    from .cli import eval_wm as run
    return run(config)


__all__ = ["EvaluationConfig", "eval_direct", "eval_wm"]
