"""stage1 direct environment evaluation; offline loss evaluation is separate."""
from nimloth.training.sft.evaluation.config import EvaluationConfig


def evaluate(config: EvaluationConfig) -> int:
    if config.stage != 'stage1' or config.mode != 'direct':
        raise ValueError('stage1 eval requires its explicit direct stage')
    from nimloth.training.sft.evaluation.early import run_early_evaluation
    return run_early_evaluation(config)
