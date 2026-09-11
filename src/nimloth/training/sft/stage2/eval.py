"""stage2 direct environment evaluation; offline loss evaluation is separate."""
from nimloth.training.sft.evaluation.config import EvaluationConfig


def evaluate(config: EvaluationConfig) -> int:
    if config.stage != 'stage2' or config.mode != 'direct':
        raise ValueError('stage2 eval requires its explicit direct stage')
    from nimloth.training.sft.evaluation.early import run_early_evaluation
    return run_early_evaluation(config)
