"""公共 module 状态工具测试。"""

import torch

from nimloth.util.module import evaluating


def test_evaluating_restores_original_training_mode() -> None:
    module = torch.nn.Dropout(p=0.5)
    module.train()

    with evaluating(module):
        assert module.training is False

    assert module.training is True


def test_evaluating_keeps_existing_eval_mode() -> None:
    module = torch.nn.Linear(2, 2)
    module.eval()

    with evaluating(module):
        assert module.training is False

    assert module.training is False
