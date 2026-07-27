"""公共 module 状态工具测试。"""

import torch

from nimloth.util.module import evaluating, move_to_device


def test_move_to_device_accepts_device_module_and_tensor_targets() -> None:
    source = torch.ones(2, dtype=torch.float32)
    module = torch.nn.Linear(2, 2, dtype=torch.float64)
    reference = torch.zeros(2, dtype=torch.float64)

    assert move_to_device(source, torch.device("cpu")).dtype == torch.float32
    assert move_to_device(source, module).dtype == torch.float64
    assert move_to_device(source, reference).dtype == torch.float64


def test_move_to_device_preserves_integer_dtype() -> None:
    source = torch.ones(2, dtype=torch.long)
    reference = torch.zeros(2, dtype=torch.float64)

    assert move_to_device(source, reference).dtype == torch.long


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
