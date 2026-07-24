"""训练和评估共用的 PyTorch module 状态工具。"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch


@contextmanager
def evaluating(module: torch.nn.Module) -> Iterator[None]:
    """临时关闭 dropout 等训练行为，退出时恢复原始模式。"""

    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)
