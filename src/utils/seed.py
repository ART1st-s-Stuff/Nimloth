"""随机种子工具。"""

from __future__ import annotations

import random

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - 允许无 torch 环境执行非训练流程
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

