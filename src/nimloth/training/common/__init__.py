"""SFT2 与 RL 共享且语义一致的训练目标。"""

from nimloth.training.common.value import ActionValueLoss, action_value_loss
from nimloth.training.common.world_model import WorldModelLoss, world_model_loss

__all__ = [
    "ActionValueLoss",
    "WorldModelLoss",
    "action_value_loss",
    "world_model_loss",
]
