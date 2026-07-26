"""SFT2 与 RL 共享且语义一致的训练目标。"""

from nimloth.training.common.value import ActionValueLoss, action_value_loss

__all__ = ["ActionValueLoss", "action_value_loss"]
