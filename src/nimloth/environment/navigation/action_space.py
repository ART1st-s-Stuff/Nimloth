"""Navigation environment 的动作空间定义。"""

from nimloth.environment.common.action_space import ActionSpec, DiscreteActionSpace


NAVIGATION_ACTION_SPACE = DiscreteActionSpace(
    identifier="navigation",
    version=1,
    actions=(
        ActionSpec("moveahead", aliases=("move_forward",)),
        ActionSpec("moveback", aliases=("move_backward",)),
        ActionSpec("moveright", aliases=("move_right",)),
        ActionSpec("moveleft", aliases=("move_left",)),
        ActionSpec("rotateright", aliases=("turn_right",)),
        ActionSpec("rotateleft", aliases=("turn_left",)),
        ActionSpec("lookup", aliases=("look_up",)),
        ActionSpec("lookdown", aliases=("look_down",)),
    ),
)

# 数量属于 navigation 动作空间，不属于 Agent、rollout 或 WM。
NUM_NAVIGATION_ACTIONS = len(NAVIGATION_ACTION_SPACE)
