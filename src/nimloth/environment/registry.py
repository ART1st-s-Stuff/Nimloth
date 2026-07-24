"""环境动作空间的显式注册与版本解析。"""

from __future__ import annotations

from nimloth.environment.common.action_space import DiscreteActionSpace
from nimloth.environment.navigation.action_space import NAVIGATION_ACTION_SPACE


_ACTION_SPACES = {
    (NAVIGATION_ACTION_SPACE.identifier, NAVIGATION_ACTION_SPACE.version): (
        NAVIGATION_ACTION_SPACE
    ),
}


def get_action_space(identifier: str, version: int) -> DiscreteActionSpace:
    """按持久化标识取得动作空间，未知版本直接报错。"""

    try:
        return _ACTION_SPACES[(identifier, version)]
    except KeyError as error:
        raise ValueError(
            f"unknown action space {identifier!r} version {version}"
        ) from error
