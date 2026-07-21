"""由 environment 提供的离散动作空间。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    """一个稳定动作 key 及环境后端可能使用的别名。"""

    key: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("action key must be non-empty")


@dataclass(frozen=True)
class DiscreteActionSpace:
    """带版本的有序动作空间；index 由 actions 的顺序决定。"""

    identifier: str
    version: int
    actions: tuple[ActionSpec, ...]

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("action-space identifier must be non-empty")
        if self.version < 1:
            raise ValueError("action-space version must be >= 1")
        if not self.actions:
            raise ValueError("action space must contain at least one action")
        keys = [action.key for action in self.actions]
        if len(set(keys)) != len(keys):
            raise ValueError(f"action keys must be unique: {keys}")
        names = [name for action in self.actions for name in (action.key, *action.aliases)]
        if len(set(name.lower() for name in names)) != len(names):
            raise ValueError("action keys and aliases must be unique ignoring case")

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(action.key for action in self.actions)

    def validate_index(self, action_index: int) -> int:
        if not 0 <= action_index < len(self.actions):
            raise ValueError(
                f"action_index must be in [0, {len(self.actions)}), got {action_index}"
            )
        return action_index

    def key_for(self, action_index: int) -> str:
        return self.actions[self.validate_index(action_index)].key

    def index_for(self, action_name: str) -> int:
        normalized = action_name.lower()
        for index, action in enumerate(self.actions):
            names = (action.key, *action.aliases)
            if normalized in {name.lower() for name in names}:
                return index
        raise ValueError(
            f"unknown action {action_name!r} for action space {self.identifier!r}"
        )
