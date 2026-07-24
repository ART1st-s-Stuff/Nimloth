"""SFT2 在线截断历史 state cache。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch


StateKey = tuple[str, int]
HISTORY_CACHE_VERSION = "online_detached_state_v1"


@dataclass
class OnlineHistoryStateCache:
    """缓存 state 在其 current step forward 时产生的 detached 值。

    cache 只跨同一 epoch、同一 phase 的后续 microbatch 使用。每个 key 必须先
    作为 current step 写入，未来 H-step WM context 才能读取；缺失或重复写入都
    说明 sampler 破坏了严格的 trajectory 时间顺序。
    """

    epoch: int | None = None
    phase: str | None = None
    _states: dict[StateKey, torch.Tensor] = field(default_factory=dict)

    def start(self, *, epoch: int, phase: str, resume: bool = False) -> None:
        epoch = int(epoch)
        phase = str(phase)
        if resume:
            if self.epoch != epoch or self.phase != phase:
                raise ValueError(
                    "history cache resume position mismatch: "
                    f"cache=({self.epoch}, {self.phase!r}), "
                    f"requested=({epoch}, {phase!r})"
                )
            return
        self.epoch = epoch
        self.phase = phase
        self._states.clear()

    def history(
        self,
        keys: Sequence[Sequence[StateKey]],
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """返回 ``(B,T-1,D)`` detached history；T=1 返回真实空时间轴。"""

        if reference.ndim < 2:
            raise ValueError(
                "current state reference must have shape (B,...state), "
                f"got {tuple(reference.shape)}"
            )
        if len(keys) != reference.shape[0]:
            raise ValueError(
                "history key batch size does not match current state batch: "
                f"keys={len(keys)}, states={reference.shape[0]}"
            )
        lengths = {len(row) for row in keys}
        if len(lengths) != 1:
            raise ValueError("history cache rows must share one context length")
        history_steps = lengths.pop() if lengths else 0
        if history_steps == 0:
            return reference.new_empty(
                (reference.shape[0], 0, *reference.shape[1:])
            )

        missing = [key for row in keys for key in row if key not in self._states]
        if missing:
            preview = ", ".join(f"{record}:{step}" for record, step in missing[:4])
            raise KeyError(
                "online history cache miss; sampler must emit every trajectory step "
                f"before its successor (missing {preview})"
            )
        rows = [
            torch.stack([self._states[key] for key in row], dim=0)
            for row in keys
        ]
        return torch.stack(rows, dim=0).to(
            device=reference.device,
            dtype=reference.dtype,
            non_blocking=True,
        )

    def store(
        self,
        keys: Sequence[StateKey],
        states: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> None:
        """保存 current state；padding batch 不写 cache。"""

        if not enabled:
            return
        if states.ndim < 2 or states.shape[0] != len(keys):
            raise ValueError(
                "current cache states must have shape (B,...state) matching keys, "
                f"got {tuple(states.shape)} for {len(keys)} keys"
            )
        seen: set[StateKey] = set()
        duplicates: list[StateKey] = []
        for key in keys:
            if key in self._states or key in seen:
                duplicates.append(key)
            seen.add(key)
        if duplicates:
            preview = ", ".join(f"{record}:{step}" for record, step in duplicates[:4])
            raise ValueError(
                "current step was emitted more than once in one cache phase: "
                f"{preview}"
            )
        for key, state in zip(keys, states, strict=True):
            self._states[key] = state.detach().to(device="cpu").clone()

    @property
    def count(self) -> int:
        return len(self._states)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HISTORY_CACHE_VERSION,
            "epoch": self.epoch,
            "phase": self.phase,
            "keys": list(self._states),
            "states": (
                torch.stack(list(self._states.values()), dim=0)
                if self._states
                else torch.empty((0, 0), dtype=torch.float32)
            ),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("version") != HISTORY_CACHE_VERSION:
            raise ValueError(
                "unsupported SFT2 history cache version: "
                f"{payload.get('version')!r}"
            )
        keys = [
            (str(record_id), int(step_index))
            for record_id, step_index in payload.get("keys", [])
        ]
        states = payload.get("states")
        if not isinstance(states, torch.Tensor) or states.ndim < 2:
            raise ValueError(
                "history cache checkpoint states must have shape (N,...state)"
            )
        if states.shape[0] != len(keys):
            raise ValueError("history cache checkpoint key/state counts do not match")
        self.epoch = int(payload["epoch"]) if payload.get("epoch") is not None else None
        self.phase = str(payload["phase"]) if payload.get("phase") is not None else None
        self._states = {
            key: state.detach().to(device="cpu").clone()
            for key, state in zip(keys, states, strict=True)
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(
            torch.load(Path(path), map_location="cpu", weights_only=False)
        )


__all__ = [
    "HISTORY_CACHE_VERSION",
    "OnlineHistoryStateCache",
    "StateKey",
]
