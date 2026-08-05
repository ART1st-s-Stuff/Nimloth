"""训练和评估共用的 rollout 数据源与采样配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout source、dataset split 与 behavior sampling 配置。"""

    train_datasets: tuple[str, ...] = ()
    eval_datasets: tuple[str, ...] = ()
    jsonl_train_sources: tuple[str, ...] = ()
    jsonl_eval_sources: tuple[str, ...] = ()
    temperature: float = 1.0
    top_p: float = 1.0
    max_episode_attempts: int = 1


def parse_rollout_config(raw: Mapping[str, Any] | None) -> RolloutConfig:
    """严格解析 rollout 配置。"""

    values = {} if raw is None else raw
    if not isinstance(values, Mapping):
        raise ValueError("rollout config must be a mapping")
    allowed = {
        "train_datasets",
        "eval_datasets",
        "jsonl_train_sources",
        "jsonl_eval_sources",
        "temperature",
        "top_p",
        "max_episode_attempts",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown rollout config field: {unknown[0]}")
    temperature = float(values.get("temperature", 1.0))
    top_p = float(values.get("top_p", 1.0))
    max_episode_attempts = int(values.get("max_episode_attempts", 1))
    if temperature < 0.0:
        raise ValueError("rollout.temperature must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("rollout.top_p must be in (0, 1]")
    if max_episode_attempts < 1:
        raise ValueError("rollout.max_episode_attempts must be positive")
    return RolloutConfig(
        train_datasets=_string_tuple(
            values.get("train_datasets"),
            "rollout.train_datasets",
        ),
        eval_datasets=_string_tuple(
            values.get("eval_datasets"),
            "rollout.eval_datasets",
        ),
        jsonl_train_sources=_string_tuple(
            values.get("jsonl_train_sources"),
            "rollout.jsonl_train_sources",
        ),
        jsonl_eval_sources=_string_tuple(
            values.get("jsonl_eval_sources"),
            "rollout.jsonl_eval_sources",
        ),
        temperature=temperature,
        top_p=top_p,
        max_episode_attempts=max_episode_attempts,
    )
