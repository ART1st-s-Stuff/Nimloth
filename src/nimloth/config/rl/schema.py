"""严格且类型化的 RL 阶段配置。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from nimloth.config.agent import AgentConfig, parse_agent_config
from nimloth.config.io import load_yaml_config
from nimloth.config.rollout import RolloutConfig, parse_rollout_config


def _section(
    root: Mapping[str, Any],
    name: str,
    allowed: set[str],
) -> Mapping[str, Any]:
    value = root.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"RL config section {name!r} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown RL config field: {name}.{unknown[0]}")
    return value


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    result = int(value)
    lower = 0 if allow_zero else 1
    if result < lower:
        raise ValueError(f"{field} must be >= {lower}, got {result}")
    return result


def _positive_float(value: Any, field: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        operator = ">=" if allow_zero else ">"
        raise ValueError(f"{field} must be {operator} 0, got {result}")
    return result


def _boolean(value: Any, field: str) -> bool:
    """拒绝会被 Python 当作真值的字符串或数字。"""

    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True)
class ActorConfig:
    lr: float = 1e-6
    entropy_coeff: float = 0.0
    clip_ratio: float = 0.2


@dataclass(frozen=True)
class FreezeConfig:
    state_proj: bool = True


@dataclass(frozen=True)
class PredictorConfig:
    lr: float = 1e-3
    emb_dim: int = 128
    history_size: int = 4
    lambda_sigreg: float = 0.1
    sigreg_num_proj: int = 1024
    sigreg_knots: int = 17


@dataclass(frozen=True)
class ValueHeadConfig:
    lr: float = 1e-3
    rank_margin: float = 0.1
    lambda_rank: float = 0.0


@dataclass(frozen=True)
class RLLoopConfig:
    iterations: int = 1000
    envs_per_iteration: int = 8
    max_steps_per_episode: int = 20
    gamma: float = 0.99
    batch_size: int = 32


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    interval: int = 50
    envs: int = 16
    checkpoint_metric: str = "success_rate"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    log_interval: int = 1
    save_interval: int = 50


@dataclass(frozen=True)
class RLConfig:
    """训练代码唯一接收的 RL 配置对象。"""

    agent: AgentConfig
    actor: ActorConfig
    freeze: FreezeConfig
    predictor: PredictorConfig
    value_head: ValueHeadConfig
    rollout: RolloutConfig
    rl: RLLoopConfig
    validation: ValidationConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        """生成适合日志与 W&B 序列化的普通字典。"""

        return asdict(self)


def parse_rl_config(raw: Mapping[str, Any]) -> RLConfig:
    """校验原始 YAML mapping，并拒绝所有未实现字段。"""

    allowed_sections = {
        "agent",
        "actor",
        "freeze",
        "predictor",
        "value_head",
        "rollout",
        "rl",
        "validation",
        "training",
    }
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"unknown RL config section: {unknown_sections[0]}")

    actor = _section(raw, "actor", {"lr", "entropy_coeff", "clip_ratio"})
    freeze = _section(raw, "freeze", {"state_proj"})
    predictor = _section(
        raw,
        "predictor",
        {
            "lr",
            "emb_dim",
            "history_size",
            "lambda_sigreg",
            "sigreg_num_proj",
            "sigreg_knots",
        },
    )
    value_head = _section(
        raw,
        "value_head",
        {"lr", "rank_margin", "lambda_rank"},
    )
    loop = _section(
        raw,
        "rl",
        {
            "iterations",
            "envs_per_iteration",
            "max_steps_per_episode",
            "gamma",
            "batch_size",
        },
    )
    validation = _section(
        raw,
        "validation",
        {"enabled", "interval", "envs", "checkpoint_metric"},
    )
    training = _section(raw, "training", {"seed", "log_interval", "save_interval"})

    actor_config = ActorConfig(
        lr=_positive_float(actor.get("lr", 1e-6), "actor.lr"),
        entropy_coeff=_positive_float(
            actor.get("entropy_coeff", 0.0),
            "actor.entropy_coeff",
            allow_zero=True,
        ),
        clip_ratio=float(actor.get("clip_ratio", 0.2)),
    )
    if not 0.0 < actor_config.clip_ratio < 1.0:
        raise ValueError("actor.clip_ratio must be in (0, 1)")

    rollout_config = parse_rollout_config(raw.get("rollout"))
    checkpoint_metric = str(
        validation.get("checkpoint_metric", "success_rate")
    )
    if checkpoint_metric not in {"success_rate", "avg_reward"}:
        raise ValueError(
            "validation.checkpoint_metric must be success_rate or avg_reward"
        )

    gamma = float(loop.get("gamma", 0.99))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("rl.gamma must be in [0, 1]")
    validation_enabled = _boolean(
        validation.get("enabled", True),
        "validation.enabled",
    )
    validation_envs = _positive_int(
        validation.get("envs", 16),
        "validation.envs",
        allow_zero=not validation_enabled,
    )

    return RLConfig(
        agent=parse_agent_config(raw.get("agent")),
        actor=actor_config,
        freeze=FreezeConfig(
            state_proj=_boolean(
                freeze.get("state_proj", True),
                "freeze.state_proj",
            )
        ),
        predictor=PredictorConfig(
            lr=_positive_float(predictor.get("lr", 1e-3), "predictor.lr"),
            emb_dim=_positive_int(predictor.get("emb_dim", 128), "predictor.emb_dim"),
            history_size=_positive_int(
                predictor.get("history_size", 4),
                "predictor.history_size",
            ),
            lambda_sigreg=_positive_float(
                predictor.get("lambda_sigreg", 0.1),
                "predictor.lambda_sigreg",
                allow_zero=True,
            ),
            sigreg_num_proj=_positive_int(
                predictor.get("sigreg_num_proj", 1024),
                "predictor.sigreg_num_proj",
            ),
            sigreg_knots=_positive_int(
                predictor.get("sigreg_knots", 17),
                "predictor.sigreg_knots",
            ),
        ),
        value_head=ValueHeadConfig(
            lr=_positive_float(value_head.get("lr", 1e-3), "value_head.lr"),
            rank_margin=_positive_float(
                value_head.get("rank_margin", 0.1),
                "value_head.rank_margin",
                allow_zero=True,
            ),
            lambda_rank=_positive_float(
                value_head.get("lambda_rank", 0.0),
                "value_head.lambda_rank",
                allow_zero=True,
            ),
        ),
        rollout=rollout_config,
        rl=RLLoopConfig(
            iterations=_positive_int(loop.get("iterations", 1000), "rl.iterations"),
            envs_per_iteration=_positive_int(
                loop.get("envs_per_iteration", 8),
                "rl.envs_per_iteration",
            ),
            max_steps_per_episode=_positive_int(
                loop.get("max_steps_per_episode", 20),
                "rl.max_steps_per_episode",
            ),
            gamma=gamma,
            batch_size=_positive_int(loop.get("batch_size", 32), "rl.batch_size"),
        ),
        validation=ValidationConfig(
            enabled=validation_enabled,
            interval=_positive_int(
                validation.get("interval", 50),
                "validation.interval",
            ),
            envs=validation_envs,
            checkpoint_metric=checkpoint_metric,
        ),
        training=TrainingConfig(
            seed=int(training.get("seed", 42)),
            log_interval=_positive_int(
                training.get("log_interval", 1),
                "training.log_interval",
            ),
            save_interval=_positive_int(
                training.get("save_interval", 50),
                "training.save_interval",
            ),
        ),
    )


def load_rl_config(config_path: Path) -> RLConfig:
    """加载并完整校验一份 RL YAML。"""

    return parse_rl_config(load_yaml_config(config_path))


def merge_rl_config_overrides(
    args: argparse.Namespace,
    config: RLConfig,
) -> RLConfig:
    """返回应用显式 CLI 覆盖后的新配置对象。"""

    loop = config.rl
    training = config.training
    if args.seed is not None:
        training = replace(training, seed=args.seed)
    if args.rl_iterations is not None:
        loop = replace(loop, iterations=args.rl_iterations)
    if args.rl_envs_per_iteration is not None:
        loop = replace(loop, envs_per_iteration=args.rl_envs_per_iteration)
    return replace(config, rl=loop, training=training)
