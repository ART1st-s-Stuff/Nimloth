"""RL trainer 对 rollout source 的启动约束与模型绑定。"""

from __future__ import annotations

import json
from pathlib import Path

from nimloth.agent import AgentPolicy
from nimloth.rollout import (
    FreshJSONLRolloutCollector,
    JSONLRolloutCollector,
    RolloutCollector,
)
from nimloth.environment.navigation.collector import VAGENNavigationRolloutCollector
from nimloth.util.distributed import is_main


def validate_collector_configuration(
    *,
    actor_enabled: bool,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    validation_enabled: bool,
) -> None:
    """在加载模型前拒绝静态 actor 数据和缺失 validation source。"""

    if (
        actor_enabled
        and isinstance(train_collector, JSONLRolloutCollector)
        and not isinstance(train_collector, FreshJSONLRolloutCollector)
    ):
        raise ValueError(
            "actor training requires fresh trajectories from the current policy; "
            "static JSONL rollout is only supported for WM/value training"
        )
    if validation_enabled and eval_collector is None:
        raise ValueError("validation.enabled requires a separate eval collector")


def validate_fresh_rollout_policy(train_collector: RolloutCollector) -> None:
    """只由 rank0 计算大 artifact 指纹，再向所有训练 rank 广播结果。"""

    if not isinstance(train_collector, FreshJSONLRolloutCollector):
        return
    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    error_message: str | None = None
    if rank == 0:
        try:
            train_collector.validate_policy()
        except Exception as error:
            error_message = str(error)
    if distributed:
        messages = [error_message]
        dist.broadcast_object_list(messages, src=0)
        error_message = messages[0]
    if error_message is not None:
        raise ValueError(error_message)


def validate_planning_initialization(
    *,
    planning_enabled: bool,
    online_policy_needed: bool,
    resume_loaded: bool,
    wm_checkpoint: Path | None,
    state_proj_checkpoint: Path | None,
    value_head_checkpoint: Path | None,
) -> None:
    """在线规划不能用尚未训练的随机 WM/Value 模块选择真实动作。"""

    if not planning_enabled or not online_policy_needed:
        return
    explicit_modules = all(
        path is not None
        for path in (
            wm_checkpoint,
            state_proj_checkpoint,
            value_head_checkpoint,
        )
    )
    if not resume_loaded and not explicit_modules:
        raise ValueError(
            "online WM planning requires a resumed RL checkpoint or explicit "
            "--wm-checkpoint, --state-proj-checkpoint and --value-head-checkpoint"
        )


def online_policy_required(
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
) -> bool:
    """判断当前 rollout source 是否需要绑定在线 Agent policy。"""

    return any(
        isinstance(candidate, VAGENNavigationRolloutCollector)
        for candidate in (train_collector, eval_collector)
        if candidate is not None
    )


def bind_online_collectors(
    *,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    policy: AgentPolicy,
    latent_token_count: int,
    world_size: int,
) -> None:
    """把 trainer 已加载的 Qwen 绑定给单卡在线 collector。"""

    online_collectors = [
        candidate
        for candidate in (train_collector, eval_collector)
        if isinstance(candidate, VAGENNavigationRolloutCollector)
    ]
    if not online_collectors:
        return
    if world_size > 1:
        raise RuntimeError(
            "分布式/FSDP trainer 不能直接让在线 rollout collector 使用 "
            "FSDP-wrapped Qwen 做动态 env rollout。各 rank 的 episode 长度、"
            "图片数和失败时机不同，会使 forward 次数不一致。请先用独立 "
            "rollout backend 生成 JSONL，再用 --use-jsonl-rollout 训练。"
        )
    for collector in online_collectors:
        collector.bind_policy(
            policy,
            latent_token_count=latent_token_count,
        )
    if is_main():
        print(json.dumps({"env_collector": "wired"}))


__all__ = [
    "bind_online_collectors",
    "online_policy_required",
    "validate_collector_configuration",
    "validate_fresh_rollout_policy",
    "validate_planning_initialization",
]
