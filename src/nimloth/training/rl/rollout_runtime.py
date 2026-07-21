"""RL trainer 对 rollout source 的启动约束与模型绑定。"""

from __future__ import annotations

import json
from typing import Any

from nimloth.backbone.qwen25vl.vagen_rollout import (
    VAGENNavigationRolloutCollector,
)
from nimloth.rollout import (
    JSONLRolloutCollector,
    RolloutCollector,
)
from nimloth.util.distributed import is_main


def validate_collector_configuration(
    *,
    actor_enabled: bool,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    validation_enabled: bool,
) -> None:
    """在加载模型前拒绝静态 PPO 数据和缺失 validation source。"""

    if actor_enabled and isinstance(train_collector, JSONLRolloutCollector):
        raise ValueError(
            "PPO actor requires fresh trajectories from the current policy; "
            "static JSONL rollout is only supported for WM/value training"
        )
    if validation_enabled and eval_collector is None:
        raise ValueError("validation.enabled requires a separate eval collector")


def bind_online_collectors(
    *,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    model: Any,
    processor: Any,
    device: Any,
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
        collector.bind_policy(model, processor, device)
    if is_main():
        print(json.dumps({"env_collector": "wired", "device": str(device)}))
