---
name: on-experiment-start
description: >-
  强制执行Nimloth实验启动合同。在任何训练、评估、收集、校准、rollout-train、远程长job、Slurm任务或GPU/昂贵计算之前立即使用。
---

# 实验启动门禁

## 触发条件

任何实验或昂贵/远程job的启动命令执行前都必须暂停；即使实施准备已经获批，也不能跳过本门禁。

## 必须执行

1. 阅读[实验索引](../../../.trellis/spec/experiments/index.md)、[任务合同](../../../.trellis/spec/experiments/task-contract.md)、[数据/split规则](../../../.trellis/spec/experiments/data-and-splits.md)、[启动/生命周期](../../../.trellis/spec/experiments/launch-and-lifecycle.md)和[输出/checkpoint证据](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md)。
2. 确认当前任务含有`task.json.meta.kind = "experiment"`，所有真实实验将remote-only执行，并且每个必填字段都已明确、有来源证据支持并已核验。
3. 搜索相关curated memory。任何会影响启动的memory都必须先执行`get`，再重新阅读其证据。
4. 阅读与任务相关的known errors；远程工作还必须阅读`.local/SERVER.md`和`slurm` skill。
5. 确认本地工作已经commit、精确commit已经记录，且远程worktree正在使用该commit。
6. 核验最终命令/config、完整参数名、数据/split、checkpoint所有权、train/freeze/objectives、唯一输出、恢复方式、指标/有效性门禁、W&B标识和资源/时间估算。
7. 若预计10分钟内完成，声明为短实验，并固定从成功提交起覆盖pending与running的15分钟总deadline、取消命令及defer/blocker路由。
8. 若是已审查scope内、预计10分钟内完成的remote reproduction或部分参数重跑，完整记录轻量task后无需额外启动询问；预计超过10分钟或实质性新实验必须展示精确合同并取得单独、明确的启动审批。
9. 任一项目缺失，或获批后发生变化时，必须停止并重新询问。禁止启动近似替代方案。
10. 启动后持续监控调度器/进程/日志/资源/指标/输出状态，直到job健康运行或进入终止状态；短实验到达deadline必须取消、核验终态并记录为defer或blocker，不能冒充验证成功。
