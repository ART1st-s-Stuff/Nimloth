# Implementation Plan

1. 为 Stage 1 CLI、YAML 映射和标准配置增加 `format_eval_batch_size` 正数参数，并传入 `evaluate_format`。
2. 将 prompt 准备与批量 processor/generate 分离，正确处理多模态图片顺序、左 padding、FSDP 同步和 EOS 后 padding 裁剪。
3. 增加 LM 验证、格式生成总耗时和逐批结构化进度，保持现有指标字段兼容。
4. 增加聚焦测试：batch 1/4 等价、调用次数、变长输入、提前 EOS、尾批、padding 设置恢复、参数校验。
5. 运行聚焦及相邻测试、编译检查和 diff 检查；用 Trellis check 复核完整数据流。
6. 提交独立代码 commit。重新查询 a100-1，选择最新完整 epoch checkpoint，按原实验合同恢复并测量首轮批量验证。

## Rollback points

- 代码回滚到当前实验 commit `2a8ac7122c8e1f389e01cd360cb592cb9d308d80`。
- 远程验证失败时不覆盖原 checkpoint；停止新进程后仍可从同一完整 epoch 以旧 commit 恢复。

