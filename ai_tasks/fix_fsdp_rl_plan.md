# fix/fsdp — RL FSDP safety refactor direction

## 背景

`ai_tasks/merge_dev.md` 记录了合并后发现的 RL 多 GPU 问题：当前 `src/nimloth/training/rl/trainer.py` 在 `world > 1` 时把 Qwen 包成 FSDP，但 `EnvRolloutCollector` 仍直接持有这个 FSDP-wrapped Qwen 做动态 online env rollout。动态 episode 长度、图片数、失败/提前结束会让各 rank 触碰 FSDP forward 的次数和形状不一致，存在 deadlock/错误训练风险。

子 Agent 对 VAGEN/verl 的调查结论：VAGEN 不直接用 FSDP model 做动态 rollout。它把 rollout 与训练分阶段：rollout 用 vLLM/inference 副本，FSDP model 只在所有 rank 同步的训练/logprob/value 阶段被触碰；batch 会 padding/chunk，保证每个 rank 进入同一组 collective。

## 人类确认的方向

选择方案 A：**VAGEN-style 两阶段化 online RL**。

目标是把 Nimloth RL 的安全边界改为：

1. rollout 阶段使用独立 inference backend 生成 trajectories/jsonl；不要让 distributed/FSDP trainer 中的 FSDP Qwen 直接和 env 动态交互。
2. training 阶段只消费固定的 rollout trajectories/transitions；所有 FSDP ranks 只在同步、固定 batch 的训练/encoding/logprob 阶段触碰 FSDP model。
3. 如果当前代码还不能完整实现 vLLM rollout backend，也必须先加 guard，禁止危险路径 silent 运行，并提供明确的 external rollout/jsonl 入口。

## 实现要求

### 必须修复

- `train_rl()` 中，当 `world > 1` 且 collector 是 `EnvRolloutCollector` 时，禁止继续运行，并报清楚：distributed/FSDP trainer 不能直接做 dynamic env rollout；请先用独立 rollout backend 生成 JSONL，再用 JSONL collector 训练。
- 让 JSONL/offline rollout collector 真正可用于训练：不要只读取当前 iteration output dir 下不存在的 `trajectories.jsonl`。需要支持从配置/CLI 指定一个或多个 rollout JSONL 输入，并按 iteration 消费或循环消费。
- FSDP training 使用 JSONL/offline 数据时，要避免各 rank 数据数量不一致导致 collective 不一致。可采用简单、安全策略：
  - 每个 iteration 由所有 rank 基于同一 JSONL 构造 transitions；
  - 使用相同随机种子/iteration 选择 batch；
  - 或显式按 world 可整除 padding/drop_last；
  - 关键是所有 rank 在 `build_rl_transitions()` / PPO logprob forward / backward 中触碰 FSDP model 的次数一致。
- 修复 PPO advantage batch size 1 的 NaN：使用 `std(unbiased=False)` 或对单样本 batch 特判。
- 更新 `ai_tasks/merge_dev.md` 中对应问题状态；如果有新增限制，也写入 `experiments/training/rl/README.md` 或 `src/nimloth/training/rl/README.md`。

### 可以接受的短期边界

- 可以先不实现真正 vLLM rollout backend，只要现有 `experiments/training/rl/rollout_env.py` 或其他独立 rollout 脚本能作为 external rollout 入口，并且 trainer 明确要求 distributed 训练走 JSONL/offline 输入。
- 可以先不优化性能；优先语义安全。
- 可以保留单进程 `EnvRolloutCollector` 在线路径（`world == 1`），但文档必须说明它不适用于 distributed/FSDP。

### 不要做

- 不要让每个 rank 独立 dynamic env rollout 并直接用 FSDP Qwen forward。
- 不要实现“rank0 用 FSDP model rollout，其他 rank 等待”的方案；这会违反 FSDP collective 语义。
- 不要用 placeholder 冒充 vLLM rollout backend。如果没有真实实现 vLLM，就明确写 external rollout/jsonl path。
- 不要修改 unrelated SFT2 代码。

## 验证建议

本地至少运行：

```bash
python -m py_compile src/nimloth/training/rl/*.py experiments/training/rl/*.py tests/test_rl_*.py
bash -n experiments/training/rl/*.sh experiments/training/rl/*.slurm
```

如本地 pytest 环境可用，新增/更新测试覆盖：

- distributed + `EnvRolloutCollector` guard；
- JSONL collector 能从指定文件读取并按 iteration 返回 trajectories；
- advantage normalization batch size 1 不产生 NaN。

## 待主 Agent 审批

实现完成后不要自行推送；提交在 `fix/fsdp` 分支，由主 Agent review 后决定是否修改/合并。
