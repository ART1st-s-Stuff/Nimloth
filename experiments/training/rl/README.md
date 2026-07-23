# RL 训练实验

从 SFT2 checkpoint warm-start，在线/离线 RL 训练 WM predictor + value head。

## 脚本

| 文件 | 用途 |
|------|------|
| `smoke_test.slurm` | 单 GPU smoke test：加载 SFT2 checkpoint，synthetic data 跑 1 步训练 |
| `rollout_env.py` | 独立 rollout 脚本：复用 Nimloth action policy，生成完整 RL JSONL（不参与训练） |
| `run_e2e_smoke.sh` | 训练 split rollout → 两卡 FSDP step → resume step 的端到端 smoke |
| `run_vllm_online_ppo_smoke.sh` | 8卡 vLLM fresh rollout → 指纹门槛 → 8卡 FSDP PPO step |
| `run_vllm_online_ppo_2node4.sh` | 一个2节点×4卡 hold 内的 Ray-vLLM TP8 → 2×4 FSDP step |

## 运行模式

### 单 GPU 在线 rollout（`world == 1`）

```bash
python -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --env-url http://127.0.0.1:5000 \
  --output-dir outputs/experiments/training/rl/test
```

配置中的 `rollout.eval_sets` 必须显式列出环境实际支持的 `*_train` datasets；trainer 会拒绝把 `base/common_sense` 等 eval assets 当作训练数据。

### 分布式/离线 JSONL rollout（`world > 1`，推荐）

**分布式/FSDP 训练禁止直接使用 `EnvRolloutCollector`**。必须先通过独立 rollout 后端生成 JSONL，再离线消费：

```bash
# 步骤 1：独立 rollout 生成 JSONL（可在 Slurm 上单卡运行）
python -m experiments.training.rl.rollout_env \
  --model /path/to/sft2/export_best_hf \
  --env-url http://127.0.0.1:5000 \
  --output-dir outputs/rollouts/batch_001 \
  --num-episodes 128 \
  --eval-set base_train \
  --split train

# 步骤 2：离线 RL 训练消费 JSONL
python -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --use-jsonl-rollout \
  --jsonl-sources outputs/rollouts/ \
  --output-dir outputs/experiments/training/rl/test
```

`--jsonl-sources` 接受一个或多个 JSONL 文件或目录（目录下递归搜索 `*.jsonl` / `*.jsonl.gz`）。也可以在 config 中设置 `rollout.jsonl_sources`。训练时轮转消费所有轨迹；数据耗尽时自动回到开头（loop）。

### 8 卡 online PPO

8 卡模式不让 environment 直接调用 FSDP forward。启动器先用当前完整
HF artifact 启动 vLLM tensor parallel rollout，写入模型内容指纹 manifest；
vLLM 退出后，8-rank FSDP trainer 只允许消费该 manifest 一次。下一个
optimizer step 必须从新 checkpoint 重新 rollout。

没有完整空闲8卡节点时，`run_vllm_online_ppo_2node4.sh` 只接受均匀的两节点、
每节点4卡 allocation。它先验证 Ray cluster 精确暴露8卡，再调用同一 rollout/
trainer 入口；训练阶段使用两个 torchrun agent、每节点4个 rank，总 world size 仍为8。

### 分布式安全说明

- `JSONLRolloutCollector` 在所有 rank 上返回相同轨迹序列（确定性轮转），保证 FSDP forward 次数一致。
- Batch 选择使用 per-iteration 确定性 generator（`seed + iteration`），不依赖全局 RNG 状态同步。
- 非 FSDP 的 `state_proj`、`wm_predictor`、`value_head` 会在 distributed setup 后从 rank0 广播初始参数；因为所有 rank 消费相同数据，它们的本地副本会保持同步。
- 所有 rank 必须调用相同的 `collect()` 次数——训练循环已保证这一点。

## 输出

```
outputs/experiments/training/rl/<date>/<name>/
├── README.md
├── train_step_log.csv
├── best/                  # best checkpoint (state_proj, predictor, value_head, optimizer)
├── iter_NNNN/             # periodic checkpoints
├── rollouts/              # per-iteration trajectory JSONL
└── final/                 # final checkpoint
```

## 入口

```bash
# Smoke test (单 GPU，synthetic data)
sbatch experiments/training/rl/smoke_test.slurm

# 真实端到端 smoke（在至少 2 GPU 的 hold allocation 内执行）
bash experiments/training/rl/run_e2e_smoke.sh
```
