# RL 训练实验

从 metadata-bearing SFT2 checkpoint warm-start，训练 k>1/inject WM predictor + value head；Qwen 行为轨迹可额外做 PPO，WM/value fast-path 轨迹不进入 Qwen PPO。

## 脚本

| 文件 | 用途 |
|------|------|
| `smoke_test.slurm` | 单 GPU smoke test：加载 SFT2 checkpoint，synthetic data 跑 1 步训练 |
| `rollout_env.py` | 独立 rollout 脚本：复用 Nimloth action policy，生成完整 RL JSONL（不参与训练） |
| `run_e2e_smoke.sh` | 训练 split rollout → 两卡 FSDP step → resume step 的端到端 smoke |

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
# Qwen behavior policy（记录真实 temperature/top-p behavior log-prob）
python -m experiments.training.rl.rollout_env \
  --model /path/to/sft2/checkpoint \
  --policy qwen \
  --env-url http://127.0.0.1:5000 \
  --output-dir outputs/rollouts/qwen_batch_001 \
  --num-episodes 128 \
  --eval-set base_train \
  --split train

# WM+ValueHead greedy fast path（不生成/伪造 Qwen log-prob）
python -m experiments.training.rl.rollout_env \
  --model /path/to/sft2/checkpoint \
  --wm-checkpoint /path/to/sft2/checkpoint \
  --policy wm_value \
  --fast-path-horizon 2 \
  --env-url http://127.0.0.1:5000 \
  --output-dir outputs/rollouts/wm_batch_001 \
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

`--jsonl-sources` 接受一个或多个 JSONL 文件或目录（目录下递归搜索 `*.jsonl` / `*.jsonl.gz`）。也可以在 config 中设置 `rollout.jsonl_sources`。训练时轮转消费所有轨迹；数据耗尽时自动回到开头（loop）。每个 transition 保存 `policy_source`、`state_source`、fast-path step 及 inject/k metadata；只有 `policy_source=qwen`、`action_log_prob_semantics=sampling_distribution_v1` 且存在真实 behavior log-prob 的 row 可进入 Qwen PPO。旧 JSONL 没有该语义标记时仍可训练 WM/value，但会从 Qwen PPO 自动排除。

### 分布式安全说明

- `JSONLRolloutCollector` 在所有 rank 上返回相同轨迹序列（确定性轮转），保证 FSDP forward 次数一致。
- Batch 选择使用 per-iteration 确定性 generator（`seed + iteration`），不依赖全局 RNG 状态同步。
- 非 FSDP 的 `state_proj`、`wm_predictor`、`value_head` 会在 distributed setup 后从 rank0 广播初始参数；因为所有 rank 消费相同数据，它们的本地副本会保持同步。
- 所有 rank 必须调用相同的 `collect()` 次数——训练循环已保证这一点。
- 多步 dynamics loss 只使用同一 trajectory 内的连续 action/state window，短 episode 用显式 mask padding。

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
