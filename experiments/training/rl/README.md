# RL 训练实验

从 SFT2 checkpoint warm-start，在线/离线 RL 训练 WM predictor + value head。

## 脚本

| 文件 | 用途 |
|------|------|
| `smoke_test.slurm` | 单 GPU smoke test：加载 SFT2 checkpoint，synthetic data 跑 1 步训练 |
| `rollout_env.py` | 独立 rollout 脚本：复用 Nimloth action policy，生成完整 RL JSONL（不参与训练） |
| `run_e2e_smoke.sh` | 训练 split JSONL → 两卡 FSDP step → resume step 的端到端 smoke |
| `dynamic_env_server.slurm` | 独立节点运行VAGEN/AI2-THOR环境，等待trainer完成 |
| `dynamic_fsdp_smoke.slurm` | 两卡NCCL/FSDP current-policy动态rollout trainer step |
| `dynamic_fsdp_smoke_hetero_2plus1.slurm` | 单个heterogeneous job原子申请dgx-32 trainer2卡+dgx-51 env1卡 |
| `dynamic_fsdp_smoke_single3.slurm` | dgx-32单节点3卡：env独占1卡、trainer使用2卡 |

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

### 分布式动态在线 rollout（`world > 1`）

```bash
torchrun --nproc-per-node=4 -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --model /path/to/k1-inject-sft2 \
  --env-url http://ENV_NODE:5000 \
  --llm-tune full --vision-tune freeze \
  --output-dir outputs/experiments/training/rl/online
```

`DistributedEnvRolloutCollector` 只让rank0访问HTTP环境；所有rank同步运行当前FSDP policy，rank0采样并广播action，然后step环境。每个iteration更新完成后，下一轮rollout直接使用更新后的policy。配置必须使用训练split并设`validation.enabled=false`；`rollout.history_window`、temperature、top-p和seed offset会写入checkpoint并在resume时核对。

### 分布式/离线 JSONL rollout

需要把 rollout 与训练分开时，仍可先生成 JSONL，再确定性消费：

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

### 分布式安全说明

- 动态模式中只有rank0拥有env状态，所有rank按同一顺序执行policy forward，并在step前核对action logits、广播rank0 action和完整trajectory。
- env/policy/schema错误不会写入默认动作或零log-prob；不完整episode整条丢弃，collective policy错误同步失败。
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
