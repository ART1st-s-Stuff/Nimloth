# RL 训练实验

从 SFT2 checkpoint warm-start，在线/离线 RL 训练 WM predictor + value head。

## 脚本

| 文件 | 用途 |
|------|------|
| `smoke_test.slurm` | 单 GPU smoke test：加载 SFT2 checkpoint，synthetic data 跑 1 步训练 |
| `rollout_env.py` | 独立 rollout 脚本：复用 Nimloth action/planning policy，生成完整JSONL及逐eval-set episode指标（不参与训练） |
| `run_e2e_smoke.sh` | 训练 split rollout → 两卡 FSDP step → resume step 的端到端 smoke |
| `run_vllm_online_ppo_smoke.sh` | config-sized vLLM fresh rollout → 指纹门槛 → distributed RL step |
| `run_vllm_online_ppo_slurm.sh` | 按 RL config 在异构多节点 allocation 上编排 Ray-vLLM 与 RL trainer |
| `run_vllm_online_ppo_full.sh` | 每次迭代重新采集 fresh rollout，并从上一步 checkpoint 恢复的正式多迭代入口 |

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

planner rollout支持`greedy/exhaustive/beam/mcts`。MCTS必须显式传入
`--mcts-num-simulations`和`--mcts-exploration-constant`；`--seed-per-eval-set`
让多个held-out dataset各自使用同一seed区间。SFT2完成后、RL之前的H=1/K自动绑定
入口位于`experiments/training/sft2/eval_mcts_rollout.py`。

### 多 GPU online RL

多 GPU 模式不让 environment 直接调用 FSDP forward。启动器先用当前完整
HF artifact 启动 vLLM tensor parallel rollout，写入模型内容指纹 manifest；
vLLM 退出后，distributed trainer 只允许消费该 manifest 一次。异构 Slurm 路径从
`distributed.nodes`、`distributed.world_size`、`distributed.gpus_per_rank` 和
`distributed.rollout_tensor_parallel_size` 读取资源语义；shell 不固定节点数或
GPU 总数。控制器为每个物理节点启动一个持有该节点全部已分配 GPU 的 Slurm
step，再按节点内 GPU 数启动 local ranks，因此允许各节点 GPU 数不同。
`world_size` 是训练进程数，物理 GPU 总数为
`world_size * gpus_per_rank`。`gpus_per_rank=1` 使用 FSDP；`gpus_per_rank=2`
让每个 Qwen 副本均衡分布在同一节点的两张 GPU 上，再由包住完整
planner Backbone forward边界的官方`DistributedDataParallel(device_ids=None)`
同步各rank梯度。DDP forward直接返回critic消费的
`BackboneOutput.hidden`；
不能只包住HF Qwen后在外层消费forward-hook hidden，否则reducer无法可靠
跟踪该反向图。direct-Qwen actor路径的loss直接消费model logits，因此
仍保留包住HF model的DDP边界。两条路径都没有手工gradient averaging。
后者要求每个节点分配的 GPU 数能被 2 整除，不支持跨节点拼接一个模型副本。
下一个 optimizer step 必须从新 checkpoint 重新 rollout。均匀两节点四卡也统一使用
`run_vllm_online_ppo_slurm.sh`，不维护固定节点拓扑的另一套入口。

正式多迭代训练使用`run_vllm_online_ppo_full.sh`。它为每个iteration的每个训练dataset
分配不重叠seed块，
用更新前的不可变policy snapshot完成rollout/reference replay，再执行恰好一次update；
下一轮只有在fresh consumption提交且`latest`完整后才开始。中间进程退出只写`latest`，
目标iteration完成时才写`final`。默认正式配置是greedy H=2、两节点四卡；额外DDP副本
不会增加有效episode batch。每个global step只序列化一次完整模型与optimizer，属于同一
训练状态的`iter_NNNN`、`latest`和`final`使用同文件系统硬链接，不重复写入或占用三份
checkpoint数据。外层入口只保留最近一个更新前policy snapshot；更早snapshot的删除会
写入相邻iteration progress log。
当`validation.external=true`时，外层入口在到期checkpoint完整提交后调用同一并行
rollout controller。当前32-GPU正式合同每10个iteration使用greedy sampling
评估held-out `base/common_sense`各seeds 1--60，结果写入
`evaluation/eval_step_log.csv`与独立eW&B eval run；评估rollout不进入optimizer。

### 分布式安全说明

- `JSONLRolloutCollector` 在所有rank上返回同一个全局轨迹序列，但planner训练把
  真实transition按rank-strided index全局唯一分配；每条真实transition只在一个rank
  上forward/backward。尾部不整除时，其他rank对一条实transition构建零loss graph
  padding，只为保持DDP collective序列一致，不把padding计入有效batch。
- Batch 选择使用 per-iteration 确定性 generator（`seed + iteration`），不依赖全局 RNG 状态同步。
- `state_proj`、`wm_predictor`、`value_head` 的可训练副本由 DDP 同步；冻结模块在 distributed setup 后从 rank0 广播初始状态。
- 所有 rank 必须调用相同的 `collect()` 次数——训练循环已保证这一点。

## 输出

```
outputs/experiments/training/rl/<date>/<name>/
├── README.md
├── train_step_log.csv
├── evaluation/
│   ├── eval_step_log.csv
│   └── iter_NNNN/          # fixed-seed held-out rollout and summary
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
