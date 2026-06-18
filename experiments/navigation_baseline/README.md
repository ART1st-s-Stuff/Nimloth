# Navigation Baseline with VAGEN

> **遗留反模式（待清理）**  
> 本目录含大量一次性 Slurm/submit 脚本，不符合当前实验目录规范。  
> **不要在此新增** `dgx*` / `retry*` / 节点名写死的脚本。  
> 新实验请遵循 [`experiments/README.md`](../README.md) 与 [`ai_rules/03_experiments_and_data.md`](../../ai_rules/03_experiments_and_data.md)；目标入口为 `experiments/training/` + `configs/training/`。  
> 下文保留作历史参考，迁移状态见 `ai_tasks/sft2_phase2_plan.md`。

基于 VAGEN (NeurIPS 2025) 的 Navigation 环境 baseline 训练。

## 架构

```
┌─────────────────────┐     ┌──────────────────────────────┐
│   Env Server Nodes ×2│     │      Training Nodes (×2)      │
│   (4 GPU each)       │     │      (8 GPU each = 16 GPU)    │
│                     │     │                              │
│  AI2-THOR envs      │◄───►│  Qwen2.5-VL-3B (FSDP)       │
│  :8000 (HTTP)       │     │  Ray cluster (head+worker)    │
│                     │     │  VERL PPO + sglang           │
└─────────────────────┘     └──────────────────────────────┘
```

## 超参数

| 参数 | 值 | 来源 |
|------|------|------|
| Model | Qwen2.5-VL-3B-Instruct | Paper navigation setting |
| Actor LR | 1e-6 | Paper/Examples |
| Critic LR | 1e-5 | Paper/Examples |
| Batch size | 128 | Paper/Examples |
| PPO mini-batch | 32 | Paper/Examples |
| GAE gamma | 1.0 | Paper |
| GAE lambda | 1.0 | Paper |
| KL coef | 0.0 | Paper/Examples |
| **max_actions_per_step** | **1** | User requirement |
| **max_turns** | **20** | User requirement |
| **save_freq** | **1** (every PPO step) | User requirement |
| total_training_steps | 50 | User requirement |
| checkpoint retention | last 10 global steps + best validation step | User requirement |

## 数据集

**Training (disjoint seeds for base vs common):**
- `base_train.json` (25k tasks, seeds 1-4096)
- `common_sense_train.json` (seeds 1-4096)

**Validation (disjoint seed range):**
- `base.json` (1.2k tasks, seeds 4097-4608)

## 运行步骤

### 1. 安装依赖

使用 project-local uv 环境，避免 `/home` 空间不足或无权访问他人的 conda env：

```bash
cd /project/peilab/atst/nimloth
.local/bin/uv venv --python 3.12 .venv
sbatch experiments/navigation_baseline/install_vagen_env.slurm
```

脚本会安装 VAGEN/VERL/sglang/ray/AI2-THOR，并预下载 navigation scenes。

### 2. 启动 Env Server

```bash
bash experiments/navigation_baseline/launch_env_servers.sh

# 默认: 两个 4-GPU env server，每个 max_envs=64/thread_pool=64，总容量128
# 单独提交一个 env server 可传参数:
# sbatch experiments/navigation_baseline/env_server.slurm [port] [max_envs] [thread_pool]
```

Env server 会将 hostname 写入 `env_server_host.txt`，training job 会读取。

### 3. 启动训练

```bash
# 确认 env server 已启动且 env_server_host.txt 存在
sbatch experiments/navigation_baseline/train.slurm
```

### 4. 监控

```bash
# 查看 job 状态
squeue -u $USER

# 查看 env server 日志
tail -f vagen-nav-env_*.out

# 查看训练日志
tail -f vagen-nav-train_*.out

# 检查 checkpoint
ls experiments/navigation_baseline/runs/vagen_nav_baseline_maxact1_turns20/checkpoints/
```

## 输出

- **Checkpoints**: `runs/vagen_nav_baseline_maxact1_turns20/checkpoints/` (每个 global PPO step 保存；训练结束后 prune 为最近10步 + best validation step)
- **Rollout data**: `runs/.../rollout_data/`
- **Validation**: `runs/.../validation/`
- **Logs**: `runs/.../vagen_nav_baseline_maxact1_turns20.log`

## 注意事项

- Env server 需要 AI2-THOR (Unity 渲染)，需要 GPU 支持 CloudRendering
- Env server 默认通过 `launch_env_servers.sh` 启动 2 个 4-GPU server（每个 `max_envs=64`，合计 128），与 `train_batch_size=128` 对齐；如仍成为瓶颈，可增加 server 数并在 hostfile 中追加 URL
- 训练使用 2 nodes × 8 GPUs = 16 GPUs，需要 cluster 有足够资源
- 当前 Slurm 默认：env server 用 `preempt`，training 用 `normal`，account 为 `peilab`
