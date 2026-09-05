# SFT2 speedup 实现进度

任务来源：`ai_tasks/sft2_speedup_plan.md`

## 目标

降低 SFT2 wall-clock，**不改变**训练语义（数据 split、loss、梯度路径、checkpoint、默认 tuning）。

## 已完成

### P0 — step timing

- `src/nimloth/training/sft2/profiling.py`：`StepTimer`
- CLI：`--step-timing`, `--step-timing-interval`
- trainer 记录 dataloader / current_forward / next_forward / value_loss / backward / optimizer 分段耗时

### P1 — preprocess cache + DataLoader workers

- `src/nimloth/training/sft2/preprocess_cache.py`：磁盘 cache、manifest fingerprint、多进程构建、`CachedTransitionDataset`
- **trajectory 级 cache**：`encode_trajectory_record`、`CachedTrajectoryDataset`、`build_trajectory_preprocess_cache`（packed 模式用，避免 O(T²) transition cache 膨胀）
- manifest 写入 `total_bytes`
- transition cache 不再落盘 `next_messages`（`__getitem__` 从 `TransitionSample` 补回）
- `encode_qwen_item()` in `qwen_batch.py`：与在线 `build_qwen_batch` 单样本等价
- CLI：`--preprocess-cache-dir`, `--preprocess-workers`, `--force-rebuild-cache`, `--dataloader-workers`
- 启用 cache 时默认 `num_workers=4` + prefetch
- **修复**：`image_grid_thw` 单图时保持 `[1, 3]` 形状（避免 vision delta 切片错误）
- `experiments/training/sft2/estimate_preprocess_cache.py`：抽样估算 transition vs trajectory cache 体积

### P2 — batch_size=2 / grad_accum=4

- **已并入默认配置** `configs/training/sft2/latent_wm_value.yaml`（人类批准增大 batch_size）
- 保留 `latent_wm_value_profiling.yaml` 作为短跑 profiling 模板

### P3 — 同 step next target 去重

- `step.py`：`_forward_next_latents` 对 batch 内相同 `next_messages` 只 forward 一次
- 支持 cached `next_enc_rows`

### P4 — trajectory-once packed forward（已集成 trainer，待 GPU 全量验收）

**方案切换**：由 KV 递增改为 **整条 trajectory 一次完整 Qwen forward**（`trajectory_once.py`）。

- `src/nimloth/training/sft2/trajectory_once.py`：
  - `encode_full_trajectory` / `forward_trajectory_once`
  - 从所有 `<|latent_state|>` 位置抽 current/next latent
  - **token-weighted CE 全局 mean**（与 legacy `build_qwen_batch` 一致，非 per-step loss mean）
- `trajectory_batching.py`：按 `record_id` + 连续 step 切 batch；`TrajectoryRecordDataset` 默认 **一条 trajectory 一个 micro-batch**
- `trainer.py` / `evaluate.py`：packed 分支接入 `forward_trajectory_once` + `compute_trajectory_wm_loss`；移除 trainer 对 KV 递增路径的依赖
- `trajectory_equiv.py` + `probe_trajectory_once_equiv.py`：legacy vs packed 对比（latent + CE + WM + value + total）
- `smoke_speedup.py`：`packed_once_equiv` 检查（`--require-packed-once-equiv`）
- **保留研究对照**：`packed_trajectory.py`（KV 递增）仅供 `probe_kv_trajectory.py` 使用，trainer 不再调用
- **生产默认**：`submit_packed_forward_8gpu.sh` 中 `PACKED_FORWARD=0`、`PREPROCESS_CACHE_DIR` 不默认开启

**验收门槛**（GPU，≥3 records）：latent `max_diff < 1e-2`（bf16）；loss `abs_diff < 1e-3`。未通过前勿默认开启生产训练。

**packed 模式 grad_accum**：micro-batch = 1 trajectory（~T 步）；建议使 `world * grad_accum * avg_T ≈ 64`，或验证 per-trajectory mean loss 与 legacy 曲线一致。

### P6 — 诊断配置

- `configs/training/sft2/latent_wm_value_vision_freeze_profiling.yaml`

### 测试

- `tests/training/sft2/test_preprocess_cache.py`
- `tests/training/sft2/test_step_next_dedup.py`
- `tests/training/sft2/test_profiling.py`
- `tests/training/sft2/test_trajectory_batching.py`（含 record 边界）
- `tests/training/sft2/test_trajectory_ce_aggregation.py`
- `tests/training/sft2/test_trajectory_forward.py`（CPU span/label）

## 未完成 / 待验证

- **GPU 全量等价验收**：`probe_trajectory_once_equiv.py` + `smoke_speedup.py --require-packed-once-equiv`（本环境无 torch/GPU，需在服务器跑）
- **P6** 诊断短跑（vision freeze/lora、no-checkpointing、attn 对比）
- **P5** vision feature cache（仅 freeze，plan 默认不做）

## 使用示例

```bash
# 默认路径不变；仅加 timing
python experiments/training/sft2/train.py --config configs/training/sft2/latent_wm_value.yaml \
  --model ... --train-jsonl ... --val-jsonl ... --output-dir ... --step-timing

# preprocess cache（train/val 子目录自动创建；packed 用 train_trajectory/val_trajectory）
python experiments/training/sft2/train.py ... \
  --preprocess-cache-dir /path/to/sft2_preprocess_cache

# 估算 cache 体积
python experiments/training/sft2/estimate_preprocess_cache.py \
  --model ... --train-jsonl ...

# P4 trajectory-once 等价性探测（GPU，≥3 records）
python experiments/training/sft2/probe_trajectory_once_equiv.py \
  --model ... --train-jsonl ...

# 服务器 GPU smoke（1 GPU）
sbatch experiments/training/sft2/smoke_speedup.slurm

# packed-forward 训练（opt-in，需先通过 probe）
PACKED_FORWARD=1 sbatch experiments/training/sft2/submit_packed_forward_8gpu.sh
```
