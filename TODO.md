# Phase 2 当前执行清单

下面所有命令默认从 `flower/` 仓库根目录执行：

```bash
cd /home/jincai_guo/atst/flower
```

W&B 配置从项目 `.env` 读取。确认 `.env` 至少包含：

```env
WANDB_API_KEY=...
WANDB_PROJECT=flower
WANDB_ENTITY=...
WANDB_MODE=online
WANDB_RUN_PREFIX=exp
```

如果只是本地调试，可以临时覆盖：

```bash
WANDB_MODE=offline ...
```

## 0. 当前关键结论

- Qwen Planner SFT 已切到 special-token 格式，不再使用旧 JSON 或 `<LATENT_STATE>`。
- Planner 输出格式为：

```text
<think>CoT</think><|latent_token|><|action_start|><|action_i|><|action_end|>
```

- EB-Nav joint training 当前应优先使用：

```bash
pipeline.train.qwen_planner_lora.response_mode=dataset
```

这会直接用 EB-Nav 原始 `reasoning_and_reflection + expert action_id` 构造逐帧 special-token response。它不是 anchor，也不依赖在线 `generate` 是否稳定。

- 不建议现在用 `response_mode=generate` 做训练。若生成结果缺少 `<|latent_token|>` 或 `<|action_start|>`，会直接报 `token id ... not found`。
- EB-Nav dataset 已修正为使用 `input_image_path` 作为 state 图像。`executable_plan[0].img_path` 是动作后的图像，不能再作为当前 state，否则会和 action 错位一拍。
- 使用修正前 EB-Nav 数据训练出的 checkpoint 不可靠，建议重新训练 EB-Nav joint checkpoint。

## 1. 准备 EB-Nav Phase2 数据

Smoke test：

```bash
uv run python -m src.train.prepare_eb_nav_phase2 \
  --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
  --images-base-dir datasets/EB-Nav \
  --sft-output /tmp/phase2_qwen_planner_sft.jsonl \
  --reward-output /tmp/phase2_reward_cache.jsonl \
  --stats-output /tmp/phase2_reward_stats.json \
  --limit-episodes 5
```

完整生成：

```bash
uv run python -m src.train.prepare_eb_nav_phase2 \
  --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
  --images-base-dir datasets/EB-Nav \
  --sft-output datasets/EB-Nav/phase2_qwen_planner_sft.jsonl \
  --reward-output datasets/EB-Nav/phase2_reward_cache.jsonl \
  --stats-output datasets/EB-Nav/phase2_reward_stats.json
```

检查点：

- `prompt == episode["input"]`
- `response` 严格是 special-token 格式
- 不应出现旧 JSON 字段、`<LATENT_STATE>`、`planner_trigger`
- `phase2_reward_stats.json` 里 reward 统计非空

## 2. 训练或验证 Qwen Planner LoRA

如果已有 `models/qwen_planner_lora`，先验证：

```bash
uv run python -m src.train.validate_qwen_planner_lora \
  --sft-jsonl datasets/EB-Nav/phase2_qwen_planner_sft.jsonl \
  --adapter-path models/qwen_planner_lora \
  --limit 512 \
  --extract-latent
```

目标：

- `format_rate` 接近或高于 95%
- `latent_extract_rate` 正常
- `action_prior_top1` 高于 8 动作随机基线 12.5%

小样本 SFT smoke：

```bash
uv run python -m src.train.train_qwen_planner_lora \
  --sft-jsonl datasets/EB-Nav/phase2_qwen_planner_sft.jsonl \
  --output-dir models/qwen_planner_lora_smoke \
  --limit 32 \
  --epochs 1 \
  --batch-size 1 \
  --save-every-steps 20 \
  --validate-every-steps 20 \
  --validation-limit 8 \
  --validation-extract-latent
```

完整 SFT：

```bash
uv run python -m src.train.train_qwen_planner_lora \
  --sft-jsonl datasets/EB-Nav/phase2_qwen_planner_sft.jsonl \
  --output-dir models/qwen_planner_lora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 2e-5 \
  --checkpoint-dir models/qwen_planner_lora/checkpoints \
  --save-every-steps 500 \
  --validate-every-steps 500 \
  --validation-limit 64 \
  --validation-extract-latent
```

断点续训：

```bash
uv run python -m src.train.train_qwen_planner_lora \
  --sft-jsonl datasets/EB-Nav/phase2_qwen_planner_sft.jsonl \
  --output-dir models/qwen_planner_lora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 2e-5 \
  --checkpoint-dir models/qwen_planner_lora/checkpoints \
  --resume-from-checkpoint latest \
  --save-every-steps 500 \
  --validate-every-steps 500 \
  --validation-limit 64 \
  --validation-extract-latent
```

## 3. EB-Nav Joint Training

当前推荐先训练：Qwen visual encoder + WM。保持 LLM backbone 冻结，planner LoRA 冻结。

原因：

- `qwen_planner_lora.trainable=false`：加载 planner LoRA 定义 latent/action token 空间，但不继续改 language LoRA。
- `qwen_encoder.llm_backbone_trainable=false`：不训练 Qwen LLM backbone。
- `qwen_encoder.train_mode=full`：训练 Qwen visual encoder。
- `response_mode=dataset`：使用 EB-Nav 每步 CoT/action 构造 response，避免 `generate` 格式不稳定。

Smoke test：

```bash
uv run python -m src.train.train_wm_joint \
  wm=lewm_qwen_llm_joint \
  pipeline.train.dataset_source=eb_nav \
  pipeline.train.eb_nav.dataset_path=datasets/EB-Nav/eb-nav_dataset_single_step.json \
  pipeline.train.eb_nav.images_base_dir=datasets/EB-Nav \
  pipeline.train.eb_nav.reward_cache_path=datasets/EB-Nav/phase2_reward_cache.jsonl \
  pipeline.train.training_mode=fully_supervised \
  pipeline.train.temporal_stride=1 \
  pipeline.train.qwen_planner_lora.enabled=true \
  pipeline.train.qwen_planner_lora.checkpoint_path=models/qwen_planner_lora \
  pipeline.train.qwen_planner_lora.trainable=false \
  pipeline.train.qwen_planner_lora.response_mode=dataset \
  pipeline.train.qwen_encoder.train_mode=full \
  pipeline.train.qwen_encoder.llm_backbone_trainable=false \
  pipeline.train.qwen_encoder.lr=1e-6 \
  pipeline.train.qwen_encoder.detach_target_latents=true \
  pipeline.train.qwen_encoder.dtype=bfloat16 \
  pipeline.train.qwen_encoder.encode_micro_batch_size=1 \
  pipeline.train.qwen_encoder.gradient_checkpointing=true \
  pipeline.train.qwen_encoder.gradient_checkpointing_use_reentrant=false \
  wm.lewm.reward.enabled=false \
  wm.lewm.perceptual.enabled=false \
  pipeline.train.max_samples=8 \
  pipeline.train.test_max_samples=0 \
  pipeline.train.test_every_n_epochs=0 \
  pipeline.train.batch_size=1 \
  pipeline.train.epochs=1 \
  pipeline.train.save_every_steps=20 \
  pipeline.train.post_visualization_enabled=false
```

完整单卡训练：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m src.train.train_wm_joint \
  wm=lewm_qwen_llm_joint \
  pipeline.train.dataset_source=eb_nav \
  pipeline.train.eb_nav.dataset_path=datasets/EB-Nav/eb-nav_dataset_single_step.json \
  pipeline.train.eb_nav.images_base_dir=datasets/EB-Nav \
  pipeline.train.eb_nav.reward_cache_path=datasets/EB-Nav/phase2_reward_cache.jsonl \
  pipeline.train.training_mode=fully_supervised \
  pipeline.train.temporal_stride=1 \
  pipeline.train.qwen_planner_lora.enabled=true \
  pipeline.train.qwen_planner_lora.checkpoint_path=models/qwen_planner_lora \
  pipeline.train.qwen_planner_lora.trainable=false \
  pipeline.train.qwen_planner_lora.response_mode=dataset \
  pipeline.train.qwen_encoder.train_mode=full \
  pipeline.train.qwen_encoder.llm_backbone_trainable=false \
  pipeline.train.qwen_encoder.lr=1e-6 \
  pipeline.train.qwen_encoder.detach_target_latents=true \
  pipeline.train.qwen_encoder.dtype=bfloat16 \
  pipeline.train.qwen_encoder.encode_micro_batch_size=1 \
  pipeline.train.qwen_encoder.gradient_checkpointing=true \
  pipeline.train.qwen_encoder.gradient_checkpointing_use_reentrant=false \
  wm.lewm.reward.enabled=false \
  wm.lewm.perceptual.enabled=false \
  pipeline.train.max_samples=0 \
  pipeline.train.test_max_samples=0 \
  pipeline.train.test_every_n_epochs=0 \
  pipeline.train.batch_size=1 \
  pipeline.train.epochs=4 \
  pipeline.train.save_every_steps=50 \
  pipeline.train.post_visualization_enabled=false
```

断点续训：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m src.train.train_wm_joint \
  wm=lewm_qwen_llm_joint \
  pipeline.train.dataset_source=eb_nav \
  pipeline.train.eb_nav.dataset_path=datasets/EB-Nav/eb-nav_dataset_single_step.json \
  pipeline.train.eb_nav.images_base_dir=datasets/EB-Nav \
  pipeline.train.eb_nav.reward_cache_path=datasets/EB-Nav/phase2_reward_cache.jsonl \
  pipeline.train.training_mode=fully_supervised \
  pipeline.train.temporal_stride=1 \
  pipeline.train.qwen_planner_lora.enabled=true \
  pipeline.train.qwen_planner_lora.checkpoint_path=models/qwen_planner_lora \
  pipeline.train.qwen_planner_lora.trainable=false \
  pipeline.train.qwen_planner_lora.response_mode=dataset \
  pipeline.train.qwen_encoder.train_mode=full \
  pipeline.train.qwen_encoder.llm_backbone_trainable=false \
  pipeline.train.qwen_encoder.lr=1e-6 \
  pipeline.train.qwen_encoder.detach_target_latents=true \
  pipeline.train.qwen_encoder.dtype=bfloat16 \
  pipeline.train.qwen_encoder.encode_micro_batch_size=1 \
  pipeline.train.qwen_encoder.gradient_checkpointing=true \
  pipeline.train.qwen_encoder.gradient_checkpointing_use_reentrant=false \
  wm.lewm.reward.enabled=false \
  wm.lewm.perceptual.enabled=false \
  pipeline.train.resume_from_checkpoint=latest \
  pipeline.train.batch_size=1 \
  pipeline.train.epochs=4 \
  pipeline.train.save_every_steps=50 \
  pipeline.train.post_visualization_enabled=false
```

## 4. 多卡训练

当前代码支持 `torchrun` DDP。注意：

- 不要设置 `NCCL_TOPO_FILE=/mnt/topology/hwloc.xml`。`hwloc.xml` 不是 NCCL native topo XML。
- 如果集群默认设置了坏的 `NCCL_TOPO_FILE`，用 `env -u NCCL_TOPO_FILE`。
- 如果 NCCL topology 仍有问题，先加 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。
- `batch_size` 是每张 GPU 的 batch size。
- DDP + Qwen gradient checkpointing 必须使用 `gradient_checkpointing_use_reentrant=false`，否则可能报 `Expected to mark a variable ready only once`。

示例：

```bash
env -u NCCL_TOPO_FILE \
CUDA_VISIBLE_DEVICES=0,1,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --nproc_per_node=3 -m src.train.train_wm_joint \
  wm=lewm_qwen_llm_joint \
  pipeline.train.dataset_source=eb_nav \
  pipeline.train.eb_nav.dataset_path=datasets/EB-Nav/eb-nav_dataset_single_step.json \
  pipeline.train.eb_nav.images_base_dir=datasets/EB-Nav \
  pipeline.train.eb_nav.reward_cache_path=datasets/EB-Nav/phase2_reward_cache.jsonl \
  pipeline.train.training_mode=fully_supervised \
  pipeline.train.temporal_stride=1 \
  pipeline.train.qwen_planner_lora.enabled=true \
  pipeline.train.qwen_planner_lora.checkpoint_path=models/qwen_planner_lora \
  pipeline.train.qwen_planner_lora.trainable=false \
  pipeline.train.qwen_planner_lora.response_mode=dataset \
  pipeline.train.qwen_encoder.train_mode=full \
  pipeline.train.qwen_encoder.llm_backbone_trainable=false \
  pipeline.train.qwen_encoder.lr=1e-6 \
  pipeline.train.qwen_encoder.detach_target_latents=true \
  pipeline.train.qwen_encoder.dtype=bfloat16 \
  pipeline.train.qwen_encoder.encode_micro_batch_size=1 \
  pipeline.train.qwen_encoder.gradient_checkpointing=true \
  pipeline.train.qwen_encoder.gradient_checkpointing_use_reentrant=false \
  pipeline.train.multi_gpu.find_unused_parameters=false \
  wm.lewm.reward.enabled=false \
  wm.lewm.perceptual.enabled=false \
  pipeline.train.batch_size=1 \
  pipeline.train.epochs=1 \
  pipeline.train.save_every_steps=50 \
  pipeline.train.post_visualization_enabled=false
```

## 5. 可选：打开 reward / perceptual

建议顺序：

1. 先跑通 `reward=false, perceptual=false`
2. 再打开 reward head：

```bash
wm.lewm.reward.enabled=true
```

3. 最后再打开 image decoder + perceptual：

```bash
wm.lewm.perceptual.enabled=true
```

如果打开 perceptual 后 OOM 或 `loss_image_recon` 过大，先关闭 perceptual，确认 latent dynamics 主链路正常。

## 6. Rollout 可视化并上传 W&B

脚本：

```bash
src.train.visualize_eb_nav_rollout
```

当前可视化脚本会：

- 从 `.env` 读取 W&B 配置
- 默认加载训练过的 Qwen vision encoder
- 先用 Qwen 编码 rollout latent 到 CPU
- 释放 Qwen
- 再加载 WM 做预测
- 默认使用低显存 planner extraction
- 默认使用 PCA 投影，并画 GT-Pred 连线
- 上传 scalar、media 和图片目录 artifact 到 W&B

短 rollout：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m src.train.visualize_eb_nav_rollout \
  --checkpoint latest \
  --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
  --images-base-dir datasets/EB-Nav \
  --reward-cache datasets/EB-Nav/phase2_reward_cache.jsonl \
  --planner-lora models/qwen_planner_lora \
  --num-rollouts 1 \
  --num-steps 8 \
  --encode-micro-batch-size 1 \
  --qwen-dtype bfloat16 \
  --projection pca
```

50 步 teacher-forced 可视化：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m src.train.visualize_eb_nav_rollout \
  --checkpoint latest \
  --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
  --images-base-dir datasets/EB-Nav \
  --reward-cache datasets/EB-Nav/phase2_reward_cache.jsonl \
  --planner-lora models/qwen_planner_lora \
  --num-rollouts 1 \
  --num-steps 50 \
  --encode-micro-batch-size 1 \
  --qwen-dtype bfloat16 \
  --projection pca
```

说明：

- 不传 `--temporal-stride` 时，脚本默认 `temporal_stride = num_steps`。
- 当前可视化是 teacher-forced one-step prediction over N steps，不是 open-loop 盲跑。
- 即每一步使用 GT state 更新 history window，再预测下一步。
- 如果要看 UMAP：

```bash
--projection umap
```

如果只想本地生成图，不上传 W&B：

```bash
--disable-wandb
```

只有在调试脚本链路时才使用：

```bash
--skip-vision-state
```

正常评估不要加它，否则不会加载训练过的 Qwen vision encoder。

**按离散动作分支的下一状态可视化**（每步对 `ACTION_MAP` 中多个候选动作分别 `predict_next`，PCA/UMAP 上色，GT 离散动作对应预测点用 `*`）：

```bash
--action-branching --branch-action-ids 0,1,2,3,4,5,6,7
```

输出文件名形如 `eb_nav_rollout_001_action_branch.png`。

## 7. 结果判断

训练时优先看：

- `loss_recon`
- `loss_sigreg`
- `loss_total_with_kl`
- 如果 reward 开启，再看 `loss_reward`
- 如果 perceptual 开启，再看 `loss_image_recon` 和 `loss_perceptual`

可视化时优先看：

- `rollout/mse`
- `rollout/mse_mean`
- PCA 图里的 GT-Pred 灰色连线长度

如果 PCA 里 Predicted 几乎是一个点：

1. 确认是否用的是修正 EB-Nav 时序后的新 checkpoint。
2. 旧 checkpoint 是用 `plan.img_path` 错位数据训练的，不要用来判断当前代码效果。
3. 确认训练命令是 `response_mode=dataset`。
4. 确认可视化没有加 `--skip-vision-state`。

## 8. 常见问题

### `token id ... not found`

通常来自 `response_mode=generate`。模型生成文本里没有 special token。

解决：训练和可视化都优先用：

```bash
pipeline.train.qwen_planner_lora.response_mode=dataset
```

### `NCCL XML Parse error`

不要把 hwloc XML 当成 NCCL topo：

```bash
env -u NCCL_TOPO_FILE ...
```

必要时加：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
```

### `Expected to mark a variable ready only once`

DDP + Qwen checkpointing 需要 non-reentrant：

```bash
pipeline.train.qwen_encoder.gradient_checkpointing=true
pipeline.train.qwen_encoder.gradient_checkpointing_use_reentrant=false
```

### 可视化 OOM

优先确认：

- `--encode-micro-batch-size 1`
- `--qwen-dtype bfloat16`
- 没有其他进程占用同一张 GPU
- 没有加 `--disable-low-memory-planner`

当前脚本已经避免 Qwen 和 WM 同时常驻 GPU。

### W&B run 是空的

当前脚本会写：

- `rollout/status`
- `rollout/mse`
- `rollout/mse_mean`
- `eb_nav_visualization/rollout`

如果仍为空，先检查 `.env` 里的：

```env
WANDB_MODE=online
WANDB_API_KEY=...
```

## 9. 之后可以再尝试

在 EB-Nav 修正时序后的 checkpoint 稳定后，再逐步尝试：

1. `wm.lewm.reward.enabled=true`
2. `wm.lewm.perceptual.enabled=true`
3. `pipeline.train.qwen_planner_lora.trainable=true`
4. `pipeline.train.qwen_encoder.llm_backbone_trainable=true`

每次只打开一个变量，观察 `loss_recon` 和 rollout PCA 是否变差。
