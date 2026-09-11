# SFT2：Query 与 DINO 空间特征对齐

入口为 `python -m nimloth.training.sft.stage2`。这一阶段优化回答 CE 和当前观测的 query/DINO 均方误差，与历史 WM/value 训练（现为 SFT3）不同。

```bash
python -m nimloth.training.sft.stage2 \
  --model <格式训练checkpoint> --train-jsonl <训练数据.jsonl> \
  --val-jsonl <验证数据.jsonl> --output-dir <输出目录> \
  --dino-cache-root <DINO缓存目录>
```

## 配置与训练参数

`--grid-size` 接受任意正整数，查询位置数 K 统一由 `grid_size²` 推导，因此 `--latent-token-count` 必须与它相等；默认 4×4 网格对应 K=16。独立 DINO 缓存构建、输入审计、容量门禁和训练必须传入相同的 grid size，不同 grid size 的缓存不能混用。损失权重由 `--weight-lm` 和 `--weight-dino` 指定，二者均须为正；`--projector-hidden-dim` 指定投影维度。YAML 可通过 `query_alignment` 提供这些参数，其余训练设置复用 SFT1。

默认全量训练 Qwen；选择 `--lora` 时沿用 adapter、embedding 和输出 head 的训练方式。共享 slot projector 始终加入优化器。
新建 projector 使用语言模型输入 embedding 的 dtype/device（BF16 模型不会新建 FP32 projector 参数）。多卡可指定 `--distributed-strategy fsdp`；语言模型和 projector 均参与分片、完整保存与恢复。

可使用 `--until-converged --convergence-min-epochs 2 --convergence-patience-epochs 2 --convergence-min-relative-improvement 0.01` 训练至收敛，不能同时指定固定 `--epochs`，也不能限制验证批次数。每轮以完整验证的加权总损失 `weight_lm * LM + weight_dino * DINO` 对比上一轮，连续两轮改善不足 1% 且达到最少轮数后停止；`best` 始终选择总损失最低的 checkpoint。预热后学习率保持不变，运行时限不代表收敛。
有限 GPU 检查可指定 `--max-optimizer-steps N`，在绝对第 N 步保存完整恢复 checkpoint 并以 75 退出，不声明收敛；正式续训移除该预算参数。

`validation_metrics.jsonl`、每轮日志与 W&B 分别记录未加权的 LM、DINO 分量和加权总损失。三者以回答为计数单位跨 batch、梯度累积和 rank 求和归约；分布式 sampler 的补齐项仍计入均值。旧 CSV 的 `val_loss` 在 query 阶段表示总损失。模型返回的分量已 detach，不改变反向传播目标。

## 模块职责与计算顺序

dataset 保持以完整轨迹为样本，因此 `--batch-size` 按轨迹计数，`--max-train-records` 也直接限制原始轨迹。collator 为每个回答记录 query 位置、回答 token 归属和当前观测，但不会复制回答前缀。每条轨迹只执行一次因果 teacher-forcing 前向。

`data.py` 保留完整多轮记录。每个回答之前的当前用户轮必须恰好对应一个观测图像，多图歧义会报错。所有真实回答都计算一次 CE；每个回答内部先对 token CE 求平均，再在回答之间等权平均。真实非空 CoT、有序连续 query 区间和观测图像必须逐回答对齐；缺失或截断回答、query 位置均拒绝，不生成替代思考内容。

`model.py` 在一次完整轨迹的 teacher-forcing Qwen 前向中，使用现有 final-norm hook 提取所有回答的 query hidden states。这些 query 位于各自真实 CoT 之后、动作之前，按位置经过 `wm.grid.SharedSlotProjector`。DINO MSE 先在每个回答的全部 K 个位置和特征维上平均，再在回答之间等权平均。DINO 目标无梯度，形状必须严格相同，不允许广播掩盖错配。

数据目标使用既有 `CachedDINOGridTargets`：训练/验证图像索引与 `dino_grid<N>` 附属缓存。backbone 加载器校验固定的 `DINOV2_LARGE_IDENTITY`、来源、图像对应关系、空间顺序和特征维度。此入口消费真实冻结 DINO 特征，不负责生成缓存。为保留观测路径，分词在线执行，不支持 SFT1 仅含 token 的 `--cache-only` / `--require-prebuilt-cache` 模式。

## 保存、交接与恢复

Checkpoint 保存 `training_stage=query`、语言模型或 adapter，以及 `slot_projector.pt` 和 `grid_state_config.json`。配置记录 teacher 身份、query token ID、projector 维度和目标权重；恢复或从 query checkpoint 初始化时先严格校验，再恢复 projector。

LoRA 合并导出保留 projector 文件及阶段元数据，SFT3 使用同一 projector 格式。完整恢复包含优化器、调度器、epoch/微批次游标、每 rank 随机数状态及收敛历史；query 收敛监控身份为 `validation_total_loss`。CPU 测试覆盖标签、梯度、空间对齐、收敛与导出接口，不作为真实 GPU 训练或 rollout 质量证据。
