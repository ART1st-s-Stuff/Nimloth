# SFT2：Query 与 DINO 空间特征对齐

入口为 `python -m nimloth.training.sft.stage2`。这一阶段优化回答 CE 和当前观测的 query/DINO 均方误差，与历史 WM/value 训练（现为 SFT3）不同。

```bash
python -m nimloth.training.sft.stage2 \
  --model <格式训练checkpoint> --train-jsonl <训练数据.jsonl> \
  --val-jsonl <验证数据.jsonl> --output-dir <输出目录> \
  --dino-cache-root <DINO缓存目录>
```

## 配置与训练参数

`--latent-token-count` 必须等于 `--grid-size` 的平方，默认对应 4×4 网格的 16 个位置。损失权重由 `--weight-lm` 和 `--weight-dino` 指定，二者均须为正；`--projector-hidden-dim` 指定投影维度。YAML 可通过 `query_alignment` 提供这些参数，其余训练设置复用 SFT1。

默认全量训练 Qwen；选择 `--lora` 时沿用 adapter、embedding 和输出 head 的训练方式。共享 slot projector 始终加入优化器。
新建 projector 使用语言模型输入 embedding 的 dtype/device（BF16 模型不会新建 FP32 projector 参数）。多卡可指定 `--distributed-strategy fsdp`；语言模型和 projector 均参与分片、完整保存与恢复。

可使用 `--until-converged --convergence-min-epochs 2 --convergence-patience-epochs 2 --convergence-min-relative-improvement 0.01` 训练至收敛，不能同时指定固定 `--epochs`，也不能限制验证批次数。每轮以完整验证的加权总损失 `weight_lm * LM + weight_dino * DINO` 对比上一轮，连续两轮改善不足 1% 且达到最少轮数后停止；`best` 始终选择总损失最低的 checkpoint。预热后学习率保持不变，运行时限不代表收敛。
有限 GPU 检查可指定 `--max-optimizer-steps N`，在绝对第 N 步保存完整恢复 checkpoint 并以 75 退出，不声明收敛；正式续训移除该预算参数。

`validation_metrics.jsonl`、每轮日志与 W&B 分别记录未加权的 LM、DINO 分量和加权总损失。三者采用同一验证批次均值及跨 rank 求和/计数归约；分布式 sampler 的补齐项仍计入均值。旧 CSV 的 `val_loss` 在 query 阶段表示总损失。模型返回的分量已 detach，不改变反向传播目标。

## 模块职责与计算顺序

`AnswerPrefixDataset` 在采样之前为每个回答建立索引，因此 `--batch-size` 按回答计数，每轮覆盖全部回答。每个样本保留截至目标回答的完整历史；collator 只监督最后一个回答，不再把一条长轨迹的全部前缀塞进同一个 batch。`--max-train-records` 仍先按原始轨迹选择，随后展开其全部回答。

`data.py` 将多轮记录展开为截至每个真实回答的前缀。当前用户轮必须恰好对应一个观测图像，多图歧义会报错。历史轮仅提供上下文，不计算其回答 CE。当前真实非空 CoT、有序连续 query 区间和观测图像必须对齐；缺失或截断回答、query 位置均拒绝，不生成替代思考内容。

`model.py` 在一次 teacher-forcing Qwen 前向中，使用现有 final-norm hook 提取 query hidden states。这些 query 位于真实 CoT 之后、动作之前，按位置经过 `wm.grid.SharedSlotProjector`。CE 监督当前回答，MSE 监督全部投影后的位置。DINO 目标无梯度，形状必须严格相同，不允许广播掩盖错配。

数据目标使用既有 `CachedDINOGridTargets`：训练/验证图像索引与 `dino_grid<N>` 附属缓存。backbone 加载器校验固定的 `DINOV2_LARGE_IDENTITY`、来源、图像对应关系、空间顺序和特征维度。此入口消费真实冻结 DINO 特征，不负责生成缓存。为保留观测路径，分词在线执行，不支持 SFT1 仅含 token 的 `--cache-only` / `--require-prebuilt-cache` 模式。

## 保存、交接与恢复

Checkpoint 保存 `training_stage=query`、语言模型或 adapter，以及 `slot_projector.pt` 和 `grid_state_config.json`。配置记录 teacher 身份、query token ID、projector 维度和目标权重；恢复或从 query checkpoint 初始化时先严格校验，再恢复 projector。

LoRA 合并导出保留 projector 文件及阶段元数据，SFT3 使用同一 projector 格式。完整恢复包含优化器、调度器、epoch/微批次游标、每 rank 随机数状态及收敛历史；query 收敛监控身份为 `validation_total_loss`。CPU 测试覆盖标签、梯度、空间对齐、收敛与导出接口，不作为真实 GPU 训练或 rollout 质量证据。

`eval.py:evaluate(EvaluationConfig)` 为本阶段的真实环境 success rate 接口，统一由
`python -m nimloth.training.sft.evaluation --stage stage2 ...` 调用；与离线 loss validation
分开，不加载 WM/value/MCTS。完整参数、导出前置条件和恢复合同见上层 evaluation/README.md。
