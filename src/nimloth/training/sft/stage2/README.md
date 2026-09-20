# SFT2：Query 与 DINO 空间特征对齐

入口为 `python -m nimloth.training.sft.stage2`。这一阶段优化回答 CE 和当前观测的 query/DINO 均方误差，与历史 WM/value 训练（现为 SFT3）不同。

```bash
python -m nimloth.training.sft.stage2 \
  --model <格式训练checkpoint> --train-jsonl <训练数据.jsonl> \
  --val-jsonl <验证数据.jsonl> --output-dir <输出目录> \
  --dino-cache-root <DINO缓存目录>
```

## 配置与训练参数

`--grid-size` 接受任意正整数。普通训练的查询位置数 K 为 `grid_size²`，因此 `--latent-token-count` 必须与它相等；默认 4×4 网格对应 K=16。独立 DINO 缓存构建、输入审计、容量门禁和训练必须传入相同的 grid size，不同 grid size 的缓存不能混用。损失权重由 `--weight-lm` 和 `--weight-dino` 指定，二者均须为正；`--projector-hidden-dim` 指定投影维度。YAML 可通过 `query_alignment` 提供这些参数，其余训练设置复用 SFT1。

从既有 K64 checkpoint 增补 DINO CLS 的评估训练必须同时指定
`--tuning-mode global_query_only --include-global-token --evaluation-only`，并使用
`build_dino_cache.py --include-cls --build-commit <40-hex>` 生成的 v2 cache。
该模式要求 `--distributed-strategy ddp --embedding-master-dtype bfloat16`：冻结的
dense embedding/LM head 保持 BF16，只有新增 input row 使用 FP32 master；避免把整张
词表为了一个新行提升到 FP32。收敛参数固定为至少2轮、patience 2、相对改善1%。
该模式把新全局 Query 放在已有空间 Query 后，只保留它的 input embedding
FP32 master 可训练；旧 K64、backbone、LM head 和共享 projector 全部冻结。
新行用旧空间 Query input rows 的 FP32 均值初始化，checkpoint 另存精确的
`selected_token_rows.pt` 以避免恢复时 BF16 舍入。空间与 CLS DINO MSE 分别
归一化再相加，验证日志分别记录两项。输出 metadata 标记
`evaluation_only=true`、`formal_stage2=false`、父 checkpoint、v2 cache identity
和显式 K64+CLS layout；它不能作为正式 Stage2 结果。正式训练可通过普通
Stage2 路径从 epoch1 设置 `include_global_token`，无需依赖 epoch16 扩展逻辑。

冻结表示的诊断续训使用 `--tuning-mode query_projector_only --include-global-token
--evaluation-only`。它从已有 K65 checkpoint 初始化权重，但因可训练参数集合改变而使用
新的 optimizer：全部65个 Query 只训练 input embedding FP32 master rows，共享 projector
以 FP32 master 训练；Qwen、vision、LM head和protocol rows冻结。若初始化checkpoint带有
旧的 input-only `selected_token_rows.pt`，按token ID严格恢复其中的精确行，其他 Query行
使用该checkpoint实际导出的dense权重。收敛监控为完整验证集的 spatial+CLS DINO分项之
和；LM、格式和direct success是保持门禁。该模式只用于 evaluation-only诊断，不构成正式
Stage2，也不允许加载旧单行optimizer冒充原样续训。

默认 `--tuning-mode selected_lora` 保留 LoRA 和选定 token 行训练。显式 `--tuning-mode full_language` 训练语言 transformer、完整 embedding/LM head 和共享 slot projector；整个 Qwen visual（包含原生 merger）冻结，DINO 目标保持固定。该模式要求 FSDP、独立 embedding/head，所有可训练参数保留 FP32 master、前向 BF16。`full_tuning.py` 定义冻结范围并打印参数数量。全量模式通过 `--lr`、`--embedding-lr`、`--projector-lr` 配置三组学习率（本次实验均为 2e-5）；不使用 query/protocol 行优化器。恢复身份区分全量和选行模式，完整 checkpoint 以 dense 权重保存，FP32 加载避免恢复时舍入。
新建 projector 使用语言模型输入 embedding 的 dtype/device（BF16 模型不会新建 FP32 projector 参数）。多卡可指定 `--distributed-strategy fsdp`；语言模型和 projector 均参与分片、完整保存与恢复。

可使用 `--until-converged --convergence-min-epochs 2 --convergence-patience-epochs 2 --convergence-min-relative-improvement 0.01` 训练至收敛，不能同时指定固定 `--epochs`，也不能限制验证批次数。每轮以完整验证的加权总损失 `weight_lm * LM + weight_dino * DINO` 对比上一轮，连续两轮改善不足 1% 且达到最少轮数后停止；`best` 始终选择总损失最低的 checkpoint。预热后学习率保持不变，运行时限不代表收敛。
收敛运行如需控制磁盘，可显式使用 `--keep-epoch-checkpoints 1`，仅保留最新已提交 epoch
与独立 `best`；该选项默认不启用，且不会删除无法完整核验或属于其他训练身份的目录。
evaluation-only `global_query_only` 是例外：它按独立的验证 CLS DINO MSE
应用同一 patience/1% 规则并选择 best，同时继续记录总损失、空间 DINO、LM、格式与
rollout 门禁，避免空间常量项掩盖新增全局 Query 的收敛。
`query_projector_only` 则按 `validation_dino_loss`（分别归一化后的 spatial 与 CLS之和）
应用同一规则。
有限 GPU 检查可指定 `--max-optimizer-steps N`，在绝对第 N 步保存完整恢复 checkpoint 并以 75 退出，不声明收敛；正式续训移除该预算参数。

`validation_metrics.jsonl`、每轮日志与 W&B 分别记录未加权的 LM、DINO 分量和加权总损失。LM 只以成功轨迹的回答计数，DINO 以全部回答计数，分别跨 batch、梯度累积和 rank 求和归约；分布式 sampler 的补齐项仍计入均值。旧 CSV 的 `val_loss` 在 query 阶段表示总损失。模型返回的均值指标已 detach；两个可微分量之和分别用于各自分母的反向传播。

## 模块职责与计算顺序

入口在初始化 distributed/CUDA、processor 和模型前，先在 CPU 上逐条预检实际会读取的
train/validation JSONL。Stage2 训练只接受带顶层 `messages`、字符串 `id`、布尔
`success` 及逐回答观测/非空 CoT 的 answer-view 数据；传入
`record_format=nimloth_trajectory_v1` 的原始 trajectory 会立即报出 split、record index
和文件路径；缺少 CoT 后 Query state 到 `action_start` 边界的数据也会拒绝，不会等到首个
DataLoader batch 才失败。`--max-train-records` 与
`--max-val-records` 同样约束预检范围。

dataset 保持以完整轨迹为样本，因此 `--batch-size` 按轨迹计数，`--max-train-records` 也直接限制原始轨迹。collator 为每个回答记录 query 位置、回答 token 归属和当前观测，但不会复制回答前缀。每条轨迹只执行一次因果 teacher-forcing 前向。

`data.py` 保留完整多轮记录。每个回答之前的当前用户轮必须恰好对应一个观测图像，多图歧义会报错。每条记录必须显式提供完整轨迹的布尔 `success`。只有成功轨迹的回答计算 LM 监督；失败回答仍作为真实因果上下文参与前向和 DINO 对齐。每个回答内部先对 token CE 求平均，再在成功回答之间等权平均；全失败更新组 LM 为图连接的零。真实非空 CoT、有序连续 query 区间和观测图像必须逐回答对齐；缺失或截断回答、query 位置均拒绝，不生成替代思考内容。

`model.py` 在一次完整轨迹的 teacher-forcing Qwen 前向中，使用现有 final-norm hook 提取所有回答的 query hidden states。这些 query 位于各自真实 CoT 之后、动作之前，按位置经过 `wm.grid.SharedSlotProjector`。DINO MSE 先在每个回答的全部 K 个位置和特征维上平均，再在回答之间等权平均。DINO 目标无梯度，形状必须严格相同，不允许广播掩盖错配。

数据目标使用既有 `CachedDINOGridTargets`：训练/验证图像索引与 `dino_grid<N>` 附属缓存。backbone 加载器校验固定的 `DINOV2_LARGE_IDENTITY`、来源、图像对应关系、空间顺序和特征维度。此入口消费真实冻结 DINO 特征，不负责生成缓存。为保留观测路径，分词在线执行，不支持 SFT1 仅含 token 的 `--cache-only` / `--require-prebuilt-cache` 模式。

## 保存、交接与恢复

Checkpoint 保存 `training_stage=query`、语言模型或 adapter，以及 `slot_projector.pt` 和 `grid_state_config.json`。配置记录 teacher 身份、query token ID、projector 维度和目标权重；恢复或从 query checkpoint 初始化时先严格校验，再恢复 projector。

### K64 epoch16 到 K65 双 projector 迁移

评估用迁移必须显式选择 `--tuning-mode split_projector_migration`，并令
`--model` 与 `--stage2-k64-migration-checkpoint` 都指向同一个 K64 Stage2 epoch16。
它是模型权重 lineage continuation，不是 optimizer resume：只追加一个 CLS Query，
使用新的 AdamW，从零开始记录 epoch/step、scheduler、数据游标和收敛历史。

源 `slot_projector.pt` 必须是严格 K64 shared projector。目标使用互不共享的
`spatial` 与 `global` 分支，两者逐值复制源权重初始化；训练时 optimizer 参数组固定为
`state_proj_spatial`、`state_proj_global`、`query_rows`。前两组使用 projector LR，
全部65个 input Query FP32 master rows使用 Query LR；Qwen、vision、LM head及
action/format/protocol rows冻结。源若有双表 `selected_token_rows.pt` 则以它为准；历史
epoch16没有该 sidecar时，只允许从 untied、完整 FP32 dense input/output tables提取64个
Query与protocol rows，非FP32源直接拒绝。新增 CLS input/output row分别由对应64行FP32
均值初始化，output row保持冻结。

checkpoint写入 `split_spatial_global_v1`、源路径及 projector/config hash、迁移规则、
K65 cache identity、`evaluation_only=true` 与 `fresh_adamw_v1`。普通 `--resume` 只能恢复
同一 split schema和完整迁移 provenance；shared K64/K65、缺失来源或cache/layout变化均
拒绝。该模式不读取 Stage3 checkpoint，也不允许 `--continue-from-epoch`。

LoRA 合并导出保留 projector 文件及阶段元数据，SFT3 使用同一 projector 格式。完整恢复包含优化器、调度器、epoch/微批次游标、每 rank 随机数状态及收敛历史；普通 query 收敛监控身份为 `validation_total_loss`，evaluation-only CLS alignment 为 `validation_dino_cls_loss`。CPU 测试覆盖标签、梯度、空间对齐、收敛与导出接口，不作为真实 GPU 训练或 rollout 质量证据。

从已提交 epoch 提高 projector 学习率时，必须同时使用
`--continue-with-projector-lr-change`。该显式门禁仅允许
`projector_lr` 改变：恢复既有优化器 moments、每 rank RNG、数据游标和完整验证历史，
然后按命令行给出的五组学习率创建新的续训 schedule；普通 resume 仍禁止改变学习率。

从已提交 epoch 调整 Query token 学习率时，必须使用
`--continue-with-query-token-lr-change --continue-from-epoch <epoch目录>`。该门禁只允许
checkpoint identity 中的 `token_row_training.query_token_lr` 改变；
包括 `protocol_token_lr`、收敛规则和 warmup 在内的其他身份字段必须保持不变。续训恢复原
optimizer moments、每 rank RNG、epoch 数据边界和完整收敛历史，再用新 Query LR 及其余
原配置学习率重启 schedule。输出 `continuation.json` 记录父 checkpoint hash、旧/新 Query
LR 与新 schedule identity。

优化器更新前只预取一个累积组的 CPU 输入并统计成功/全部回答数，跨 rank 求和后逐微批单次前向。两个损失分别按各自全局分母缩放；恢复身份包含此监督语义，旧的全部回答 LM 优化器状态不能续训为新目标。

## 数据加载与计算开销

Stage2在线图像处理使用配置的 `num_workers`/预取参数；多进程采用spawn，避免从已初始化CUDA的父进程fork。DINO缓存先由父进程完整验证，再以文件引用传给worker并重新mmap，不复制整份特征张量到共享内存；文件身份变化拒绝加载。失败回答在CE计算前排除，仍保留完整因果前向、DINO监督和LM head的零梯度连接。当前仍生成完整词表logits，不能将此优化描述为消除了全部失败轨迹LM投影开销。


`build_dino_cache.py` also indexes every T+1 real image of structured trajectories,
including their terminal observation; answer-view datasets retain current-answer indexing.
Empty splits, mixed record formats within a split, and missing terminal/image paths are rejected.
`--reuse-cache` accepts only completed standalone `dino_grid_images_v1` caches with validated
teacher/grid identity, source JSONL hashes, image byte hashes, and shard hashes. It copies
matching image rows into a new immutable output and invokes the teacher only for missing images.
If every image is reused, no teacher is loaded and `teacher_provenance` is null; the source
cache path/fingerprint and reused image count remain in the new manifest's `reuse` lineage.
Legacy path-only caches are not sufficient evidence for byte-exact feature reuse.
