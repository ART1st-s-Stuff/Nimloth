# SFT1：回答格式监督

标准流程调用 `python -m nimloth.training.sft.stage1.workflow --source-model MODEL --train-jsonl TRAIN_ALL --val-jsonl HELDOUT_ALL --format-eval-jsonl HELDOUT_ALL --output-dir NEW_RUN --config configs/training/sft1/format.yaml --nproc-per-node 8 --success-only`。workflow 从前两个输入筛选成功轨迹供 LM 训练和验证，第三个输入保留完整 heldout prompt，只做每轮自由生成格式检查。追加训练器 CLI 参数覆盖默认配置；续训传相同参数并追加 `--resume`。运行时继承调用者的解释器、GPU 和依赖环境，不写死机器路径。

`preparation.py` 要求每条记录提供可核验的 `source_identity`；train/val 另外要求布尔 `success` 并仅保留 `success is True`。成功 train/val 的每条轨迹必须提供合法且非空的 `action_indices`，但成功子集不要求覆盖全部八类动作；完整 format-eval 不筛 success，也不要求失败轨迹具有参考动作。准备清单记录 trajectory/record 数、assistant turn 数、成功子集实际动作分布（包括零计数）、source identities、筛除数及输入输出 hash。workflow 同时按 record ID、`(eval_set, seed)` 和 seed 核验 train/heldout 隔离，缺失身份时停止，不自行构造。源 JSONL 不改写。每轮格式检查固定取完整 format-eval 文件按记录顺序的前 32 条；逐条保存 record ID、token IDs、`raw_response`、`parsed_body`、实际分母、终止/解析原因及源 hash。`initialization.py` 以 BF16 创建独立 base：两个动作边界从 EOS 初始化，八个动作从 forward/backward/right/left/rotate right/rotate left/up/down 原始词向量均值初始化，分别处理 input/head。全 tensor 验证后发布。`workflow.py` 顺序执行准备、CPU cache，再启动多卡训练；恢复复用准备好的 base/data/cache 和训练器完整状态。底层训练 rank 退出 75 表示已保存边界后暂停，不是收敛；torchrun 可能将它映射为启动器退出 1，workflow 原样返回启动器状态，不自动重试。续训前检查完整、已提交的 resume checkpoint，再以 `--resume` 启动。workflow 的 SIGTERM/中断会停止所属进程组，不承诺保存新的边界；它不提供定时分段或 SIGUSR1 转发，恢复只能使用已有完整 checkpoint。

标准配置 action 权重 8，完整验证 LM loss 连续两轮相对改善不足 1% 且至少完成两轮才收敛。每十步保存，完整 epoch 发布后清理其覆盖的中间 step checkpoint，保留 epoch/best/final。配置与 CLI 可以显式覆盖。

每轮格式检查的标准 `train.format_eval_batch_size` 为 4，也可通过
`--format-eval-batch-size` 显式覆盖为其他正整数。训练器保持前 32 条记录的原始
顺序，将它们分批送入同一次 generation model 上下文；FSDP 下所有 rank 仍处理
相同批次并使用同步停止。批量 decoder-only 生成期间 tokenizer 临时切换为左侧
padding，结束或异常后恢复训练使用的右侧 padding。每条结果从批内统一 prompt
宽度之后切出；若首个 EOS 后只有生成器补入的 pad token，则移除这些补齐并保留
EOS。EOS 后存在非 padding 内容时不得裁掉，必须交给严格 validator 判为
`content_after_eos`；缺少 EOS 和达到长度上限仍按原合同失败。

epoch 末尾先输出 `validation_complete`，包含 `epoch`、`global_step` 和
`validation_seconds`。每个生成批次完成后输出 `format_eval_batch`，包含 `batch`、
`batches`、`completed_samples`、`total_samples` 和累计 `elapsed_seconds`。
`validation_metrics.jsonl` 的最终记录分别保存 `validation_seconds` 与
`format_eval_seconds`；启用 W&B 时同名耗时也记录在 `val/` 命名空间。这些字段只
描述运行耗时和进度，不改变验证 loss、格式判断或收敛状态。

底层 `python -m nimloth.training.sft.stage1` 是已准备数据/base 的训练器接口，供 workflow 和集成测试使用。

## 模块职责与数据流

- `data.py`：读取记录中的对话和截图，构造仅监督回答的标签，屏蔽提示词和填充，stage1移除所有角色文本中的历史 latent 标记（原始 JSONL/截图不变），保留真实 CoT 和动作；使用右侧填充保留文本区间的位置关系，并负责样本编码缓存。
- `config.py`：读取 YAML 默认配置。`cli.py`：定义命令行选项，在加载模型前校验训练阶段和参数。
- `trainer.py`：加载 Qwen、设置可训练参数、构建优化器，驱动梯度累积、离线验证和 epoch checkpoint 保存。SFT1直接使用 teacher forcing 的回答 CE，不计算 DINO 或 WM 损失。SFT2显式选择 query 阶段后复用同一训练生命周期。
- `loss.py`：`--action-token-loss-weight`（YAML `train.action_token_loss_weight`）仅为八个 `<|action_(i)|>` 动作编号 token 加权；动作起止、EOS 和其他有效回答 token 权重均为 1。loss 按每微批次有效权重和归一化，保持梯度累积方式。底层参数 1 保留未加权 loss 路径，标准 Stage 1 配置为 8；stage2 拒绝大于 1。验证与收敛仍使用未加权 LM loss，缓存不因权重改变而重建；checkpoint 身份同时包含权重和 `action_number_tokens_v1` 范围，旧的边界加权 checkpoint 不可恢复优化器或收敛历史。
- `convergence.py`：验证 loss 收敛状态和可恢复的停止策略。
- `distributed.py`：建立和清理分布式进程组，提供主进程判断与同步；checkpoint 模块不依赖训练循环。
- `checkpoint.py`：保存训练状态、查找恢复位置和校验阶段身份，独立于训练循环。
- `checkpoint_export.py`：负责 LoRA 合并与导出校验，包括单独训练的 embedding 和输出 head；历史合并脚本调用此实现。

## 保存与恢复

Stage1 不接受 K、query mode、query mask 的 CLI、环境变量或 YAML 配置。共享内部接口用 None 标记无 query 的阶段；stage2 仍要求正数 K。

Cache 使用 `nimloth_early_stage_cache_v7`，记录 `format_answer_ce_v2` 与 `remove_latent_markers_all_roles` 投影身份；旧缓存或无身份 tensor 不能静默复用。格式指标检查模型生成的 CoT 与动作块，不要求 latent 块。

Checkpoint 保存 `training_stage=format`、`format_objective=format_answer_ce_v2`、`action_token_loss_scope=action_number_tokens_v1`，query 参数为空。旧 query 训练 checkpoint 以及缺少当前 loss 范围的旧 Stage 1 checkpoint 不可恢复为当前格式阶段。完整优化步 checkpoint 以 COMMITTED 标记发布，保存优化器、调度器、各 rank RNG 和数据位置；恢复校验完整身份。训练期自由生成必须实际生成 EOS，EOS 后只能有 padding；移除终止表示后的正文交给正式 Stage 1 rollout 的严格全文 parser。缺 EOS、长度截断、尾随内容或重复动作块均失败，失败原因写入每轮 validation metrics。离线 loss/格式验证不等于环境 rollout；format-only 产物通过统一评估入口的 `--stage stage1` 执行真实环境验收。

## 训练至收敛

格式阶段可显式使用 `--until-converged --convergence-min-epochs N --convergence-patience-epochs P --convergence-min-relative-improvement R`，不能同时指定固定 `--epochs`。每轮完整验证后判断回答LM loss是否相对上一轮改善达到阈值；连续P轮未达到阈值且达到最少轮数后停止。绝对最低验证loss另外记录用于best checkpoint。

此模式按首轮预计优化步数和warmup_ratio确定预热步数，之后保持设定学习率，不设置虚构的总epoch数供cosine调度。Checkpoint保存收敛计数和每rank随机状态，续训恢复原有判断历史。运行时限或外部暂停不代表收敛；只有实际满足条件且final保存后才产生CONVERGED.json。

Stage1 and query stage2 optionally accept `--distributed-strategy fsdp` for multi-rank CUDA
FULL_SHARD training; the default remains DDP. Query checkpoints split the gathered
language-model state and projector state into the same portable artifacts as DDP,
without reading live sharded parameters from rank zero.
FSDP preserves original parameters and separates linear/embedding leaves to keep
PEFT FP32 trainable tensors distinct from BF16 frozen weights. Each microbatch
reduces sharded gradients; accumulation does not retain full replicated gradients.
Global gradient clipping includes every rank and supports mixed gradient dtypes.

Every rank participates in full CPU model/optimizer checkpoint collection, while
rank zero publishes ordinary PEFT artifacts plus complete training state. Resume
loads adapters before wrapping and converts the full named optimizer state into
local shards after wrapping. FSDP is part of the resume identity. Epoch, best,
final and optimizer-boundary saves use the same collective path. Format generation
runs the same prompts on all ranks with synchronized stopping; validation LM loss
and the convergence rule are unchanged.

The explicit GPU integration probe is
`torchrun --nproc_per_node=8 tests/integration/sft1_fsdp_roundtrip.py --output-dir UNIQUE_PATH`.
It compares uninterrupted and restored next updates exactly, including optimizer
state, and checks collective generation and epoch export. CPU tests alone do not
establish FSDP runtime, memory capacity, or model quality.

For bounded diagnostics, `--max-optimizer-steps N` pauses after absolute optimizer
step N, using the existing complete resume checkpoint path and exit code 75.
It does not complete an epoch or declare convergence, and does not change the
learning-rate schedule or objective identity. Resuming at or above N rejects
before restoring the optimizer; remove or raise the cap to continue training.

`eval.py:evaluate(EvaluationConfig)` 为本阶段的真实环境 success rate 接口，统一由
`python -m nimloth.training.sft.evaluation --stage stage1 --format-gate-jsonl FULL_HELDOUT ...`
调用。统一入口先以同一 `EarlyVLLMGenerator` 对固定 32 条执行严格原文门禁，至少
31/32 才开始环境 episode；失败保存证据并返回 2。门禁和真实环境 runner 共享
sampled-token EOS 校验。该过程与离线 loss validation 分开，不加载
WM/value/MCTS。完整参数、导出前置条件和恢复合同见上层 evaluation/README.md。
