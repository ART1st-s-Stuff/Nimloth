# SFT1：回答格式监督

入口为 `python -m nimloth.training.sft.stage1`。输入沿用格式训练 JSONL，必需参数为 `--model`、`--train-jsonl`、`--val-jsonl` 和 `--output-dir`；其余训练参数见 `--help`。支持既有 YAML 默认值、全量微调和 LoRA。外层 `experiments/training/sft1/train.py` 调用此实现。

## 模块职责与数据流

- `data.py`：读取记录中的对话和截图，构造仅监督回答的标签，屏蔽提示词和填充，stage1移除所有角色文本中的历史 latent 标记（原始 JSONL/截图不变），保留真实 CoT 和动作；使用右侧填充保留文本区间的位置关系，并负责样本编码缓存。
- `config.py`：读取 YAML 默认配置。`cli.py`：定义命令行选项，在加载模型前校验训练阶段和参数。
- `trainer.py`：加载 Qwen、设置可训练参数、构建优化器，驱动梯度累积、离线验证和 epoch checkpoint 保存。SFT1直接使用 teacher forcing 的回答 CE，不计算 DINO 或 WM 损失。SFT2显式选择 query 阶段后复用同一训练生命周期。
- `loss.py`：`--action-token-loss-weight`（YAML `train.action_token_loss_weight`）为动作起止及八个动作 token 加权，其他有效回答 token（含 EOS）权重 1；按每微批次权重和归一化，保持梯度累积方式。默认 1 保留原 loss 路径；stage2 拒绝大于 1。验证与收敛仍使用未加权 LM loss，缓存不因权重改变而重建；checkpoint 身份包含权重，改变权重不可原样恢复。
- `convergence.py`：验证 loss 收敛状态和可恢复的停止策略。
- `distributed.py`：建立和清理分布式进程组，提供主进程判断与同步；checkpoint 模块不依赖训练循环。
- `checkpoint.py`：保存训练状态、查找恢复位置和校验阶段身份，独立于训练循环。
- `checkpoint_export.py`：负责 LoRA 合并与导出校验，包括单独训练的 embedding 和输出 head；历史合并脚本调用此实现。

## 保存与恢复

Stage1 不接受 K、query mode、query mask 的 CLI、环境变量或 YAML 配置。共享内部接口用 None 标记无 query 的阶段；stage2 仍要求正数 K。

Cache 使用 `nimloth_early_stage_cache_v7`，记录 `format_answer_ce_v2` 与 `remove_latent_markers_all_roles` 投影身份；旧缓存或无身份 tensor 不能静默复用。格式指标检查模型生成的 CoT 与动作块，不要求 latent 块。

Checkpoint 保存 `training_stage=format`、`format_objective=format_answer_ce_v2`，query 参数为空。旧 query 训练 checkpoint 不可恢复为新格式阶段。完整优化步 checkpoint 以 COMMITTED 标记发布，保存优化器、调度器、各 rank RNG 和数据位置；恢复校验完整身份。离线 loss/格式验证不等于环境 rollout；当前共享 direct evaluator 仍要求 query 协议，尚不能用它验收此 format-only 产物。

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
