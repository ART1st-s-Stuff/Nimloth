# SFT1：回答格式监督

入口为 `python -m nimloth.training.sft.stage1`。输入沿用格式训练 JSONL，必需参数为 `--model`、`--train-jsonl`、`--val-jsonl` 和 `--output-dir`；其余训练参数见 `--help`。支持既有 YAML 默认值、全量微调和 LoRA。外层 `experiments/training/sft1/train.py` 调用此实现。

## 模块职责与数据流

- `data.py`：读取记录中的对话和截图，构造仅监督回答的标签，屏蔽提示词、填充和注入的 query 位置；使用右侧填充保留文本区间的位置关系，并负责样本编码缓存。
- `config.py`：读取 YAML 默认配置。`cli.py`：定义命令行选项，在加载模型前校验训练阶段和参数。
- `trainer.py`：加载 Qwen、设置可训练参数、构建优化器，驱动梯度累积、离线验证和 epoch checkpoint 保存。SFT1直接使用 teacher forcing 的回答 CE，不计算 DINO 或 WM 损失。SFT2显式选择 query 阶段后复用同一训练生命周期。
- `distributed.py`：建立和清理分布式进程组，提供主进程判断与同步；checkpoint 模块不依赖训练循环。
- `checkpoint.py`：保存训练状态、查找恢复位置和校验阶段身份，独立于训练循环。
- `checkpoint_export.py`：负责 LoRA 合并与导出校验，包括单独训练的 embedding 和输出 head；历史合并脚本调用此实现。

## 保存与恢复

Checkpoint 保存优化器、调度器、epoch/step 和 `training_stage=format`。明确识别出的旧格式训练 checkpoint 仍可恢复；Query/WM checkpoint 不能静默按格式阶段恢复。

恢复从已保存的 epoch 边界继续，现有训练循环不保存随机数状态，因此不保证逐位重放。离线 loss/格式验证不等于环境 rollout；环境评估见 [`../evaluation/`](../evaluation/README.md)。
