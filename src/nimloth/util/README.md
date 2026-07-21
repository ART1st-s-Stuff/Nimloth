# 公共工具

`nimloth.util` 保存训练和评估共享、且不定义 Agent、environment、rollout、
backbone 或 world model 语义的运行工具。

- `distributed.py`：进程组和 rank 工具。
- `module.py`：临时 train/eval mode 等 module 生命周期工具。
- `metrics.py`：指标累计。
- `csv_log.py`：固定列的训练/评估 CSV 记录。
- `optim.py`：optimizer group、学习率和公共梯度更新运行期。
- `profiling.py`：可选的步骤计时。
- `experiment.py`：实验元数据和输出目录创建。
- `wandb.py`：可恢复的 W&B 初始化及指标记录。
- `cache/`：可复用的预处理缓存设施。
