# E0113：提交batch前必须验证cluster module名称

## 错误

ID175 batch脚本直接写入`module load cuda/12.8 slurm`，没有先在目标cluster验证
`cuda/12.8` modulefile是否存在。

## 后果

Job`519755`虽获得8×H800 allocation，却在0秒内因`Unable to locate a modulefile for
'cuda/12.8'`退出；模型、数据、Python和GPU任务均未启动，ID175仍被消耗。

## 正确做法

- batch脚本只能加载已在目标cluster验证存在且确实需要的module。
- 使用项目固定Python venv自带的PyTorch/CUDA runtime时，不额外猜测CUDA module名称。
- module失败发生在所有业务preflight之前，也必须使用新实验ID/output重试。
