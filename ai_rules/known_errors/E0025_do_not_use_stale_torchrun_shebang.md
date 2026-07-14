# E0025：不能假设指定环境目录中的 `torchrun` 使用同一环境

## 已发生的错误

RCDM 并行 cache smoke 使用 `/project/peilab/atst/nimloth/.venv-vagen-main/bin/torchrun`，但该脚本的 shebang 实际指向旧 `.venv/bin/python3`。job `474334` 因此加载了不兼容的 Transformers，并在模型权重加载前报 `AttributeError: 'dict' object has no attribute 'to_dict'`。

## 原因

可执行脚本位于目标环境目录，不代表它的 shebang 也指向该环境。直接调用 `torchrun` 绕过了已经明确固定的 Python 解释器。

## 正确做法

服务器分布式任务必须使用：

```bash
/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 -m torch.distributed.run ...
```

禁止依赖 `activate`、`PATH` 或 `.venv-vagen-main/bin/torchrun` 的 shebang；启动日志应记录实际 `sys.executable`。

## 证据

- 失败 job：`474334`
- 失败输出：`.../rcdm/0_smoke_parallel2_sft2e2_k8_statecache/logs/slurm-474334.err`
- 同类历史记录：`AI_branch_progress.md` 中旧 SFT2 startup `.venv` / `.venv-vagen-main` 对比。
