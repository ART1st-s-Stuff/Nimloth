# E0118：被捕获的optional JIT warning不是fatal job结果

## 错误

ID180的TP8 vLLM初始化期间，`tvm_ffi`尝试构建可选torch DLPack addon。
`/usr/bin/nvcc`是cluster教程wrapper，因缺少`colorama`抛出异常；但
`tvm_ffi/_optional_torch_c_dlpack.py`在外层捕获该异常，只警告
`EnvTensorAllocator will not be enabled`并返回`None`。

Agent只看到warning中包含的traceback和一段安静日志，就把它误判为vLLM fatal failure；
尽管8卡仍各持有约47--55 GiB且有间歇利用率，仍在4200秒phase timeout之前取消了Job
`519889`。

## 后果

ID180在rollout、optimizer、source777、checkpoint和restore-only phase之前被agent主动终止，
没有产生可用于判断实现正确性的实验结论。该ID、输出和W&B identity已消耗且不可恢复。

## 正确做法

- 判断fatality必须依赖未捕获异常/failed future、主进程或worker死亡、terminal Slurm状态、
  明确无进展的资源证据，或合同timeout；不能只看warning里嵌入的traceback。
- 先读取异常调用点的外层控制流，确认异常是否被捕获、是否只是optional capability fallback。
- 模型初始化期间同时检查GPU memory/utilization、进程存活和日志阶段；资源仍活跃时禁止以
  “日志安静”为理由提前取消。
- 取消前重新核对当前状态，也必须核对“为什么确定它不可能继续”；`RUNNING`本身不能支持取消。
