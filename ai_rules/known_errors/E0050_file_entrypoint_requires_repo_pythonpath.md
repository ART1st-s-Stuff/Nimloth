# E0050: 文件入口也必须显式导出repo PYTHONPATH

## 已发生错误

ID25 memory probe的rank runner直接执行`experiments/.../probe_actor_recompute_memory.py`。虽然先`cd`到repo，但Python文件入口只把脚本目录加入`sys.path`；runner未导出`${REPO}/src`，所有rank在模型加载前触发`ModuleNotFoundError: nimloth`，28秒失败且没有显存结果。

## 正确做法

- 每个独立rank runner都必须显式导出repo `src`、VAGEN、VERL和le-wm路径；不得假设`cd`会让src-layout package可导入。
- 提交前用runner的同一Python文件入口至少执行`--help`或import probe，不能只做`py_compile`和bash语法检查。
- GPU allocation一旦实际启动，该实验identity即使模型forward为0也应terminal；修复后使用新identity。
