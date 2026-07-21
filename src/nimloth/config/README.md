# 配置模块

`nimloth.config` 负责配置加载和严格的阶段 schema。运行时代码只接收已经
解析的配置，不自行读取 YAML，也不依赖命令行 namespace。

- `sft2/`：SFT2 YAML 到 CLI 的映射。
- `rl/`：RL YAML 加载与命令行覆盖。
- `io.py`：各阶段共享的文件格式加载工具。

具体实验配置仍保存在仓库顶层的 `configs/`。
