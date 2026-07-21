# 配置模块

`nimloth.config` 负责配置加载和严格的阶段 schema。运行时代码只接收已经
解析的配置，不自行读取 YAML，也不依赖命令行 namespace。

- `agent/`：训练和评估共用的 prompt 模板配置。
- `rollout/`：训练和评估共用的数据源与 behavior sampling 配置。
- `sft2/`：SFT2 阶段 YAML 到 CLI 的映射。
- `rl/`：RL 阶段配置及命令行覆盖；组合公共 Agent/Rollout 配置。
- `io.py`：各阶段共享的文件格式加载工具。

具体实验配置仍保存在仓库顶层的 `configs/`。
