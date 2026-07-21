# Agent 配置

本目录保存训练阶段无关的 Agent 配置。`AgentConfig` 选择 prompt 模板并生成可随
trajectory 持久化的 `PromptTemplateSpec`。环境 system prompt 和动作说明不属于
此配置，由 `EnvironmentSession` 在 reset 后提供。
