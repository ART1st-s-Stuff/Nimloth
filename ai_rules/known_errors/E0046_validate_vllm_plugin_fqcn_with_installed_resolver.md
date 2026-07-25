# E0046：vLLM 插件路径必须用安装版 resolver 验证

## 已确认错误

planner rollout 为 `worker_extension_cls` 传入了
`module:Class`，fake LLM 测试只检查该字符串以冒号结尾，因此测试通过；集群实际安装的
vLLM 0.11 使用 `resolve_obj_by_qualname()`，该字段只接受 `module.Class`，真实 worker
初始化会失败。

## 原因

vLLM 0.11 的两类插件使用不同语法：`logits_processors` 接受 `module:Class`，
`worker_extension_cls` 接受 `module.Class`。测试复述了实现字符串，没有调用安装版
resolver 验证。

## 正确做法

- 分别遵守两个字段的实际解析协议，禁止按相似名称推断。
- 涉及外部框架的字符串插件入口时，除单元测试外，必须用目标安装版本的 resolver
  或真实初始化路径验证。
- 当前正确 worker extension 路径为
  `nimloth.backbone.qwen25vl.vllm_hidden.PolicyStateCaptureWorkerExtension`。

相关代码：`src/nimloth/backbone/qwen25vl/vllm_policy.py`、
`tests/backbone/qwen25vl/test_vllm_policy.py`。
