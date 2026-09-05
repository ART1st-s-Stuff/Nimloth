# E0104: Custom vLLM replica also needs a worker rollout class

## 已发生的错误

ID166 Job `519129`的8个FSDP actor/rollout worker已加载并包装模型，但在
`_build_rollout()`失败：`RolloutReplicaRegistry`中注册`nimloth_vllm`只覆盖agent-loop
HTTP replica；VERL worker另调用`verl.workers.rollout.base.get_rollout_class(name, mode)`，
其独立registry没有`nimloth_vllm/async`。

## 正确做法

新增custom async vLLM rollout name时必须同时注册两层：

1. worker rollout class（本项目复用stock `vLLMAsyncRollout`负责权重/engine adapter）；
2. agent-loop HTTP replica（Nimloth自定义server/capture行为）。

外部库应在fresh worker process经`import_external_libs`后测试
`get_rollout_class(custom_name, "async")`，不能只测driver registry或standalone replica。

## Evidence

- `external/VAGEN/verl/verl/workers/rollout/base.py`：strict external class registration API。
- `external/VAGEN/vagen/rollout/nimloth_vllm.py`：worker class与HTTP replica双注册。
- `external/VAGEN/tests/test_joint_training_config_wiring.py`：fresh-process worker注册回归。
- 服务器ID166 `failure_analysis.md`：真实8-rank失败边界。
