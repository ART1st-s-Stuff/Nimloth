# E0033：checkpoint delta probe必须使用实际HF export key

## 已发生的错误

RL k=8 hybrid smoke job`482484`完成两次finite FSDP update和fresh-process resume，但最终validator错误报告`could not find Qwen language/vision probe tensors`，令Slurm状态为FAILED。

## 原因

Validator把language probe key模式硬编码为`language_model.layers.0`，而本checkpoint的实际HF index使用`model.layers.0.self_attn.q_proj.weight`；vision key实际以`visual.`开头。训练产物本身没有触发该错误。

## 正确做法

编写parameter-delta gate前必须检查源与目标`model.safetensors.index.json`的实际公共key；language和vision probe分别匹配当前export的`model.layers.*`与`visual.*`。Validator key选择也要有回归测试，不能凭Transformers类名猜state-dict前缀。

## 证据

- job：`482484`；W&B：`nimloth-rl/nrxroyos`
- 源/目标index公共key：`model.layers.0.self_attn.q_proj.weight`和`visual.*`
- 修复位置：`experiments/training/rl/run_k8_wm_fastpath_smoke.sh`
- 回归测试：`tests/training/rl/test_cli_metadata.py::test_k8_wm_fastpath_smoke_config_covers_two_step_windows`
