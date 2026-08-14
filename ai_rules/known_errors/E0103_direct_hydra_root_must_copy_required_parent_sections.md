# E0103: Direct Hydra root must copy required parent sections

## 已发生的错误

ID165为解决nested Hydra search-path问题，从继承`vagen_multiturn`改为直接组合
`ppo_trainer + env_registry`。迁移时遗漏了`huggingface_hub`。真实Job `519097.1`
在构造`RayPPOTrainer`时，`HFUploadManager`从缺失字段得到普通`{}`，随后
`OmegaConf.to_container`报错；模型worker尚未创建。

## 正确做法

把已有Hydra配置改成新的direct root时，必须审计原parent提供的全部运行时section，不能只复制
当前功能看起来会使用的参数。对于即使disabled也会在constructor读取的组件，应使用真实composed
config构造该组件作为preflight。source-string检查和`--cfg job`成功不足以证明constructor可运行。

## Evidence

- `external/VAGEN/vagen/configs/joint_id165_gate.yaml`：显式disabled `huggingface_hub` section。
- `external/VAGEN/tests/test_joint_training_config_wiring.py`：真实`HFUploadManager` constructor回归。
- 服务器ID165输出`failure_analysis.md`：Job 519097 root exception与运行边界。
