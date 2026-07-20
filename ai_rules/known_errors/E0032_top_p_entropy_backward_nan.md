# E0032：top-p masked entropy必须验证backward有限

## 已发生的错误

RL k=8 hybrid smoke job`482474`的iteration 1所有forward指标有限，但optimizer step后Qwen language、WM predictor和ValueHead checkpoint参数变成NaN；fresh-process resume的iteration 2全部训练loss为NaN。

## 原因

Top-p把排除动作的logit写为`-inf`。`compute_action_entropy()`用`torch.where(p > 0, p * log_p, 0)`得到有限forward值，但autograd仍经过未选分支中的`0 * -inf`，产生NaN gradient。随后global gradient clipping得到NaN norm并污染全部trainable gradient。

## 正确做法

计算entropy前把非有限masked logits替换为足够小的有限sentinel，再执行softmax/log-softmax；测试必须调用`backward()`并检查原始logits gradient有限，不能只检查entropy scalar有限。Optimizer step前还必须对loss和clipped gradient norm启用non-finite fail-fast，禁止写出已损坏checkpoint。

## 证据

- 失败job：`482474`；W&B：`nimloth-rl/ixvtbpqr`
- 失败输出：`outputs/experiments/training/rl/2026-07-20/64_smoke_k8inject_qwenfirst_wmsecond_base4x2_h2_multistep2_fsdp2_iter2_retry1`
- 修复位置：`src/nimloth/training/rl/loss.py::compute_action_entropy`
- 回归测试：`tests/training/rl/test_actor_ownership.py::test_behavior_log_probs_use_temperature_and_top_p_transform`
