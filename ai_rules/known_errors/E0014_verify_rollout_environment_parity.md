# E0014：rollout 不能只按 `prompt_format` 名称判断与源 checkpoint 一致

## 已发生的错误

源 step60 与新 rollout 都写成 `prompt_format=eval_mode`，因此错误地把 pinned legacy VAGEN rollout 当成源训练 evaluation 的等价复现，并开始了 full-scale 数据采集。

## 原因

同名格式来自两套不同的 navigation 实现。源 step60 transcript 的 prompt/action/reward feedback 与 VAGEN `f7aefd3` 实现逐字匹配；该实现配置为 canonical underscore actions、0.3m step、1.0m success threshold、0.01 per-turn format reward。本次 VAGEN `44be18c` legacy 实现使用 compact action aliases、不同 prompt、0.5m step、1.5m threshold、0.5 format reward和10.0 success reward。源 W&B 没有 git commit，因此几何参数仍须由 parity smoke 最终确认；生成采样参数相同不能消除已确认的 transcript 差异。

## 正确做法

从旧 checkpoint 收集 rollout 前，必须逐项对照真实 transcript 和环境实现：system/init/action prompt、action vocabulary/parser、step length、success threshold、reward feedback、max turns/actions及 generation kwargs。full-scale 前必须做相同 seed 的 source-compatible parity smoke，不能只核对配置字段名称或正常退出。

## 证据

- 源记录：`.../vagen_legacy_wm_entropy01_kl001_60step_2env4train/validation/60.jsonl`
- 无效尝试：`.../full_2e66e97/rollout/invalid_attempt_dafbd30_prompt_env_mismatch/`
- 当前 legacy 实现：`external/VAGEN/vagen/env/navigation/{prompt.py,env.py,env_config.py}`
