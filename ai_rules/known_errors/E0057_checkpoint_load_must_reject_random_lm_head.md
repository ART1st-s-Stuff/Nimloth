# E0057 — checkpoint加载不能接受随机`lm_head`

## 已发生的错误

ID32在Transformers4.55.4加载k8 merged checkpoint时，八个rank均报告`lm_head.weight`缺失并随机初始化。源config把语言模型tying放在嵌套`text_config`，但顶层`tie_word_embeddings=false`；VERL worker按顶层配置构建模型。继续执行会让actor/ref拥有独立随机head并破坏PPO-old/reference语义。

## 正确做法

- VERL actor/ref override显式设置`tie_word_embeddings=true`。
- worker初始化后检查output embedding与input embedding共享同一storage。
- 任意missing/random lm_head必须fail closed，不能只当Transformers warning忽略。

## 证据

- `src/nimloth/training/rl/verl_gate.py`
- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID32 README：`outputs/experiments/training/rl/2026-07-18/32_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_wandbidentityfix_maskedgae/README.md`
