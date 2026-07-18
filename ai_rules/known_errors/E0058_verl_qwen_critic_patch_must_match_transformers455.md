# E0058 — VERL Qwen critic patch必须匹配Transformers4.55结构

## 已发生的错误

ID33已完成full actor FSDP、PPO-old和immutable-ref初始化/计算，随后critic初始化导入旧patch失败：Transformers4.55删除了`Qwen2_5_VL_START_DOCSTRING`等常量，并把视觉和语言模块统一放进`Qwen2_5_VLModel`。旧patch面向4.49 flat结构。

## 正确做法

- 4.55 token critic直接组合`Qwen2_5_VLModel(config)`和标量`score` head。
- 禁止导入已删除的docstring常量，禁止重复创建visual tower。
- VERL worker加载critic前显式安装Nimloth 4.55 patch，并用真实multimodal exact transcript做value/update gate。

## 证据

- `src/nimloth/training/rl/verl_critic_455.py`
- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID33 README：`outputs/experiments/training/rl/2026-07-18/33_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_torchrun_tiedhead_maskedgae/README.md`
