# 2026-07-24：删除 SFT2 fixed terminal CoT

## 人类确认的 state 语义

- state 为 CoT-conditioned state；
- 普通 state 使用实际 rollout assistant response 中的真实 CoT；
- terminal observation 额外使用 SFT1 初始化 checkpoint 生成一次真实 CoT并持久化，
  但不执行对应动作；
- 未明确的生成参数必须询问人类，禁止 agent 猜测。

## 本次修改

- `rollout/transitions.py` 删除 terminal `assistant_prefix()` fixed fallback；
- 结构化轨迹展开不再由模板重建响应，改为读取真实 `assistant_responses`；
- 新增 `terminal_assistant_prefix` 数据契约与离线生成入口；terminal prefix 只包含实际
  生成的 CoT和注入的 latent query/`action_start`，不包含未来 action token；
- cache expansion version 升级为 `wm_expand_v3_terminal_cot`，旧 fixed cache 失效；
- 更新 AGENTS.md 与 known error，禁止 AI 再自行发明 fixed CoT。

## 验证状态

- `git diff --check` 通过；
- 本机 Python 环境没有 pytest，远端定向回归待执行；
- 尚未生成 train/val terminal CoT 数据，尚未重建 cache，也未启动 SFT2/RL 实验。
