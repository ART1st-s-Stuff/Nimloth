# 2026-07-24：删除 SFT2 fixed terminal CoT

## 人类确认的 state 语义

- state 为 CoT-conditioned state；
- 普通 state 使用实际 rollout assistant response 中的真实 CoT；
- terminal observation 额外使用 SFT1 初始化 checkpoint 生成一次真实 CoT并持久化，
  但不执行对应动作；
- 未明确的生成参数必须询问人类，禁止 agent 猜测。
- 人类已确认 terminal CoT 使用 VAGEN validation sampling：`temperature=0`、
  `top_p=1.0`、`top_k=-1`、`do_sample=false`、`n=1`。
- 人类进一步确认参考VAGEN数据使用`max_reasoning_tokens=128`、`seed=42`、
  `max_pixels=602112`。全量train/val/test共69,776段CoT，最大93 tokens且0段超过128；
  602112与SFT1初始化checkpoint processor一致，旧512px图像按504px条件生成。

## 本次修改

- `rollout/transitions.py` 删除 terminal `assistant_prefix()` fixed fallback；
- 结构化轨迹展开不再由模板重建响应，改为读取真实 `assistant_responses`；
- 新增 `terminal_assistant_prefix` 数据契约与离线生成入口；terminal prefix 只包含实际
  生成的 CoT和注入的 latent query/`action_start`，不包含未来 action token；
- cache expansion version 升级为 `wm_expand_v3_terminal_cot`，旧 fixed cache 失效；
- 更新 AGENTS.md 与 known error，禁止 AI 再自行发明 fixed CoT。

## 验证状态

- `git diff --check` 通过；
- 本机 Python 环境没有 Torch/pytest；superpod SSH 建立 host key 后连续约60秒无响应，
  已按服务器规则中止且未重试，因此远端定向回归待执行；
- 尚未生成 train/val terminal CoT 数据，尚未重建 cache，也未启动 SFT2/RL 实验。
- 全部生成参数已确认；下一步为远端回归、生成train/val terminal CoT、重建cache并
  以新实验ID重跑SFT2。
- ID47在hold `486556`/dgx-42上执行首条21帧真实terminal生成smoke，入口先误报128
  tokens内未闭合；512诊断也同样误报。实际continuation仅15 tokens，并在约第5 token
  解码为`Move left.</think>`。根因是BPE把句点与`</`合并，代码却查找独立编码的
  `</think>` token子序列。修复改为对continuation解码文本精确查找/停止，并新增跨
  merged-token boundary回归；正式augmented JSONL/cache/SFT2仍未创建。
- 修复提交`ebc4d3b`后，superpod定向回归`8 passed`；同一条21帧真实GPU smoke使用
  完整确认参数成功，terminal CoT为3 tokens，输出/manifest SHA256和全部生成参数齐全。
  正式train/val augmented JSONL仍未开始。
- 人类因当前RL分支与此前DINO监督SFT2 lineage存在冲突而要求暂停。hold `486556`
  已取消并退出队列；未创建正式augmented JSONL、cache、W&B run、optimizer step或
  checkpoint。冲突解决前不得继续启动。
