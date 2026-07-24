# Agent prompt/runtime 重构进度

日期：2026-07-21

## 目标

- 让 `src/nimloth/agent/` 成为 Agent transcript、prompt 模板和实际运行时的 owner。
- SFT2、RL rollout 与 PPO 重放使用同一份 prompt/action 契约。
- 删除当前没有仓库内调用方的 `WMAgent` 原型，避免死代码继续充当公共接口。

## 已确认设计

- 使用环境返回的 `system_prompt` 和每步 `obs_str` 作为权威 transcript 文本，不重新发明导航 instruction 文案。
- transcript 单独保存图片路径/对象，消息文本保留 `<image>` 占位符；模型调用时由共享模板绑定真实图片。
- 在线 action query 与 supervised action response 共享相同 assistant prefix，保证 rollout old probability、PPO new probability 和 SFT2 action target 的条件输入一致。
- 新结构化记录走共享模板；缺少结构化字段的旧 SFT2 JSONL 继续读取原有 `messages`。

## 完成状态

1. 已定义 agent-owned transcript、prompt 与 action 契约。
2. 已实现真实 `NavigationAgent` 运行时并移除死 `WMAgent` 原型。
3. Qwen policy、RL rollout、PPO replay 和 WM state encoding 已统一使用 Agent policy prompt。
4. SFT2 structured transition 使用同一模板；SFT1 converter 的 assistant action block 也由该模板生成，并保留原 reasoning。
5. 新 RL JSONL 会在写入前和训练前校验 prompt 版本、完整 observation/image/action 数量、action name/index、8-way 行为分布、采样参数及 prompt/message 重建一致性。
6. 旧 SFT2 `messages` 记录仍可读取；旧 RL 记录无法精确重放 policy state，因此 trainer 会拒绝。

## 文件修改

- `src/nimloth/agent/prompt.py`：共享 transcript、prompt 和图片绑定。
- `src/nimloth/agent/runtime.py`：真实 Agent episode 状态与 policy 调用。
- `src/nimloth/backbone/qwen25vl/policy.py`：Qwen action policy adapter。
- `src/nimloth/training/rl/rollout.py`：Agent 驱动 rollout、结构化 JSONL 和统一 schema 校验。
- `src/nimloth/training/rl/trainer.py`：统一 state 编码和 PPO prompt/distribution 重放。
- `src/nimloth/wm/dataset.py`：结构化记录的 SFT2 transition 展开。
- `experiments/training/sft1/convert_rollouts.py`：supervised action block 复用 Agent 模板。
- `src/nimloth/agent/inference.py`：删除死代码。
- 对应 README、质量清单、后续 k>1 计划、代码地图和单元测试已同步。

## 验证

- Agent/Qwen/SFT2/RL/WM 相关组合测试：`116 passed, 1 deselected`。
- 扩大到本地可收集测试：`184 passed, 1 deselected`。
- deselected：两进程 Gloo metric 测试在当前沙箱无法解析 loopback；之前已在允许本地通信的环境通过，不属于本次改动。
- 完整测试收集的三个基线环境阻塞：旧 VAGEN `envs/` 测试文件路径不存在；两个无 package 的 `test_config.py` 发生同名导入冲突；当前 Python 测试环境缺少 pandas。
- `compileall` 与 `git diff --check` 通过。

## Memory 评估

- 本任务未读取 repo memory，因此没有需要重新检查或 upvote 的项目 memory。
- 设计与迁移规则已经由源码 README、质量清单和分支进度完整记录，不额外创建重复的 durable memory。

## 待确认问题

- 无。旧 SFT2 JSONL 兼容路径仅保留读取，不作为新数据的规范写入格式。
