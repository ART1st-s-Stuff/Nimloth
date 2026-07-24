# E0044：不得把direct-policy PPO smoke声称为完整PPO

## 已确认错误

agent 曾把已经完成一次真实 optimizer step 的 direct-policy online PPO smoke 概括为
“PPO 已做完”。人类指出，planning policy 的 PPO 并未实现，因此该表述扩大了实际完成
范围。

## 错误原因

把运行链路验证与算法功能完整性混为一谈。ID94只证明Qwen直接behavior的fresh
rollout、HF token replay、PPO loss、backward、gradient synchronization和checkpoint
可以完成一次；它没有证明planner behavior replay、正确的逐步return/truncation语义或
长时多次online update已经实现。

## 正确做法

- 汇报时必须写“direct-policy online PPO单次GPU optimizer-step smoke已通过”。
- planning behavior未实现policy replay/update时，禁止称为planning PPO。
- 未验证多次fresh rollout/update闭环前，禁止称完整online PPO训练已经跑通。
- 分别汇报已实现机制、真实运行验证范围、尚未实现机制和已知目标语义缺口。

## 证据

- `src/nimloth/training/rl/rollout_runtime.py`显式拒绝同时开启planning和actor。
- `src/nimloth/agent/planning.py`只保存planner root behavior分布，没有Qwen token trace。
- `src/nimloth/rollout/from_agent.py`只把逐步reward求和后写入trajectory。
