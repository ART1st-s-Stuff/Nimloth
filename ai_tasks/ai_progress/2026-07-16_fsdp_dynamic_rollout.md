# FSDP dynamic rollout

## 目标

为现有 Nimloth RL FSDP trainer 实现真正的在线动态 rollout：每个 RL iteration 使用当前更新后的 k=1/inject policy 访问 VAGEN train-split 环境，随后执行 PPO + WM/value update；参考 VAGEN actor-rollout/env-service 循环，但保留 Nimloth 的 latent、WM/value loss 和 checkpoint ownership。

## 约束

- 仅 rank0 访问外部 VAGEN/AI2-THOR HTTP env service；所有 FSDP rank 必须以相同顺序和形状参与 policy/encoding/PPO forward。
- rank0 采样并广播动作；各 rank 计算同一 action distribution，并校验一致性。
- 所有 rank 获得完全相同的完整 trajectory；只允许 rank0 写 JSONL/PNG，禁止文件写入竞争。
- env、policy 或 schema 错误必须同步失败或丢弃完整 episode，不得以默认动作/零 log-prob 冒充有效 rollout。
- train rollout 只允许实际 `*_train` split；当前 runtime 仅支持 k=1 inject。
- resume 后 rollout seed 不得从0重复；checkpoint world size gate保持不变。

## 当前计划

1. 将 action distribution 计算与 rank0 sampling 解耦。
2. 新增 distributed env collector，同步 rank0 env control、all-rank FSDP forward、rank0 action broadcast、trajectory broadcast。
3. trainer 允许 world>1 env collector，并在每轮 rollout 时保持 inference mode、设置可恢复 seed offset。
4. 增加单测及2-rank CPU/gloo同步集成测试；再做服务器2-GPU短 smoke。
5. 通过后再为 dgx-09 env + dgx-32 FSDP 正式在线 RL 请求昂贵实验确认。

## 已完成

- 创建分支/worktree：`feat/fsdp-dynamic-rollout` / `../nimloth-feat-fsdp-dynamic-rollout`，起点 dev `25f4237`。
- 核实 VAGEN：每个 global step动态 reset/step env，当前 actor产生trajectory，随后 update_actor；下一个step生成前同步最新FSDP权重。
- 核实现有 blocker：`train_rl`明确拒绝 world>1 `EnvRolloutCollector`；JSONL path不能提供每轮更新后的policy rollout。

## 修改

- 尚无代码修改。

## 验证

- 尚未执行。

## 待确认/风险

- 外部环境服务超时期间其他rank会等待rank0广播；必须使用有限HTTP timeout并同步传播错误。
- 先实现正确的逐episode同步forward；性能批处理可在语义验证后单独优化。
