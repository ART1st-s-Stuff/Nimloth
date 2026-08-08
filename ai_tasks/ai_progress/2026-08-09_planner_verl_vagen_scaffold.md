# 2026-08-09 PlannerPolicyHead 复用 VERL/VAGEN 脚手架

## 目标

在不改变当前 PlannerPolicyHead 环境动作 PPO、executed-action `Q(s,a)`、WM/DINO 联合目标和 fresh-consumption/checkpoint 语义的前提下，尽可能复用 VERL/VAGEN 的训练与 rollout 脚手架，降低当前逐 transition 完整 prefix 重算和低并发 rollout 带来的耗时。

## 人类确认的边界

- 保留 PlannerPolicyHead、`Q(s,a)`、WM/DINO 目标，不切换为 stock response-token PPO。
- 正在运行的正式 ID147 Job `511059`继续跑完；本任务在独立 worktree 开发，不改变 ID147 runtime commit、输出或训练配置。
- 第一阶段同时覆盖训练后端和 rollout，但按可独立验证的适配层分步落地。

## 已确认事实

- 当前 pinned VAGEN 默认 `actor_rollout_ref.actor.ppo_epochs: 1`；critic 继承同一值。现有 Nimloth VAGEN baseline launcher 没有覆盖它，因此不是4 epochs。
- VERL actor/critic 的每个 epoch 会遍历全部 PPO mini-batch；一个 global step 的 optimizer-step 数还取决于 train batch / mini-batch，而不能只看 `ppo_epochs`。
- 当前 PlannerPolicyHead run 配置为每个 fresh rollout batch 做4 epochs；这是 Nimloth 自定义训练语义，不是从 VAGEN 默认值继承。
- VERL可复用能力包括 DataProto、token-budgeted dynamic micro-batch、FSDP、gradient checkpointing、offload、worker/resource orchestration、checkpoint；VAGEN可复用多轮 navigation rollout manager、env client/protocol和并发调度。
- stock VAGEN/VERL actor/critic训练 response token policy/value，不能直接表达当前环境动作分布、完整 decision prefix、PlannerPolicyHead 和 `Q(s,a)`。

## 既有资产

`feat/fsdp-dynamic-rollout` 已实现并真实门禁过一套旧版 VERL/VAGEN 在线路径，包括 `verl_adapter.py`、`vagen_online_rollout.py`、DataProto、FSDP full actor/critic、动态 rollout 和 checkpoint sidecar。该分支基于较早 commit，目标后来演变为 token PPO/critic，且与当前 PlannerPolicyHead 代码大幅分叉。

因此本任务从当前 `dev` commit `79f12f06`建立新分支 `feat/planner-verl-vagen-scaffold`。旧分支只作为经过验证的组件来源；禁止整分支合并或静默恢复已经废弃的 token-PPO/固定 thought/旧 schema 语义。

## 实施计划

1. 记录 ID147 的 rollout、训练、保存分项耗时，建立不改变算法的性能基线。
2. 写 RED contract tests：同一 batch 下 old log-prob、advantage、clipped actor loss、executed-action value、WM/DINO eligibility 和 consumption 边界必须与当前实现一致。
3. 定义 Planner decision batch/DataProto adapter；对可变长 prefix 做 token-budgeted pack，禁止截断、补默认 action 或丢弃 terminal CoT。
4. 实现 Nimloth custom VERL worker：复用 FSDP/checkpointing/offload/dynamic micro-batch，但调用当前 PlannerPolicyHead、ValueHead、WM/DINO loss。
5. 接入 VAGEN rollout manager 的 active-env 并发和 Ray 资源编排；保留当前真实 CoT、严格 manifest、seed ownership 和两路 TP rollout 的可审计输出。
6. 依次通过 CPU parity、单GPU非消费型 batch、分布式非消费型 mechanics、fresh rollout smoke；通过前不替换正式 runner。
7. 用 wall-clock、GPU峰值、transition/s、trajectory/s 和逐项数值 parity 比较旧/新后端，再决定默认 `ppo_epochs`。VAGEN默认1只能作为性能对照，不能未经批准改变 ID147 的4-epoch算法。

## 当前状态

- 独立 worktree：`/workspace/remote2/nimloth-feat-planner-verl-vagen-scaffold`
- 分支：`feat/planner-verl-vagen-scaffold`
- 起点：`79f12f06601ce514dabe8fac957317007804506d`
- 尚未修改生产代码、未提交GPU实验、未改变ID147。
- 最近一次查询 ID147 时 SSH 连接在登录入口被关闭，未取得新状态；不能把此前 iteration 2 状态当作当前状态。
