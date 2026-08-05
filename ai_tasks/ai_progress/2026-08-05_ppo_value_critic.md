# 2026-08-05 PPO ValueHead critic

## 目标

- 保留当前 receding-horizon planner/MCTS 对 environment action 的 ownership。
- 将 planner RL 的执行动作 `Q(s_t,a_t)` 从单次 Monte Carlo MSE 改为带 frozen
  old-value 的 PPO-style clipped critic objective。
- ValueHead critic 梯度继续经过当前 decision state、StateProjector 和完整 Qwen
  prefix；不把 planner action 冒充为 Qwen action-token policy PPO。

## 基线与边界

- feature branch：`feat/ppo-value-critic`。
- 实现基线：`13b3c711`，包含 ID122/ID125 使用的 rank-aware transition sharding、
  fresh rollout lifecycle 与 decoded-text stop 修复；不基于本地较旧的 `2007c661`
  training loop 直接修改。
- SFT2 warm-start 的 outgoing `Q(s_t,a_t)` 时序保持不变。
- 不启动 GPU、Slurm、rollout 或训练实验。

## 当前计划

1. 定义 frozen old-value、clipped critic loss、PPO update epoch 与指标合同。
2. 修改配置、algorithm、loop、checkpoint objective/invariants 和正式 planner 配置。
3. 增加 clipping、Qwen梯度、multi-rank padding、fresh-consumption、config/checkpoint 回归。
4. 运行本地定向测试、compile/static check，审查并提交相关文件。

## 待验证风险

- 只有同一 frozen rollout batch 上至少两个 optimizer epoch 时，old-value clipping
  才会在参数更新后产生实际约束；单 epoch 与原 MC MSE 的首步行为等价。
- 多个 critic epoch 会增加 Qwen full-prefix forward/backward 成本；实现必须用显式配置，
  后续真实资源合同需单独确认。
- old value 必须来自 update 前的 rollout decision state 和同一 ValueHead checkpoint；
  不使用 action-token log-prob，也不把 MCTS root score当作 direct `Q(s_t,a_t)`。

## 状态

- 已建立独立 worktree，并 fast-forward 到当前 active RL source `13b3c711`。
- 已实现frozen rollout old value、执行动作的clipped critic objective和多critic epoch；
  首个epoch同时训练WM/DINO，后续epoch只训练critic及其上游Qwen表征。
- 已将planner objective标记更新为`receding_horizon_decision_state_ppo_value_v1`，
  checkpoint保存并严格校验完整ValueHead配置，旧objective checkpoint fail closed。
- planner PPO拒绝普通静态JSONL；只有同进程在线rollout或精确fingerprint匹配的fresh
  manifest能提供有效的behavior-checkpoint decision state与frozen old value。
- 所有正式planner YAML显式使用`ppo_clip_range: 0.2`和`ppo_epochs: 4`；这些是待真实
  实验确认的超参数，不视为质量结论。
- 已增加loss、Qwen梯度、saved-state old value、loop epoch、配置与resume回归。
- CPU测试：包含PPO critic、原公共WM objective、config/resume/transition/loop及freshness
  门禁的直接相关套件`80 passed`。当前完整RL套件为`183 passed, 1 failed`；唯一失败
  来自VAGEN测试导入
  时缺少未安装的`gym`传递依赖，失败路径未经过本次修改代码。该环境失败不算回归通过，
  但本次所有直接相关测试均已通过。
- `py_compile`、`git diff --check`通过；19个启用planner的YAML全部同时包含
  `ppo_clip_range: 0.2`与`ppo_epochs: 4`。
- 未运行GPU、Slurm、rollout或训练实验。

## GPU mechanics gate preflight

- 人类随后明确要求使用GPU测试。正式planner拓扑的每个训练rank跨2张GPU，因此跨rank
  `model_parallel_ddp`最小真实门禁是`world_size=2 x gpus_per_rank=2`，总计4张GPU；
  2张GPU只能覆盖单rank模型并行，不能证明DDP同步。
- commit `64726911`新增单卡真实Qwen critic backward和4卡正式拓扑AdamW多epoch门禁；
  使用ID125 iteration1中与SFT2 epoch1 fingerprint精确匹配的真实轨迹，只读验证其
  manifest，不生成fixed CoT、不保存checkpoint、不把该门禁当作新鲜rollout或质量实验。
- 完整资源、输入、冻结边界、成功判据和证据边界见
  `ai_tasks/ai_progress/2026-08-05_ppo_value_critic_gpu_gate.md`。当前仅完成preflight和
  静态校验，尚未提交Slurm job或产生GPU结果。
- 人类确认4卡合同后提交ID126 Job `506808`，实际在`preempt/dgx-16:4`运行2分44秒后
  exit1。artifact fingerprint与Qwen双shard加载均完成，但门禁错误地在collector而非
  manifest上调用`validate_processor`，在任何state forward/backward/optimizer/DDP之前
  fail closed；没有梯度结果或checkpoint，ID126不可resume，修复后必须使用新ID127。
- 修复processor API后，ID127 Job `506813`单卡阶段通过：真实planner动作`lookup`的
  ValueHead loss产生Qwen final-norm最大梯度`0.0107421875`与ValueHead最大梯度
  `0.651180625`，`lm_head.grad=None`且frozen StateProjector/vision无参数梯度，峰值显存
  14.78 GB。随后正式`world_size=2 x gpus_per_rank=2`完成首个backward+AdamW step；
  Qwen witness精确同步，ValueHead witness仅有`1.024e-7`跨设备舍入差，但门禁错误要求
  bit equality而终止，未完成epoch2--4。ID127无checkpoint且不可resume；显式FP32容差和
  梯度/参数差值记录后用新ID128重试。
- ID128 Job `506831`的新梯度assertion排除了“只是ValueHead浮点舍入”
  的旧解释：首个2-rank critic backward后Qwen final-norm梯度差为
  `0.002227783203125`，在optimizer前fail closed。根因是旧DDP只包HF Qwen，
  critic消费的forward-hook hidden不在DDP返回值中；修复必须包住直接返回
  `BackboneOutput.hidden`的Backbone forward。ID128无optimizer step/checkpoint、
  不可resume，不能通过放宽容差绕过。
- ID129 Job`506846`证明只把DDP移到Backbone返回边界仍不足：新包装确已
  进入`model_parallel_ddp`，但首轮Qwen梯度差仍为`0.002227783203125`。
  PyTorch 2.8源码进一步显示`static_graph=True`不遍历返回图；planner路径改为
  `find_unused_parameters=True, static_graph=False`以显式跟踪hidden与unused
  `lm_head`，direct actor保留原静态logits DDP。ID129在optimizer前失败，无
  checkpoint且不可resume。
- ID130 Job`506862`在dynamic Backbone DDP下完成单卡真实backward和
  2-rank×2-GPU全4个PPO epoch/4次AdamW step；Qwen/ValueHead梯度与参数
  replica最大差均为0，ValueHead delta为`2.6123e-4`，epoch2--4 clip
  fraction为0.5。Qwen final-norm BF16 witness因LR`1e-6`未发生可表示参数
  变化，但非零梯度`0.00778198`已跨rank精确同步。运行期同时确认没有
  未使用的trainable Qwen参数，因此最终生产设置收敛为
  `find_unused_parameters=False, static_graph=False`并保留同一GPU梯度门禁。
