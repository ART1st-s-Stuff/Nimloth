# 评估 SFT1 epoch 5/6 测试集成功率

## Goal

用统一 Stage 1 评估入口比较 epoch 5 与 epoch 6 在固定 held-out test 集上的 success rate，为最终 checkpoint 选择提供真实环境证据。

## Confirmed inputs

- 训练已按人类要求人工停止；没有 `CONVERGED` 声明。
- epoch 5 与 epoch 6 均有完整 `COMMITTED` adapter 和训练状态，目标为 `format_answer_ce_v2`，动作编号 token 权重为 8，范围为 `action_number_tokens_v1`。
- 两个 checkpoint 分别对应完整验证 LM loss 0.4325086 与 0.4255139；训练期 128-token 格式检查均为 31/32。
- 评估源码为远端干净 worktree commit `2a8ac7122c8e1f389e01cd360cb592cb9d308d80`；环境服务使用 `/mnt/nimloth/sources/vagen` commit `844378ce8a5727d8274b0c7024573031f9b1296d`。

## Requirements

1. 每个 adapter 必须通过正式 `nimloth.training.sft.stage1.checkpoint_export` 导出完整 HF checkpoint，不直接把 adapter 交给 vLLM。
2. 每个导出 checkpoint 先由统一入口执行固定 32 条严格无约束格式门禁；至少 31/32 才进入环境 rollout。
3. 每个 checkpoint 使用相同的 Base 60 + Common Sense 60 test episodes、seed offset 1、最多 20 步和确定性生成，分别保存独立的 120-episode 原子记录与 summary。
4. prompt 使用 Stage 1 action-token 转换；环境使用原 VAGEN BatchEnvironmentServer，不加载 WM/value/MCTS。
5. 两组使用独立环境进程、端口、AI2-THOR HOME、policy GPU 和输出目录，可并行运行但不得共享有状态环境实例。
6. 只以 `metrics.traj_metrics.success` 统计 success；不足 120 条或门禁失败必须报告为部分/阻塞结果。

## Acceptance Criteria

- [ ] epoch 5 和 epoch 6 的完整 HF 导出都通过 stage/objective/action-weight/shard 校验。
- [ ] 两组各保存 32 条严格格式门禁原文、token、EOS/finish reason 与 parser 结果。
- [ ] 门禁通过的 checkpoint 各完成 Base 60 + Common Sense 60 共 120 个 episode。
- [ ] 分别报告 overall、Base 和 Common Sense 的 completed、successes 和 success rate，并给出输出路径。
- [ ] 日志无未解释的 OOM、NaN、traceback、环境 API 错配或不完整 summary。

## Out of Scope

- 不重新训练、修改 checkpoint、补造失败 episode 或更换测试集。
- 不把训练期格式通过率或 validation LM loss称为 success rate。
- 不把少于 120 条的部分结果作为标准测试集结论。

## 2026-09-12 confirmed continuation (supersedes earlier no-training scope)

User requests continuation from completed epoch7. Confirmed action-number tokens, action_start, action_end and tokenizer EOS all weight16; every other supervised answer token stays1. Retain BF16 FSDP LoRA, existing full embedding/head training, data/cache/prompt and learning rates. No full freeze, row-freeze, text mask, query/WM/RL change. New weighted objective must have explicit checkpoint identity; old run is preserved. Continue to the previously approved validation-LM-loss convergence policy (minimum2 epochs; two consecutive relative improvements below1%), not a fixed one-epoch budget. Loss change must not silently import stale convergence history or bypass resume guards.

Acceptance: exact token-weight and gradient tests, default compatibility, checkpoint/export/eval compatibility, safe epoch7 continuation with new output identity and visible data/optimizer/scheduler/RNG semantics, committed remote source, real forward/backward progress and monitoring. Scope includes implementation, tests, spec pseudocode alignment, and normal Git synchronization to the existing experiment worktree on a100-1.

## Latest user correction: existing resume only

User explicitly forbids a new entry and requests replacing weight-change rejection with warnings. This supersedes the prior separate-run/explicit-continuation design above. Use original workflow --resume with CLI action16 and boundary16 in the original run, loading epoch7 and preserving optimizer/scheduler/RNG/data cursor AND unweighted-LM convergence state. Only recognized loss-weight changes are warnings; stage/data/model/other hyperparameter mismatches remain errors. No new continuation option, phase offset, optimizer reset, new model initialization, or edited training_state. New checkpoints record actual weight identity; existing epoch checkpoints stay immutable. Workflow records the accepted parameter transition.

User additionally requires weighted validation loss as comparison: log validation_weighted_lm_loss with the exact training weights alongside validation_lm_loss using the same forward. Convergence/best remain unweighted. User approved the displayed minimal spec pseudocode diff and instructed continuation after tests.

## 2026-09-12 current user scope: original VAGEN test128 paired comparison
Supersedes earlier fixed120/greedy scope for this new comparison only. User accepts original test.parquet task overlap with train and explicitly requests epoch18 on original128 vs VAGEN. Run two frozen arms with identical ordered parquet identities (Base64/CommonSense64), original per-row environment settings, current source VAGEN844378c and exact checkpoints already verified. Existing run.sh gains explicit manifest argument; no new entrypoint. Preserve prior120 outputs and pausedtraining351. Shared decoding temp0.7/top_p0.95,max256,seed0,20steps/history5; both same. Report original128 task reproduction, not independent heldout generalization. Stage1 format diagnostic warning remains nonblocking under previously accepted change. Acceptance: exact128 each, validated manifest/hash/identity/environment binding, separate summaries, paired success/action/format comparison. Tests must verify arbitraryseed order, duplicate rejection, perrowconfig and resume identity drift rejection.
