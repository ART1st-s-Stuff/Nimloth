# SFT2 modularity refactor

## Goal

Refactor `nimloth.training.sft2` around explicit configuration, data, forward,
objective, evaluation, and checkpoint boundaries while preserving the verified
per-prefix training semantics. Remove the non-equivalent packed-forward path
from the production CLI and keep research-only trajectory code under
`training/sft2/diagnosis`.

After the SFT2 work is complete, perform a separate read-only audit of
`nimloth.training.rl` and report any recommended refinement.

## Plan

1. Add behavior-protection tests and correct model-independent validation
   success metrics.
2. Introduce typed SFT2 configuration and data contracts.
3. Split cache storage/build/collation and consolidate trajectory samplers.
4. Share one step runner between training and validation.
5. Clarify public artifact/checkpoint and outer-module dependency boundaries.
6. Run focused and repository-adjacent regression tests.
7. Audit `training/rl` without mixing RL changes into this refactor.

## Current state

- Worktree: `/workspace/remote2/nimloth-dev`
- Branch: `fix/sft2-review-bugs`
- Existing unrelated worktree changes preserved:
  `ai_rules/events/on_experiment_start.md` and `.until-done/`.
- No experiment or remote job is in scope.

## Changes

- Added a strict SFT2-owned YAML schema in `training/sft2/config.py`; unknown
  phase-specific fields now fail instead of being silently ignored by common
  config code.
- Split the data plane into `data/batch.py`, `data/samplers.py`,
  `data/factory.py`, and a focused `data/cache/` package. Model-independent
  transition records now come from `nimloth.wm`; Qwen message adaptation now
  comes from `nimloth.backbone.qwen25vl`.
- Added `components.py` for Qwen/head/DDP/EMA/optimizer construction and
  `engine.py` for the forward path shared by training and validation.
- Renamed tensor-level loss code to `objectives.py`; simplified the package
  initializer so it no longer re-exports Qwen/WM/common symbols.
- Removed packed/full-trajectory forward flags and launchers from production.
  The known non-equivalent implementations and cache are isolated under
  `training/sft2/diagnosis`.
- Removed static success-label metrics from SFT2 validation and checkpoint
  selection. `val_wm_mse` is the only production checkpoint metric; static
  dataset success prevalence is explicitly named under `wm.statistics`.
- Moved Qwen transition and checkpoint wrapper knowledge into
  `backbone/qwen25vl`, and removed reconstruction/evaluation imports of private
  SFT2 dataset code.
- Added package/experiment documentation describing dependency direction and
  production versus diagnosis boundaries.

## Verification

- `python -m compileall` passes for the changed SFT2, Qwen backbone, WM,
  experiment, and test modules.
- Related SFT2/common/backbone/WM/eval/recon/RL tests: 130 passed in the normal
  sandbox. The two-rank Gloo aggregation test also passed when run separately
  with loopback socket access, for 131 passing tests total.
- All five SFT2 YAML profiles pass strict schema parsing.
- RL read-only tests are included above: 19 passed (one intentional PyTorch
  single-sample std warning).

## Open decisions

- The SFT2 implementation decisions are resolved: packed-forward is diagnosis
  only, and train/eval runtime helpers may live under `utils.py`.
- The separate RL audit has identified correctness and modularity issues; the
  final report will prioritize them without changing RL in this refactor.

## 2026-07-21 SIGReg batch 语义修复

- 已确认原 `build_trajectory_sigreg_inputs` 把每条轨迹分别变成 `(T_i,1,D)`，
  但 SIGReg 实际沿 `B` 维估计分布，因此该实现不能检测跨样本表示坍缩。
- 人类进一步指出 `T` 不能取 rollout 的变长轨迹长度。LeWM 的 `T` 由
  `history_size + prediction_offset` 决定；SFT2 当前为一步上下文、一步预测，
  因此每个 transition 提供 `[s_t,s_{t+1}]`，固定构造 `(2,B,D)`。
- 新的 `nimloth.wm.OneStepSIGReg` 统一持有该契约；SFT2 algorithm 只筛选
  有效 transition，`B<2` 时保留 WM/CE/value 并显式记录跳过。validation
  不再计算随机 SIGReg。
- RL 的 `history_size` 必须继续可配置。本轮撤回了把 RL 限制为 1 的错误尝试；
  真正的多步 RL WM/SIGReg 需要单独实现连续上下文，暂不在本修复中落地。
- 提交 `7be6ba2`；本地静态编译与 `git diff --check` 通过。superpod SSH
  超过 60 秒未进入 shell，依服务器规则停止重试；远程定向/相邻回归待 VPN
  恢复后运行。
