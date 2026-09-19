# Progress

## 2026-09-19 planning

- 用户批准 evaluation-only K64+CLS Stage2/Stage3 实验方案：真实 DINO CLS、固定二维
  sine-cosine 位置、从 Stage2 epoch16 只训练新增 CLS input embedding row、Stage3
  ValueHead/OutcomeHead 照常训练但只读取 K64；主要验收为 DINO、WM 与 reconstruction。
- `prd.md`、`design.md`、`implement.md` 与 implement/check manifests 已完成并通过
  `task.py validate`。实现基点为 `codex/stage3-residual-rl` commit
  `38a797cc5d7e02adeeef776f64bcf16e03d4f3b4`。

## 2026-09-19 implementation and local check

- 专用 branch/worktree：`codex/stage3-cls-fixed-2d-position`，
  `/workspace/remote2/nimloth/.worktree/stage3-cls-fixed-2d-position`。
- 实现真实 DINO CLS + K64 cache v2、显式 `GridStateLayout`、global Query prompt/token与
  input-only FP32 selected row、evaluation-only Stage2、K65 fixed-2D residual Stage3、
  spatial/CLS 分项 WM/DINO loss、K64-only Value/Outcome/SIGReg/Planner compatibility、
  K65 feature export 与 frozen K64 CFM evaluation/identity gate。
- 独立检查直接修复：v2 cache 必须分开保存 `spatial_features`/`cls_features`并拒绝 proxy；
  空 protocol selected-row forward；K64/K65 spatial comparator 和多K65 CLS一致性；
  reconstruction copy baseline；K65 ordering/loss checkpoint metadata；cache reuse teacher
  provenance；distributed loss与manifest测试fixture。
- 本地验证：核心受影响文件 Ruff 通过，`compileall`、AST parse、`git diff --check`通过；
  focused tests 100 passed、2 deselected。跳过项分别依赖本地缺失的 `peft`，以及当前沙盒
  无法解析 loopback 的 Gloo 测试。未配置静态 type checker；changed-file Ruff 的18项为
  既存 style（主要是 `LatentActionTokens()` 默认参数），未跨任务重构。
- 提交前配置审计发现新 YAML 沿用了旧模板参数，已校正为近期 Outcome+BCE baseline：
  5 epochs、固定 schedule 46 steps、Qwen `2e-7`、projector `8e-6`、Query `1e-5`、
  protocol rows `2e-6`、WM `3e-4`、Value/Outcome `1e-4`、max length 16384；配置测试
  固定这些值，避免 evaluation-only 架构变更同时混入旧学习率。
- 实现提交：`f91e95b9c4c49ba8f6c2e26f497ea18d0b86180e`。尚未启动远端/GPU实验。
  下一门禁：远端完整依赖回归；真实 v2 cache
  lineage 核验；production-shaped 单步 GPU canary；epoch16/aligned Stage2 observed K64与
  frozen reconstruction identity；随后展示正式长训练 launch contract 并取得单独批准。
