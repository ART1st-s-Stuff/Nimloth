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

## 2026-09-19 Stage2 data preflight follow-up

- 远端 canary 暴露输入接线错误：Stage2 query alignment 收到了
  `record_format=nimloth_trajectory_v1`，旧入口直到8 rank加载模型后访问首条记录才因缺少
  `messages` 失败。
- 新增 CPU-only answer-view preflight，并在 `setup_dist()`、processor及模型加载之前同时
  检查 train/validation；错误明确包含 split、record index、文件路径与收到的 record
  format。正确 answer-view 继续复用既有 observation/CoT 语义检查。
- 独立检查补齐坏 JSONL 的物理行/record 定位、空集、实际 `max_records`/
  `max_images_per_record` 范围及多模态 part 结构回归；新增 preflight 测试现为
  `8 passed`。preflight 与相邻 query-alignment 最终回归
  `37 passed, 1 deselected`（缺少可选 `peft`），此前 Stage1 continuation、Stage2
  convergence/query alignment 回归 `51 passed, 1 deselected`；Stage2 multiprocessing
  回归在允许 IPC 的环境中 `3 passed`。Ruff、`compileall` 与 `git diff --check` 通过。
- 全 Stage2 邻接测试另有4个与本补丁无关的既有失败：cache reuse fixture 未接收
  `include_cls`、full-tuning fixture 缺 `config`、一个测试缺可选 `peft`，以及本地
  Transformers API 与测试预期的 `visual` 属性不一致；本次不扩大范围修改。
  未修改实验超参、CLS/WM实现或远端产物。

## 2026-09-19 Stage2 CLS canary r2

- 使用实现提交 `4e1f840c`（远端记录 HEAD `1913f779`）、epoch16、原 Stage2
  answer-view 1709/193 条数据和 K65 cache `199285c3bf8d22d8` 启动8卡 DDP 单步
  canary；CPU preflight 对真实 train/validation 均通过。
- 运行在第一次前向、optimizer update 前以 exit 1 失败；全部 rank 的文本
  FlashAttention 收到 FP32 hidden。无有效 checkpoint，仅留下日志表头，GPU 已全部释放。
- 只读探针确认当前 Transformers 4.49 对该 Qwen composite config 的
  `torch_dtype=bfloat16` 仅使视觉模块成为 BF16；epoch16 的文本 embedding、attention、
  norm 仍按 checkpoint config 保持 FP32。旧 FSDP 路径会在前向转 BF16，新的 DDP
  单行模式没有该 wrapper。
- 修复限定于 `global_query_only`：在安装新行之前只将冻结参数转为 BF16，保留 rotary
  等 FP32 buffer；随后新增 Query 行仍建立 FP32 master，forward hook 输出 BF16。
  同时移除该模式下错误的“full fine-tuning”提示。重跑前须通过 focused tests、远端
  dtype 探针，并使用新的唯一输出目录。
