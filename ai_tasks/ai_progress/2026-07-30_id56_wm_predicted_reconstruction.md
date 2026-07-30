# ID56 WM-predicted reconstruction

## 目标

使用 ID56 epoch2 的完整 Qwen、`state_proj` 和 H=1/T=4 WM checkpoint，在已有 diverse40
轨迹上生成真实 state 与自回归 WM-predicted state 的匹配噪声 reconstruction；旧 Qwen
ViT-token CFM 和旧 SFT1 DINO-grid CFM 分别作为正对照和 decoder-lineage 对照。

## 当前计划

1. 在独立实验 worktree 实现严格对齐、冻结加载、轨迹级 resume、指标与 contact sheet。
2. 本地静态检查；superpod 固定环境运行定向测试和真实 artifact CPU/GPU preflight。
3. 提交并同步 clean remote worktree，执行 on-experiment-start 完整门禁。
4. 请求 1 张 H800 运行 Euler50/CFG2/t+1...t+4 正式评估，监控至完成或失败。
5. 执行 on-experiment-end，核验 metadata、160 行样本、contact sheets 和 W&B。

## 已完成

- 建立 `exp/id56-wm-reconstruction` 和 sibling worktree。
- 核验 ID56 epoch2 checkpoint 完整，旧 DINO CFM 与 cache lineage 匹配；diverse40 的
  40 条 record 均存在于当前 val JSONL，前五个 action 全部一致。
- 数值核验旧 SFT1 与 ID56 projector 并非同一权重，因此保留旧 SFT1 DINO reconstruction
  列，并增加 ID56 actual-grid 列，避免把 projector/decoder distribution shift 误判为 WM error。
- 实现五列 evaluator、t+1...t+4 horizon 指标、matched noise、严格对齐检查、空 output
  保护、合同校验、轨迹级 state 编码原子恢复和 W&B 上传。
- 添加 diverse40 配置与定向单元测试。本地 `py_compile`、`git diff --check` 通过；本地
  Python 缺少 torch/pytest，因此功能测试待 superpod。

## 文件修改

- `src/nimloth/eval/dino_grid_wm_reconstruction.py`
- `configs/eval/reconstruction/id56_wm_predicted_diverse40.json`
- `tests/eval/test_dino_grid_wm_reconstruction.py`
- `AI_branch_progress.md`
- 本进度文件

## 验证与结果

- `PYTHONPYCACHEPREFIX=/tmp/id56_wm_reconstruction_pycache python -m py_compile ...`：通过。
- `git diff --check`：通过。
- 本地 `pytest`：未运行；环境没有 pytest，且系统 Python 没有 torch。这是环境边界，
  不能作为代码测试失败或通过的证据。

## 待完成

- superpod 定向 pytest、真实 cache/checkpoint loader 与 1-run 小样 preflight。
- 正式 on-experiment-start、Slurm/W&B/output identity 和 1-H800 运行。
- 完成后 on-experiment-end、输出/W&B/指标审计和进度归档。
