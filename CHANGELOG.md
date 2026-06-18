# Changelog

## 2026-06-18

- **LeWM 清理**：`wm/_vendor_lewm.py` 最小 vendoring；移除 pixel JEPA（`wm/model.py`、`pretrain_lewm_navigation.py`）；`LatentWMPredictor` 在 `wm/predictor.py`
- **Training / experiments 结构（第一轮）**：
  - WM 组件迁入 `wm/`（`state_proj`、`value_head`、`collate`）
  - 训练逻辑下沉 `training/sft2/`（`trainer.py`、`step.py`、`checkpoint.py`、`evaluate.py` 等）
  - `training/common/dist.py`、`qwen_batch.py`；`experiments/.../train.py` 薄入口
- **模块拆分（第二轮）**：
  - `backbone/`：`qwen_tuning.py`、`vision_ema.py`
  - `eval/rollout.py`：离线 `val_rollout_success_rate`；`training/sft2/metrics.py` 仅 batch 内指标
  - 测试：`tests/backbone/`、`tests/eval/`
- 文档：`wm/`、`backbone/`、`eval/`、`training/` README；`ai_tasks/sft2_phase2_plan.md`

## 2026-06-13

- 新增 `DESIGN_DOCS.md`：定义 Nimloth 的 world model / latent state / action prior 方案
- 新增 `ai_tasks/latent_action_extraction.md`：latent state 和 action prior 提取任务说明
- 新增 `ai_tasks/sft1_exp.md`：第一阶段 SFT 实验流程说明
- 新增 `experiments/navigation_baseline/`：VAGEN navigation baseline 的配置、脚本、说明文档
- 更新 `.gitignore`：忽略本地运行产物、缓存、`.ai2thor-home/`、`.cache/`、`.home/`、`.local/`、`runs/`、`*.out/*.err/*.pid`
- 更新项目规则文档：`ai_rules/03_experiments_and_data.md`、`ai_rules/04_code_and_repo.md`
- 子模块 `external/VAGEN`：提交 navigation prompt 与 env client 改动
- 子模块 `external/VAGEN/verl`：提交 rollout 并发限制与 SGLang/TCPStore 端口冲突修复
