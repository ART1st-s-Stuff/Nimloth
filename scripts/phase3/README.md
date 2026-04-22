# Phase 3 最小实验手册

本目录用于执行 Phase 3（VLM 接地与语义状态对齐）的最小回归流程。

## 脚本列表

- `semantic_align_train.sh`：训练语义对齐模型（InfoNCE + 时序一致性）
- `semantic_align_eval.sh`：评估同意图/异意图相似度与时序平滑性
- `export_pm_ready_features.sh`：导出 Phase 4 可消费的 `state={z_t,s_t,env_context}`
- `run_phase3_minimal.sh`：按“训练 -> 评估 -> 导出”顺序一键执行

## 默认前置条件

- 已完成 Phase 1 数据采集，且存在 `manifest.jsonl`
- 当前 `wm` 配置可构建图像编码器（如 `cfm_dinov2m`）
- 训练参数来自 `configs/pipeline/train/default.yaml` 的 `semantic_align` 段

## 一键运行

```bash
bash scripts/phase3/run_phase3_minimal.sh
```

## 分步运行

```bash
bash scripts/phase3/semantic_align_train.sh
bash scripts/phase3/semantic_align_eval.sh
bash scripts/phase3/export_pm_ready_features.sh
```

## 常用覆盖参数示例

1) 指定 manifest：

```bash
WM_MANIFEST_PATH=datasets/ai2thor/2026-04-22_12-00-00/manifest.jsonl \
bash scripts/phase3/semantic_align_train.sh
```

2) 指定评估 checkpoint：

```bash
SEM_ALIGN_CKPT_PATH=models/semantic_align/qwen_vl/2026-04-22_13-00-00/semantic_projector.pt \
bash scripts/phase3/semantic_align_eval.sh
```

3) 覆盖训练超参（Hydra）：

```bash
bash scripts/phase3/semantic_align_train.sh \
  pipeline.train.semantic_align.batch_size=4 \
  pipeline.train.semantic_align.positive_k=2 \
  pipeline.train.semantic_align.negative_gap=8 \
  pipeline.train.semantic_align.temporal_weight=0.3
```

## 结果产物位置

- 训练：`models/semantic_align/qwen_vl/<run>/`
- 评估：`models/semantic_align_eval/qwen_vl/<run>/`
- 导出：`models/pm_ready_features/qwen_vl/<run>/`

其中导出目录包含：

- `pm_ready_features.json`
- `pm_ready_contract.md`
