# E0017：不能把一个 eval set 的成功率推广到其他数据分布

## 已发生的错误

把 base/common held-out eval 120条的71.67%表述成模型采集新样本的普遍成功率，并据此预期 `*_train` 和 long-horizon 数据也有相近表现。实际前320 seeds的train采集仅11.04%。

## 原因

只比较了generation参数，没有同时固定任务集合。held-out `base/common` 与 `base_train/common_sense_train/long_horizon_train` 是不同资产和难度分布；源checkpoint也未在long-horizon上训练。

## 正确做法

成功率只能在相同checkpoint、generation kwargs、环境语义、eval set、seed/task composition下比较。报告指标时必须带上数据集和类别，禁止把held-out结果简称为通用“成功率”。全量前应在实际目标split上做分层smoke。

## 证据

- 数据资产：`external/VAGEN/vagen/env/navigation/datasets/`
- 本任务进度：`ai_tasks/ai_progress/2026-07-09_vagen_legacy_wm_k8_prep.md`
