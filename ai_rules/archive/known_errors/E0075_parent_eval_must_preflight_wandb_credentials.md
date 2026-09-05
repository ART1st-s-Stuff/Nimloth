# E0075：parent评估必须在占GPU前检查W&B凭据

## 已确认现象

SFT1 parent评估完成300条trajectory并写出`summary.json status=ALL_OK`后，finalizer才调用
W&B；batch环境没有加载项目`.env`，因`WANDB_API_KEY`缺失退出。核心结果有效，但Slurm
状态失败且没有`done.flag`，浪费了完整GPU评估后的收尾机会。

## 正确做法

- parent评估脚本在任何render/model工作前加载服务器项目`.env`并要求`WANDB_API_KEY`存在。
- 加载`.env`前先保存本次显式W&B entity/project/run name/run ID，加载secret后恢复这些值；
  项目级默认值不得覆盖实验identity。
- W&B上传失败不得导致重跑已通过严格300条门禁的rollout；应从现有原子输出重新运行
  finalizer，并记录post-hoc恢复。
- `summary.json`证明本地结果合同通过；`done.flag`还要求W&B finalizer完整结束。

## 2026-07-30补充

首次凭据修复直接source `.env`，将显式`WANDB_PROJECT=nimloth-sft1`覆盖为`flower`。job
`498102`在26秒render probe阶段被主动取消，尚未创建W&B run或开始rollout。此后必须同时
检查controller记录的完整W&B identity，不能只检查API key存在。
