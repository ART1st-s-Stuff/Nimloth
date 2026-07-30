# E0075：parent评估必须在占GPU前检查W&B凭据

## 已确认现象

SFT1 parent评估完成300条trajectory并写出`summary.json status=ALL_OK`后，finalizer才调用
W&B；batch环境没有加载项目`.env`，因`WANDB_API_KEY`缺失退出。核心结果有效，但Slurm
状态失败且没有`done.flag`，浪费了完整GPU评估后的收尾机会。

## 正确做法

- parent评估脚本在任何render/model工作前加载服务器项目`.env`并要求`WANDB_API_KEY`存在。
- W&B上传失败不得导致重跑已通过严格300条门禁的rollout；应从现有原子输出重新运行
  finalizer，并记录post-hoc恢复。
- `summary.json`证明本地结果合同通过；`done.flag`还要求W&B finalizer完整结束。
