# E0016：必须按人类指定的 train-time eval 参数采集新样本

## 已发生的错误

人类要求用源训练过程中执行 evaluation 的生成参数采集所有新样本。源 run 同时存在 optimization rollout sampling 和 train-time evaluation 两套参数；agent 错误地把“train数据”理解为应改用 optimization rollout 的 `temperature=0.7`，并据此给出错误结论。

## 原因

只按数据 split 猜测 generation 场景，没有按人类指定的调用语义定位源配置字段。这里应读取 `actor_rollout_ref.rollout.val_kwargs`，而不是 top-level training rollout sampling kwargs。

## 正确做法

所有新样本使用源 train-time evaluation 参数：`do_sample=false`、`temperature=0`、`top_p=1.0`、`top_k=-1`、`n=1`；每轮最多512 tokens、20 turns、每轮1个action。optimization rollout 的 `temperature=0.7, top_p=0.95` 与本任务无关，不得擅自替换。启动前必须把这些字段显式写入命令并核对实际 runtime kwargs。

## 证据

- 源 run 日志：`.../vagen_legacy_wm_entropy01_kl001_60step_2env4train/vagen_legacy_wm_entropy01_kl001_60step_2env4train.log` 中 `actor_rollout_ref.rollout.val_kwargs`
- 源 validation YAML：`.../resume_459939_control/tmpcfg.YMBucF/val.yaml`
- production wrapper：`experiments/training/sft1/rollouts_greedy_parallel.slurm`
