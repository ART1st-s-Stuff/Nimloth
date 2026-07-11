# E0016：不能把 validation 的 greedy 参数用于训练数据 rollout

## 已发生的错误

源 run 的 validation 使用 `do_sample=false, temperature=0, top_p=1`，训练 rollout 使用 `do_sample=true, temperature=0.7, top_p=0.95`。本次错误地把 validation parity 的 greedy 参数用于 production train rollout，已采集的960条 train记录成功率仅11.04%。

## 原因

核对环境一致性时混淆了“验证复现”和“训练数据采集”两个调用场景；没有在启动清单中分别固定 train rollout 与 val/test generation kwargs，并把错误的 greedy 设置写入了 production wrapper和进度文档。

## 正确做法

启动数据采集前必须分别从源配置核对并记录 train rollout 与 val/test generation kwargs。训练数据采集复现源 train sampling；greedy参数只用于 validation/test。任何跨场景复用都必须由人类明确批准。使用错误 sampling 产生的数据必须隔离，不能进入转换或训练。

## 证据

- 源训练配置与日志：`.../vagen_legacy_wm_entropy01_kl001_60step_2env4train/`
- 本任务进度：`ai_tasks/ai_progress/2026-07-09_vagen_legacy_wm_k8_prep.md`
- production wrapper：`experiments/training/sft1/rollouts_greedy_parallel.slurm`
