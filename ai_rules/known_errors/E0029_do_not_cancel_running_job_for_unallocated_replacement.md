# E0029：不要为尚未获得的 replacement 取消正在运行的任务

## 已发生的错误

preprojection query cache job `475052` 已在 normal/dgx-29 使用 2 GPU 正常运行。看到 dgx-22 当时显示 4 张空闲 GPU 后，agent 先取消了 `475052`，再提交固定 dgx-22 的 4-GPU job。新 job 没有获得 allocation，原本已获得的 normal 资源和约 5 分钟进度被白白丢失。

## 原因

把“节点当前显示空闲”错误地当成“replacement 已获得资源”。Slurm 空闲快照不保证低优先级新 job 会立即调度；在提交与调度之间，资源可能被更高优先级任务占用。

## 正确做法

- 正常运行且无错误的任务应继续运行，不为投机性加速主动取消。
- 若确需切换资源，先获得独立 hold allocation 并确认其为 `RUNNING`，再停止旧任务并在 hold 内启动 replacement。
- 若两个任务会写同一输出目录，禁止通过同时提交 workload 来“抢跑”；只能先占资源，不启动第二个 writer。
- 只有旧任务本身错误、用户明确要求停止，或 replacement allocation 已经确认时，才能取消正在运行的任务。

## 再次发生（2026-07-15）

fragmented SFT2 candidate `476457` 最后一次查询时仍是 pending；agent 随后为了让
world5 candidate 调度而直接执行 `scancel`。在查询和取消之间，`476457` 已经获得
allocation 并运行约 18 秒。它尚未产生 train step/checkpoint，但这仍然是使用 stale
状态取消 running allocation。

以后取消 pending candidate 必须在同一个远程 shell 中立即读取状态，并且仅当读取值
仍严格等于 `PENDING` 时执行 `scancel`。如果已经变成 `RUNNING`，必须按本文件的
健康门禁处理。

## 证据

- `AI_branch_progress.md` 中 preprojection cache 与 fragmented SFT2 调度记录。
- 服务器实验 README：`.../query_state_ablation/10_preproj_vs_projected_k8_all3217_steps18560_b16_h256d4/README.md`。
- Slurm `476457`：三个 heterogeneous components 均显示约 18–19 秒后 cancelled。
