# E0031 — 多候选 `--nodelist` 不能当成 Slurm 硬 allowlist

## 错误

在请求 `--nodes=2` 时提供多于两个节点的 `--nodelist`，并假定 Slurm 只能从这些候选节点中选择。

## 实际表现

2026-07-17 heterogeneous job 477542 的 group0 显示 `ReqNodeList=dgx-[18,21,32,34]`，但实际被分配到 `dgx-[14,21]`。其中 dgx-14 正被受保护的 job477349 使用。任务在9秒内被立即取消，尚未启动trainer或产生训练artifact。

## 正确做法

1. 保护正在运行的任务时，使用 group0 的 `--exclude=<protected nodes>` 作为硬边界，不要依赖较长的候选 `--nodelist`。
2. heterogeneous job 中，exclude不能包含后续group的显式nodelist节点；否则会触发E0029所记录的冲突。
3. 每次任务开始后立即检查每个heterogeneous group的实际NodeList。
4. 若分配到受保护节点，必须立即取消自己的任务，不能等待模型或环境初始化。

## 本次修复

k8 fragmented launcher的trainer group改为硬排除job477349的`dgx-[14,26,51,54]`；env group继续显式使用dgx-37。
