# E0041: `IDLE+PLANNED` GPU 不是立即可用资源

## 已发生错误

资源排障时只看节点当前`AllocTRES`，把dgx-27的`State=IDLE+PLANNED`、8张未分配GPU误读为可立即使用，并短暂提出8+1 replacement。

## 原因

`PLANNED`表示调度器已为未来任务预留该节点。此次`dgx-27`实际是他人job478950的`SchedNodeList`，不是ID19可抢占资源。另一个相关错误是只合计空闲GPU，没有先检查heterogeneous job每个component所需的独立physical fragment是否同时可满足。

## 正确做法

- 查询资源时同时检查`State`；`PLANNED`不得计入立即可用GPU。
- 对`PLANNED`节点查询pending jobs的`StartTime`和`SchedNodeList`，确认预留归属。
- heterogeneous job按component逐项核对节点、GPU、CPU和主机内存，不得只看集群空闲GPU总数。
- replacement前继续按E0026即时复核旧job状态，并用`scancel --state=PENDING`避免误取消已启动任务。
