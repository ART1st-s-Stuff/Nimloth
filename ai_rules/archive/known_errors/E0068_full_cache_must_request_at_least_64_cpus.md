# E0068：全量cache不得只用少量CPU

## 已确认错误

ID53全量SFT2 preprocess cache job `495702`在拥有224个CPU core的`intel-01`上
只申请了8 cores，导致cache构建不必要地持续了2小时以上。

## 正确做法

- 今后启动全量cache任务时，单任务至少申请64 CPU cores，并令worker数与
  实际`SLURM_CPUS_PER_TASK`匹配。
- 提交后必须核对`scontrol`/`sacct`的`ReqTRES`与`AllocTRES`，不能只看节点
  物理core总数。
- 若当前partition/QoS无法提供64 cores，应改用能满足资源要求的提交方式；
  仍无法满足时必须先告知人类，禁止默认降到8 cores。
