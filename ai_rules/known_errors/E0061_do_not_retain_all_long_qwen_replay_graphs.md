# E0061：禁止在一次RL backward前保留全部长Qwen replay计算图

## 错误

把一个RL batch中的所有history step依次执行Qwen token replay，把每次forward产生的
可微log-prob和token hidden都保留到最后，再执行一次整体backward。

## 为什么错

真实20-step history中的prompt会持续增长，CoT也可能达到配置的完整response token
上限。即使每次Qwen forward只处理一个turn，只要多个forward的activation graph一直
保留到同一次backward，显存仍会按replay turn累积。ID108在两份双卡Qwen副本上都把
第二卡推到约77.95 GiB allocated，并在再申请74 MiB时OOM。

## 正确做法

- planner训练以相邻Qwen anchor之间的实际执行segment为TD step。每个TD step只重放一个
  anchor response和该段WM dynamics，立即backward并释放activation。
- TD loss按整个episode batch的segment总数缩放；ValueHead在所有TD backward之后，使用
  detached全episode state和MC return单独backward。所有backward结束后只执行一次
  optimizer step，因此参数在rollout batch内保持不变。
- 多rank同步继续使用PyTorch官方DDP/FSDP。TD与MC forward使用不同参数子集时，使用官方
  unused-parameter机制；禁止恢复手工gradient all-reduce。
- 不能通过缩短真实CoT、减少`history_size`或事后改写rollout token来伪造通过。

## 验证

先在CPU小模型上验证TD只更新StateProjector/WM/Qwen action路径、MC只更新ValueHead、
完整segment覆盖和一次optimizer step；再使用新采集的真实20-step rollout执行GPU门禁，要求finite
component loss、两rank完成backward/step、无OOM/NCCL hang，并验证trainable/frozen
参数边界和完整checkpoint。
