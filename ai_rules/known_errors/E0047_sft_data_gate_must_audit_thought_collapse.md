# E0047: SFT数据门禁必须审计thought塌缩

## 已发生错误

k8 SFT1数据只检查了schema、图片、动作有效性、split和成功率，没有检查thought多样性与动作平衡。实际训练集`train_success.jsonl`的7309个turn中，`Move forward.`占58.20%，前8个短动作复述占98.74%，95.57%的完整`<think>...</think>`恰为8个BPE token；`look_up`训练样本为0。ID24随后稳定生成`<think>Move forward.</think>`。

源run并非起点即塌缩：rollout step1的有效thought中位数30 tokens、top1占1.76%、exact action-copy为0；到step60变为中位数8、top1占56.54%、exact action-copy75.22%。SFT1从step60 greedy采集，继承并进一步集中该分布。converter大体忠实保留源thought，因此不是主要根因。

## 正确做法

- SFT采集前必须跨checkpoint审计thought长度、unique/top-k占比、action-copy比例及每类action覆盖。
- schema正确、action有效或成功率高不能替代thought质量门禁。
- 已塌缩teacher数据训练出的checkpoint只能作mechanics artifact，不得宣称reasoning/world-model policy质量。
- 数据重建前停止以该actor继续正式RL；先确定thought目标与token credit协议，再选择未塌缩source或重新生成teacher数据。
