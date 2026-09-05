# E0084: planner 多 DDP wrapper 的 collective 顺序必须显式对齐

## 现象

同一批 planner transition 在多个 rank 上具有不同长度的完整 Qwen prefix。快 rank
已进入 ValueHead backward 的 `ALLREDUCE`，慢 rank 仍在另一个 DDP wrapper 的
forward buffer `BROADCAST`；相同 NCCL sequence number 因 collective 类型不同而在
watchdog timeout 后失败。

## 已知机制与根因边界

planner 的 Backbone、WM predictor 和 ValueHead 分别由 DDP 包装，并共享默认 process
group。ID134/ID136曾显示不同rank在相同NCCL sequence进入one-element `BROADCAST`和
ValueHead `ALLREDUCE`，当时被误判为正常训练路径的独立collective-order根因。

ID137保留rank-local原始异常后，rank1/rank2先在Qwen language MLP forward发生CUDA OOM，
其余rank才进入监控barrier或被torchrun终止。由此可知先前的one-element broadcast属于
rank-local训练失败后的异常清理分叉，会掩盖原始错误；当前没有证据证明健康forward/
backward路径仍存在独立ValueHead/DDP collective-order bug。

## 防复发

- 分布式异常路径不得再进入要求所有rank参与的fresh-rollout abort/broadcast；必须先记录
  rank、异常类型和原始错误，并让consumption fail closed保持`in_progress`。
- `broadcast_buffers=False`和逐transition backward后的barrier继续作为当前wrapper配置约束，
  但不能单独证明健康路径。production-shaped门禁仍需完成至少一个真实、finite的PPO critic
  epoch和optimizer step，才能证明当前collective序列。
- 遇到one-element broadcast/allreduce分叉时，必须先查找更早的rank-local CUDA、processor、
  输入或控制流异常；不得把watchdog末端的collective mismatch直接当作首因。

## 证据边界

ID134 Job `507599`完整提交了前15次更新；第16轮319条 transition 的 processor 复现显示
四个 rank 最大 state token 为14441/16005/16178/14268，均未超过16384，因此该次失败
不是 token budget 超限。NCCL在sequence 6099记录 rank0/3 的1057800-element
ValueHead `ALLREDUCE`与rank1/2的1-element `BROADCAST`冲突。

ID136使用exact runtime commit`d6197e84`，其中已知planner DDP wrapper均设置
`broadcast_buffers=False`且训练循环每条transition backward后执行barrier。正式rollout完整
合并16条/319 transitions后，首个PPO critic epoch仍在sequence6046记录rank0/3的
1,057,800-element ValueHead `ALLREDUCE`与rank1/2的one-element `BROADCAST`，600秒后watchdog
失败；没有optimizer step或checkpoint。

ID137在commit`c3215592`移除分布式异常清理collective后，同形状16条/319 transitions运行
直接暴露rank1/rank2的Qwen activation OOM：分别在GPU3/GPU5的language MLP forward失败，
均早于optimizer。rank0随后才报告failed peer。该证据把ID136末端collective mismatch降级为
次生异常路径，同时把后续门禁的首要问题转为E0086记录的实际梯度检查点失效。
