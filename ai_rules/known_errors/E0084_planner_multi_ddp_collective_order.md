# E0084: planner 多 DDP wrapper 的 collective 顺序必须显式对齐

## 现象

同一批 planner transition 在多个 rank 上具有不同长度的完整 Qwen prefix。快 rank
已进入 ValueHead backward 的 `ALLREDUCE`，慢 rank 仍在另一个 DDP wrapper 的
forward buffer `BROADCAST`；相同 NCCL sequence number 因 collective 类型不同而在
watchdog timeout 后失败。

## 原因

planner 的 Backbone、WM predictor 和 ValueHead 分别由 DDP 包装，并共享默认 process
group。逐 forward buffer broadcast 与多个 wrapper 的 backward all-reduce 之间没有
逐 transition 的共同边界；rank 间可变的 prefix 计算时间会改变这些异步 collective
进入共享 process group 的先后顺序。

## 防复发

- 对没有训练期可变 buffer、且 wrap 前已显式同步 state 的 DDP 模块设置
  `broadcast_buffers=False`。
- planner 每条 transition 的 backward 后执行分布式 barrier，再进入下一条 forward。
- 回归必须同时断言 DDP 参数和逐 transition 同步次数；只验证每个 rank 的 forward 数量
  相同仍不足以证明 shared-process-group collective 顺序一致。

## 证据边界

ID134 Job `507599`完整提交了前15次更新；第16轮319条 transition 的 processor 复现显示
四个 rank 最大 state token 为14441/16005/16178/14268，均未超过16384，因此该次失败
不是 token budget 超限。NCCL在sequence 6099记录 rank0/3 的1057800-element
ValueHead `ALLREDUCE`与rank1/2的1-element `BROADCAST`冲突。
