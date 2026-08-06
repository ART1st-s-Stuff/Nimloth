# E0084: planner 多 DDP wrapper 的 collective 顺序必须显式对齐

## 现象

同一批 planner transition 在多个 rank 上具有不同长度的完整 Qwen prefix。快 rank
已进入 ValueHead backward 的 `ALLREDUCE`，慢 rank 仍在另一个 DDP wrapper 的
forward buffer `BROADCAST`；相同 NCCL sequence number 因 collective 类型不同而在
watchdog timeout 后失败。

## 已知机制与未决根因

planner 的 Backbone、WM predictor 和 ValueHead 分别由 DDP 包装，并共享默认 process
group。ID134证明不同rank能在同一sequence进入forward侧的one-element `BROADCAST`和
ValueHead backward `ALLREDUCE`。当时将该broadcast归因于DDP buffer同步，但ID136已在所有
已知planner wrapper使用`broadcast_buffers=False`、并在每条transition backward后barrier的
版本上复现同类分叉。因此此前归因不完整；剩余one-element broadcast的精确发起位置和
rank间控制流边界仍未定位。

## 防复发

- `broadcast_buffers=False`和逐transition backward后的barrier可以保留为已验证的配置约束，
  但不得再声称它们已经修复此错误。barrier发生在backward之后，无法单独证明此前forward/
  backward内部发起的collective顺序已对齐。
- 在定位one-element `BROADCAST`的实际调用栈、确认所有rank在首条transition内的collective
  序列一致前，禁止再次直接启动全规模训练。
- 修复必须先通过production-shaped多rank GPU门禁：使用真实planner wrappers、不同rank的
  可变prefix长度、至少一个完整PPO critic epoch，并检查collective序列、finite loss和完整
  optimizer step。CPU/Gloo、参数断言、forward/backward计数或同步次数回归只能作为补充，
  不能替代该门禁。

## 证据边界

ID134 Job `507599`完整提交了前15次更新；第16轮319条 transition 的 processor 复现显示
四个 rank 最大 state token 为14441/16005/16178/14268，均未超过16384，因此该次失败
不是 token budget 超限。NCCL在sequence 6099记录 rank0/3 的1057800-element
ValueHead `ALLREDUCE`与rank1/2的1-element `BROADCAST`冲突。

ID136使用exact runtime commit`d6197e84`，其中已知planner DDP wrapper均设置
`broadcast_buffers=False`且训练循环每条transition backward后执行barrier。正式rollout完整
合并16条/319 transitions后，首个PPO critic epoch仍在sequence6046记录rank0/3的
1,057,800-element ValueHead `ALLREDUCE`与rank1/2的one-element `BROADCAST`，600秒后watchdog
失败；没有optimizer step或checkpoint。该证据将旧防复发结论降级为“不充分”，但尚不足以
断言新的唯一根因。
