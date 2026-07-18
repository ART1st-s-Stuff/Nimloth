# E0047: Full-response mask 不等于 VAGEN credit 对齐

## 已发生错误

曾把“所有sampled thought/action token都进入PPO loss”描述为完整VAGEN PPO，却仍用每turn action ValueHead产生一个advantage并广播。VAGEN navigation实际可配置advantage estimator；被核对的`run.sh`选择`masked_gae`，由独立Qwen token critic为每个loss-mask token提供不同value和advantage。Full-response loss范围相同不能证明credit assignment相同。

## 正确做法

- 分别核对policy loss mask、reward placement、advantage estimator和critic粒度，禁止用其中一个代替全部协议。
- VAGEN对齐配置必须显式记录`adv_estimator`、gamma/lambda、reward placement和critic backend。
- `masked_gae`使用token-level values/returns/advantages及独立critic checkpoint；历史turn广播路径保留为明确命名的可选`turn_mc`，不得称为默认VAGEN navigation credit。
