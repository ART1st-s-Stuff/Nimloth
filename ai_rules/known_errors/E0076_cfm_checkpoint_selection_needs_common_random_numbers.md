# E0076：CFM checkpoint selection 必须使用同一组随机数

## 已确认现象

ID50 CFM 每隔1,000 step用validation flow MSE选择`best.pt`，但每次把当前step加入
validation seed。不同checkpoint因此使用不同的noise和flow time，数值不能直接排名。
step 10,000被选为best后，正式matched-noise reconstruction反而比旧decoder更差；该结果
不能证明step 10,000是本次训练轨迹中视觉最好的checkpoint。

## 正确做法

- 同一次训练的所有checkpoint validation必须使用固定subset和固定noise/time seed。
- checkpoint选择合同应把validation seed写入metadata和resume invariants。
- flow MSE与最终reconstruction指标属于不同统计量；正式视觉目标失败时，应在同一held-out
  reconstruction协议下冻结扫描已有checkpoint，不能直接把最低的异种随机flow MSE当作成功。
- 修复位置：`src/nimloth/training/reconstruction/cfm_sft2.py`；ID50历史checkpoint使用冻结
  diverse40 matched-noise sweep做事后排名，不改写原训练结果。
