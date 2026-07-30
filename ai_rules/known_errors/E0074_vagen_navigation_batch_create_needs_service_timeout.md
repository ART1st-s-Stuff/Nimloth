# E0074：VAGEN navigation批量创建环境不能使用120秒短超时

## 已确认现象

VAGEN正式validation的首个`val_batch_size=24` micro-batch在单GPU navigation service上
创建24个AI2-THOR环境。服务端已经持续完成初始化，但客户端在120秒时先触发
`requests.exceptions.ReadTimeout`，使整个val-only job在任何trajectory落盘前退出。

## 正确做法

- 使用VAGEN官方navigation脚本的`rollout_manager.timeout=500`；基础trainer默认值1200秒。
- timeout只控制service HTTP等待，不改变batch、seed、模型输入、采样或环境动力学。
- 不得把批量环境初始化期间的`Initialize return`数量误报成已完成trajectory；必须等到
  rollout loop开始并最终生成严格validation dump。
