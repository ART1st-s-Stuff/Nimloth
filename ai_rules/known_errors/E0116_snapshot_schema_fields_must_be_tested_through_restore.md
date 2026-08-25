# E0116：snapshot schema字段必须经过真实restore路径测试

## 错误

K4 checkpoint restore实现把`FrozenK4PlannerTransport.snapshot_source_step`误写成
不存在的`source_step`属性。此前仅有checkpoint assembly与transport publication测试，
没有执行actor full-planner restore，因此静态检查和局部测试均未发现该字段名错误。

## 后果

fresh-runtime restore会在加载完planning module和optimizer后因`AttributeError`失败。
错误在任何production/canary运行前由新增的tiny full-planner round-trip测试发现；没有启动
GPU实验，也没有产生或发布错误snapshot。

## 正确做法

- identity字段只能通过schema dataclass的真实属性访问，不复制近似字段名。
- snapshot/checkpoint功能必须测试完整的`export -> persist -> fresh object restore -> fingerprint`
  路径；只测assembly或mapping结构不够。
- restore测试必须覆盖active transport文件内容、source step、contract、score dtype和snapshot
  fingerprint的一致性。
