# E0070：Hydra structured config 的新增评估键必须显式使用 `+`

## 已确认现象

VAGEN parent评估在trainer启动时以`Could not override 'data.seed'`退出。其data schema没有
预定义`seed`、`base_seed`或`validation_shuffle`；直接写`data.seed=42`会在任何模型加载前
被Hydra拒绝。

## 正确做法

- 对schema中不存在的新键使用`+data.seed=42`等显式add override。
- 对已经存在的键直接override，禁止加`+`；若现有父键为`null`，应一次替换完整mapping，
  不要向`null`父键逐项追加子键。
- 提交昂贵任务前，用完整正式override集合运行Hydra `--cfg job` compose gate。
- compose失败不属于checkpoint、GPU或模型质量结果；不得靠删除seed合同绕过。
