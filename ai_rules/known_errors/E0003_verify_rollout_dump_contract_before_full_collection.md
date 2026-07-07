# E0003 — 全量采集 rollout 前必须验证 dump 能转换成训练数据

## 错误

曾在 Plan B retry15 中只检查 rollout JSONL 的行数和日志是否报错，就继续全量采集 train shards。后来才发现这些 JSONL 不能用于 SFT1/SFT2。

## 问题

当前 legacy-dev VAGEN 路径中：

- rollout manager 内存里有 `image_data`；
- 但 `external/VAGEN/vagen/trainer/ppo/ray_trainer.py::_dump_validation_records()` 写 JSONL 时执行：

```python
serializable = {k: v for k, v in item.items() if k != "image_data"}
```

这会把图像从落盘数据中丢掉。

旧 converter `experiments/training/sft1/convert_rollouts.py` 需要：

- 可解析的 assistant action；
- 与 `<image>` placeholder 对应的图片路径；
- 旧格式时还依赖 `image_0/images_<idx>/*.png`。

retry15 生成的原始 JSONL 只有 `output_str` 和 metrics，没有图片文件，导致首次转换虽有 `train_all=3240`，但全部记录 `actions=0`、`assistant_turns=0`、`image_paths=0`，不能用于训练。

## 正确做法

任何 rollout smoke 成功后，在扩大到全量采集前，必须立即做最小 conversion smoke，至少检查：

1. `train_all.jsonl` 能生成；
2. `image_paths > 0`；
3. `image_paths` 数量等于消息中的 `<image>` 数量；
4. `assistant_turns > 0`；
5. `actions > 0`；
6. `validation_issues == 0` 或明确解释每类 issue；
7. 随机打开几张图片确认分辨率和内容合理。

只有这些检查通过，才允许提交长时间/大规模 rollout collection。不要把“JSONL 行数正确”当成“训练数据可用”。

## 修复方向

- 修复 VAGEN dump：保存 `image_0/images_<idx>/*.png`，并在 JSONL 写入 `image_paths`；
- converter 支持当前 `output_str` schema；
- 每次改 rollout dump/schema 后重新跑 conversion smoke。
