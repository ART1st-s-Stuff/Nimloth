# E0119：区分rank-local empty K4 target与global empty batch

## 错误

ID181的global K4 batch存在valid WM windows，但production DP排序使一个rank的当前
micro-batch没有local valid row/window。`vagen/joint_policy/actor.py`中的
`_k4_dino_target_tensor`只检查local mask，因position为空而误报
`K4 DINO target batch contains no valid image`，导致Job `519935`在actor update失败。

## 后果

24-lane rollout已返回471条turn records，但没有complete global update、source777、
checkpoint或restore-only证据。ID181不可恢复或复用。

## 正确做法

- global valid/window合同必须基于all-reduce结果；不能把某rank的empty local shard当成
  global empty batch。
- local mask为空时返回shape/dtype/device正确的zero placeholder并继续相同DDP计算图；
  该rank必须由既有valid mask保证对DINO loss贡献为零。
- local mask含valid position而对应image为`None`时仍须fail closed；禁止用zero target
  伪造DINO teacher监督。
- DP测试必须覆盖至少一个rank local-empty、另一个rank global-valid的micro-batch边界。
