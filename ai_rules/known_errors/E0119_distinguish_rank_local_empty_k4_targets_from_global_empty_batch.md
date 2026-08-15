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
  global empty batch。K4 module仍只拒绝真正global-empty的window/critic batch。
- local mask为空时返回shape/dtype/device正确的zero placeholder；该rank由既有valid mask保证
  DINO loss为零，并需用zero autograd anchor触达所有predictor参数以维持默认DDP graph。
- 若当前micro-batch在所有rank都只有padding，所有rank必须在任何model/teacher调用前根据
  all-reduced count一致skip；不能让mini-batch整体的valid count掩盖empty尾micro。
- local mask含valid position而对应image为`None`时仍须fail closed；禁止用zero target
  伪造DINO teacher监督。
- DP测试必须同时覆盖rank local-empty/peer valid与all-ranks-empty trailing micro-batch。
