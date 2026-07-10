# E0005 — 转换前必须核对当前 rollout dump schema

## 错误

Rollout 成功后直接运行 legacy converter。Pinned VAGEN 把完整 transcript、指标和图像路径分别写在 `output_str`、`metrics`、`image_paths`，而 converter 只读取旧 `input`/`output` 和 top-level success/score，因而静默产出空 messages/actions 与 false success。

## 正确做法

每次升级或切换 VAGEN 后，先抽查 raw JSONL keys 和一条完整 transcript，再验证 converted record：

- assistant turns 与 actions 非空且数量一致；
- success/score 与 raw metrics 一致；
- image placeholder/path 数量一致且文件存在；
- prompt 不再含 XML action example；
- 所有 k 个 latent query tokens 和 action tokens 都存在。

不能只看 converter 进程 exit code。
