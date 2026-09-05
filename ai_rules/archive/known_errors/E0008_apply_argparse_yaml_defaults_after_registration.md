# E0008 — argparse YAML defaults 必须在参数注册后应用

## 错误

SFT2 parser 在 `add_argument` 之前调用 `parser.set_defaults` 应用 YAML。后续每个带 `default=` 的 `add_argument` 又覆盖 YAML，导致 `latent_wm_value_k8.yaml` 实际解析成默认 k=1。Cache builder 正常退出，但 manifest 已是错误语义。

## 正确做法

先注册全部 argparse arguments，再调用 YAML `set_defaults`；CLI 解析仍自然覆盖 YAML。必须有回归测试同时检查：

- k=8 YAML 在没有显式 CLI override 时得到 k=8；
- 显式 CLI 值优先于 YAML。

任何 cache/train 启动后都要检查实际日志或 manifest 的关键参数，不能只相信传入了 `--config`。
