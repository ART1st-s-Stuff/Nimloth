# E0045: Stage trigger 必须匹配 launcher 变量契约

## 已发生错误

ID23 held-allocation stage01把模型和配置导出为`MODEL_DIR`与`CONFIG_PATH`，但目标launcher要求`MODEL`与`CONFIG`。Stage在启动env、trainer或W&B前立即失败；虽然没有实验artifact，却浪费了一次人工trigger并增加了identity审计负担。

## 正确做法

- 发布held stage前，直接读取目标launcher顶部全部`${VAR:?}`契约，并逐名核对trigger导出变量。
- 对trigger先执行静态契约测试，禁止凭相近脚本中的变量名推断。
- 若错误发生在任何env/trainer/W&B初始化前，可以在记录零artifact证据后用同一实验identity重试；一旦产生真实实验artifact则必须终止identity。
