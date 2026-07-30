# E0069：vLLM 与 Ray 的运行时目录必须满足 AF_UNIX 路径长度限制

## 已确认现象

SFT1/VAGEN parent配对评估job `498026`已通过checkpoint、render和环境prewarm，但两边在
模型加载阶段同时失败：vLLM ZMQ IPC socket和Ray plasma-store socket都位于很长的实验
output子目录下，最终路径超过Linux `sockaddr_un.sun_path`的107-byte上限。

## 禁止做法

- 禁止把`TMPDIR`、`RAY_TMPDIR`或vLLM/Ray runtime root直接放在深层实验output路径下。
- 禁止只检查父目录长度；vLLM会追加UUID，Ray会追加session和`sockets/plasma_store`。
- 禁止把这种启动失败解释为checkpoint、显存或模型质量问题。

## 正确做法

- 使用短的节点本地runtime root，例如`/tmp/npe-<job>-<arm>`；正式artifact仍写入实验output。
- 提交前用代表性的最长Ray/vLLM后缀验证完整socket path小于107 bytes。
- cleanup只能删除经过精确prefix/identity guard的本job runtime目录。
