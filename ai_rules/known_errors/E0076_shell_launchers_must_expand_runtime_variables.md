# E0076：shell launcher 必须实际展开运行时变量

## 已确认现象

新建的SFT2 WS16 batch和node脚本曾把shell变量写成字面量反斜杠加美元符号。
该形式能通过bash语法检查，却不会展开REPO、RUN_OUTPUT等运行时变量；若获得allocation，
脚本会在任何模型加载前因必需路径检查失败。对应job 500294始终PENDING并在拓扑改变时
取消，因此没有执行这个错误、没有占用GPU，也没有产生W&B或训练产物。

## 正确做法

- shell脚本中的运行时变量必须使用正常参数展开，禁止在提交文件里保留反斜杠转义。
- 静态回归必须扫描正式batch和node入口，拒绝反斜杠加美元符号再加左花括号。
- 除bash syntax外，还要测试节点数、local/global rank、gradient accumulation和完成
  validator参数由同一拓扑合同驱动。
- 发现此类错误后，必须在新commit和新实验identity上重新执行CPU/preflight门禁；
  不得把从未运行的旧job视为GPU smoke证据。
