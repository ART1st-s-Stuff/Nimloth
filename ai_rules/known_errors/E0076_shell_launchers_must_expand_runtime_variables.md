# E0076：shell launcher 必须实际展开运行时变量

## 已确认现象

新建的 SFT2 WS16 batch 和 node 脚本曾把 shell 变量写成带反斜杠的字面量。
这种写法能通过 bash 语法检查，但不会展开 `REPO`、`RUN_OUTPUT` 等运行时变量；任务获得
allocation 后会在模型加载前因路径检查失败。对应 job 500294 在分配前取消，没有占用 GPU，
也没有产生 W&B、optimizer 或 checkpoint。

## 正确做法

- shell 脚本中的运行时变量必须使用正常参数展开，禁止保留反斜杠转义。
- 静态回归必须扫描正式 batch 和 node 入口，拒绝带反斜杠的 `${...}`。
- 除 bash syntax 外，还要核对节点数、local/global rank、gradient accumulation 和完成
  validator 参数来自同一拓扑合同。
- 修复后必须使用新 commit、新实验 ID、空输出和新 W&B identity 重新执行 preflight；
  不得把从未运行的旧 job 当作 GPU 证据。
