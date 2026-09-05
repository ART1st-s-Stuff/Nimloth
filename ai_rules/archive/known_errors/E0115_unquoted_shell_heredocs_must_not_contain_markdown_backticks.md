# E0115：未引用shell heredoc内禁止直接写Markdown反引号

## 错误

ID178 runner用`<<EOF`生成需要展开环境变量的metadata，同时在正文写了Markdown
反引号形式的`--model`与`--critic-checkpoint`。

## 后果

shell把反引号内容当command substitution执行，最终README对应字段变成空字符串。
校准本身未受影响，但自动实验记录不完整，需要on-experiment-end重写README。

## 正确做法

- 需要变量展开的未引用heredoc中，不写未转义反引号；使用普通文本或转义反引号。
- 或使用quoted heredoc并通过显式参数生成动态内容。
- launcher测试必须覆盖关键metadata文本，不能只验证业务命令。
