# E0043: `</think>` 检测不能依赖固定 token-ID 子序列

## 已发生错误

ID20 attempt5的实际decoded输出以`<think>Move forward.</think>`开头，但生成器仍跑满2048 tokens并报“未输出`</think>`”。

## 原因

代码预先单独编码`</think>`并匹配固定token-ID子序列。实际上下文中前一个句号与标签开头发生BPE合并，token序列为`.</` + `think` + `>`；decoded文本已有完整闭合标签，但固定ID序列无法命中。

## 正确做法

- 在每个采样token后增量decode当前prefix，并检测decoded文本中第一次完整`</think>`。
- 命中时保留产生该decoded边界的最短采样token prefix及对应behavior log-probs；不能重新tokenize文本来替代真实采样IDs。
- 仍需fail-closed：到上限后decoded文本确实没有完整闭合标签才报错。
- 测试必须覆盖标签与前置标点/文本发生BPE合并的tokenizer行为，而不只覆盖独立编码标签。
