# E0109: Calibration failures must persist measured diagnostics

## 已发生的错误

ID173完成24条真实TP8 K4 trajectory后，beta校验发现LLM action-logit median spread为0。
入口先检查beta必须为正并抛异常，随后才会写`summary.json`和`turn_records.jsonl`，导致已测得的
LLM/MCTS spread、latency和逐turn证据没有持久化。`failure.json`只保存了笼统异常文本。

同时，批准合同只规定MCTS spread接近0时失败；实现自行增加“beta必须为正”的门槛，未单独记录
beta0及其原因。

## 正确做法

- 任何有限scale测量都应先原子写入diagnostic summary；逐turn记录也应在最终接受/拒绝beta前落盘。
- rejection必须保存LLM spread、MCTS spread、公式结果、阈值及明确reason code。
- 不得自行增加改变实验结论的数值门槛；遇到zero prior spread时停止请人类选择，而不是丢弃证据。
- failed calibration仍不得生成optimizer、training checkpoint或启动后续canary。

## Evidence

- ID173 Job`519648`完成24条trajectory后报
  `RuntimeError: calibrated K4 beta is not finite and positive`。
- `vagen/k4_beta_calibration.py::validate_and_summarize`在写summary之前执行positive-beta检查。
- 服务器`.../173_calibration_k4mcts_.../failure.json`只有异常，缺少scale诊断；README记录了由校验顺序可确定的zero prior spread。
