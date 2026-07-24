# E0031：训练重构不能用 service object 代替完整模型图

## 已确认错误

SFT2/RL 可读性重构曾新增 `SFT2Algorithm`、`RLAlgorithm`、多个 loss result 和
`Components` 容器，但仍把 LLM、StateProjector、WMPredictor、ValueHead 分散传递。
代码文件变多后，开发者仍无法从一个 `nn.Module` 看见完整模型，loss 也继续接收
多个子模块参数。

## 错误原因

重构按训练流程创建了 service object，却没有先建立深度学习项目的模型边界：
模型参数应由 `nn.Module` 拥有，公共 loss 应通过该模块调用，processor、EMA、
optimizer 等运行期状态才留在 trainer/runtime。

## 正确做法

- 保留项目已稳定使用的 `StateProjector`、`LatentWMPredictor`、`ValueHead` 命名。
- `WorldModel` 组合上述三个模块，神经网络 `Agent` 再组合 `Backbone` 与
  `WorldModel`；episode 状态机使用独立 `AgentRuntime`。
- loss 应成为模型或阶段算法的成员行为，并接收结构化 batch/output；禁止重新拆成
  多个模型参数传递。
- SFT2/RL 只保留各自确实不同的梯度策略和训练生命周期。
