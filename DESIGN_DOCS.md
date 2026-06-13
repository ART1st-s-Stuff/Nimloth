--------
本文为人类编写。如需修改需要得到人类同意。
--------

# Nimloth架构
Nimloth是一个世界模型。使用两个显式的模型来分别建模state transition和state value。

我们初步实验会使用与VAGEN相同的Navigation (EB-Nav)环境。但是，我们的设计中每一个step只会产生一个**action prior**，这一点和VAGEN不同。

## Prompt设计
参考VAGEN，只需做少量改动。

1. 配合<|latent_state|> token，获取当前state。格式要求该token位于CoT </think>标签的后面，然后以该token处的attention embedding作为当前state。
2. 使用<|action_(idx)|> token。action不再使用<action>some action</action>的文本形式输出，而是以<|action_start|><|action_(idx)|><|action_end|>的形式输出，这样可以在<|action_start|>后一个token的位置获取当前的action先验。

## State transition model
直接使用LeWM：https://github.com/lucas-maes/le-wm

## Value head
预测state对应的value，直接使用MLP网络。我们使用value head来进行多步MCTS，以便选择几个最佳action。

对于第一步，我们直接从Qwen的输出获取action prior。