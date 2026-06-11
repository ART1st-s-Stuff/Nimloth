# 05 Flower Reference

## 参考定位

`../flower` 是 Nimloth 的历史参考项目。可以阅读它的文档、结构和代码来理解背景，但 Nimloth 是重制项目，不是机械复制。

## 可以参考

- AI 协作规则的组织方式；
- world model / agent / training / evaluation 的模块划分经验；
- 已验证有效的实验记录方式；
- 旧项目中明确成功、仍符合 Nimloth 目标的设计思想。

## 不得盲目迁移

不要未经验证迁移：

- 已知错误实现；
- 临时 hack；
- 过时路径；
- 为旧实验定制的脚本；
- 与 Nimloth 新目标不一致的架构；
- 语义不清的 loss、head、checkpoint、data split、rollout 逻辑。

## 迁移前检查

如果未来要从 flower 迁移设计或代码，必须先说明：

1. 迁移对象；
2. 为什么适合 Nimloth；
3. 依赖和风险；
4. 与旧项目已知问题的关系；
5. 如何验证迁移正确。

若无法回答，停止询问人类。
