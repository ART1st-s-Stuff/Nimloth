# E0049：显式Ray head必须驱动worker排除逻辑

## 已确认错误

多节点launcher新增可配置`RAY_HEAD_NODE`后，head启动使用该节点，但worker循环仍固定跳过
排序节点列表的第一个元素。head不是列表第一项时，会漏掉一个物理节点并在head上重复启动
worker。Ray可能仍报告期望的node/GPU总数，使只检查计数的probe给出假通过。

## 正确做法

- worker遍历完整allocation节点列表，只按节点名跳过实际`HEAD_NODE`。
- 集群probe除alive node数量和GPU总数外，还必须检查`NodeManagerAddress`唯一数等于配置
  节点数。
- 只要物理节点映射不一致，即使环境health或模型初始化已开始，也必须停止；不得把这种
  infrastructure启动解释为有效rollout或PPO证据。
