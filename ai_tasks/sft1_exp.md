--------
本文为人类编写。如需修改需要得到人类同意。
--------

# 第一阶段SFT实验
目前我们已经具有了50 global step的VAGEN baseline。

第一阶段SFT在VAGEN baseline的基础上对格式进行SFT，使用 ../DESIGN_DOCS.md 里描述的prompt方案。

## 实验步骤
1. 在VAGEN baseline上进行rollout (在train/val/test set上都做)，获取~6000条完整rollout数据，包括实际观察到的截图、每一步的完整chat history。
2. 把收集到的rollout转换成我们的prompt格式
3. 选取Training set上所有的成功rollout，进行SFT，每一轮在val set上进行eval。该步骤需要持续**直到loss收敛**。
4. 比较最终success rate和VAGEN baseline的差异