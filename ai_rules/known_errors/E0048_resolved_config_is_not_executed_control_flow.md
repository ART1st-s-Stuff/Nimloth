# E0048：resolved config字段不等于实际执行路径

## 已确认错误

核对VAGEN源run时只看到`use_kl_in_reward=true`，一度据此判断该实验同时执行了reward
KL。继续检查trainer控制流后确认：该run同时启用`actor.use_kl_loss=true`，训练器因此
跳过reward KL，只在actor loss中执行`low_var_kl × 0.001`。

## 原因

resolved config只能证明字段取值，不能证明该字段所在分支被执行。互斥开关、上层条件、
短路逻辑或配置优先级都可能使一个true字段不生效。

## 正确做法

- 对外部训练run复现算法语义时，同时检查resolved config和目标版本源码控制流。
- 分别记录“配置字段存在”“执行条件满足”“实际指标/日志出现”三层证据。
- KL尤其要区分reward shaping与actor loss；两者会改变不同的return、advantage和value
  target，不能因名称相近而合并描述。
