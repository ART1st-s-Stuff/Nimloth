--------
本文为人类编写。如需修改需要得到人类同意。
--------

# 第二阶段SFT实验
在第一阶段SFT的基础上，让<|latent_state|>对应的隐状态保存足够的信息，能够被一个WM predictor正确预测。

WM predictor使用LeWM: https://github.com/lucas-maes/le-wm

这一阶段仍然使用之前收集到的rollout数据，选取train set进行训练，但是主loss改为predictor的loss。

需要记录训练过程中predictor的MSE曲线，以及在val set上的成功率。