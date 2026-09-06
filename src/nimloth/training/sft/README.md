# SFT训练与评估

算法说明见[spec.md](spec.md)。这里将格式训练、Query对齐和WM/value训练放在同一个目录，保留独立且可顺序阅读的阶段流程。

- [`stage1/`](stage1/README.md)：格式监督，复用原格式训练生命周期。
- [`stage2/`](stage2/README.md)：回答CE加query state到冻结DINO空间特征的对齐，共享projector供下一阶段使用。
- [`stage3/`](stage3/README.md)：原`training/sft2/`实现迁移，保留其数据、梯度、DDP、checkpoint和恢复语义。
- [`evaluation/`](evaluation/README.md)：`eval_direct`和`eval_wm`的真实环境rollout入口，复用环境、Agent及现有planner。

## 阅读顺序

先读各阶段算法/模型计算，再读trainer与loop的构建和生命周期。状态预测、损失与反传顺序应在拥有它们的模块中可见；不把配置解析、数据读取和优化器更新塞进算法函数。

SFT1仅监督目标回答token。SFT2同样保留CE，并比较projector输出和同观测的DINO grid。SFT3从真实起点状态沿记录动作递推：下一状态用于WM监督，动作前状态用于Q(s,a)监督，MC return在完整episode上计算后切窗。

## 阶段兼容

旧`sft2`名称始终代表WM/value阶段，即新`stage3`。旧配置、checkpoint字段和公共类型名继续保留原意义；不能用旧SFT2 checkpoint表示新query对齐阶段。旧 `training.sft1` / `training.sft2` Python 包已移除，活动调用统一使用本目录的阶段入口。

旧单步历史窗口及多步H=1行为沿用；本次不是重新设计SFT3算法。SIGReg仍在主loss反传后计算，当前状态detach、下一在线状态接收正则梯度。

## 评估边界

离线validation的loss与真实环境rollout指标分开。direct直接执行模型生成的动作；wm以真实观测和对应真实CoT构建状态，调用既有planner，只执行首个动作并重新感知。两条路径沿用相同episode、步数上限与指标口径。WM模拟步不计作真实环境步数。

本地CPU测试只能验证算法、梯度和接口；真实GPU训练、vLLM、DDP和环境成功率需要另行执行实验门禁。
