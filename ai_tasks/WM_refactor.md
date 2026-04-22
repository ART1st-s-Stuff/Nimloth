我希望WM支持连续的动作。分以下步骤执行：

## 更改AI2THOR的数据收集
**数据集标签修改**
- 现在转视角默认是90度，前进也是默认的固定速度，现在将其改为在一定范围内的数值
- 每帧的动作标签为(move_ahead_distance, delta_yaw, delta_pitch)

**采样策略修改**
使用类似Ornstein-Uhlenbeck (OU) 过程来使采样Agent的运动具有一定趋势，而不是随机噪声。

减少卡墙等无效数据，但又需要让WM学习到卡墙的后果。通过读取 event.depth_frame，检测中心区域的平均深度。如果深度值小于某个阈值（如 0.5米），判定为即将撞墙。Agent有一定概率（可配置概率数值）撞墙或躲避；当深度变小时，减小 MoveAhead 的步长，并增大 Rotate 的概率。一旦真正发生撞墙，那么在至多N次失败动作后强制Rotate。

除了收集随机游走数据外，也收集一些使用场景导航网格 (NavMesh) 的rollout数据。AI2-THOR 内部维护了一个可达点集（Reachable Positions），在初始化场景时，调用 controller.step(action="GetReachablePositions")。从返回的 reachable_positions 中随机选择一个作为临时目标点，然后使用路径规划插件（如 SimpleNavMesh）生成路径。限制 Agent 的动作，使其每一帧的坐标始终落在 reachable_positions 的邻域内。

## 更改WM训练流程
目前，WM使用的是Ground truth action训练。增加两种训练范式：
- a) 无监督学习。使用一个逆动力学模型，从过去帧提取当前的Action，然后再让WM根据Predicted action预测下一帧，使用重构Loss监督。不使用数据里标注的Action监督信号。
- b) 半监督学习。同样使用逆动力学模型，但将Predicted action线性映射到Ground truth action space，使用标注的监督信号监督。但WM不直接使用Ground truth action监督。

## 增加可配置项
- WM以及WM的逆动力学模型中，可以看到过去的K帧，K是可配置项
- 你可能需要把WM改为Transformer实现，并配置Transformer的层数、参数量等参数。