这是一份细化后的项目文档和执行计划。它吸取了之前讨论中关于世界模型累积误差（Stationarity Fallacy）、语义保真度（Semantic Fidelity）以及系统切换延迟的教训，并参考了 FOREWARN、GRIF 和 HiLAM 等前沿研究的数学方案。
项目文档：HLWM-MTG (分层隐层世界模型与多模态思维引导)
1. 核心架构设计
系统采用双回路分层控制，模拟人类大脑的快慢系统：
快速回路 (System 1): 包含基于流匹配的 WM 和 Transformer 架构的 PM，运行频率为 50-100Hz 1。
慢速回路 (System 2): VLM (Qwen-2.5-VL-8B) 通过思维链 (CoT) 生成语义状态向量 $s_t$，仅在 System 1 不确定度高时触发 2。
2. 模块化细节与数学描述
2.1 隐层世界模型 (WM) - 物理演化预测
原理： 使用条件流匹配 (CFM) 拟合速度场 $v_t$ 2。
损失函数： $\mathcal{L}{WM} = \mathbb{E} \| v\phi(z_t, a_{t:\tau}) - (z_{t+1} - z_t) \|^2$。
退出机制 (不确定度)：
散度计算： 对 $z_t$ 注入扰动 $\epsilon \sim \mathcal{N}(0, 0.05 \cdot \text{std}(z))$，计算 $\text{Div}(v_\phi) \approx \frac{1}{K} \sum \|v_\phi(z_t + \epsilon) - v_\phi(z_t)\|$ 2。
阈值控制： 设定 $\theta_{div}$ 为训练集分布的 95% 分位数。
2.2 语义状态 $s_t$ 与对齐逻辑
为了解决物理 $z$ 与语义 $s$ 的鸿沟，不再直接预测状态，而是预测**“状态的变化”** 3。
$s_t$ 定义： VLM 在 CoT 后输出的隐层 Embedding，代表当前阶段的“物理演化意图”。
对比任务对齐 (Contrastive Alignment)：
使用 InfoNCE Loss 训练一个 Adapter $h_\psi(z_t, z_{t+k})$ 4, 5：$$\mathcal{L}{align} = -\log \frac{\exp(\text{sim}(s_t, h\psi(z_t, z_{t+k}))/\tau)}{\sum \exp(\dots)}$$
动态分段： 参考 HiLAM 的动态块机制，当物理状态变化率 $p_t$ 超过阈值时（表示子任务完成），强制 VLM 更新 $s_t$ 6, 7。
2.3 策略模型 (PM) - 底层执行
架构： 接收 $z_t, s_t, \text{proprioception}$ 的 Transformer 模型 2, 8。
训练： 结合行为克隆 (BC) 与基于 WM 的离线想象训练 (Dyna-style) 8。
3. 针对弱点的补强策略 (Reviewer-Driven)
累积误差修正： 引入 VLM 作为结果验证器 (Outcome Verifier)。VLM 定期对比 WM 预测的 $\hat{z}_{t+k}$ 与真实观测，若语义描述不符（如“杯子已倒”），则强制重置 WM 状态 9, 10。
视图不相关特征： 在数据采集时增加随机多视角，并使用 Scene-View Decomposition 训练 Encoder，确保 $z_t$ 过滤掉背景和视角噪声 11, 12。
4. 执行计划 (Timeline)
阶段 1: 数据采集与标注 (AI2THOR) - 第 1-2 周
任务：
收集 10,000 条包含“碰撞”、“抓取失败”及“成功演示”的混合质量轨迹 13, 14。
利用 AI2THOR Metadata 自动生成 CoT 标注（例如：“检测到距离 0.1m，准备闭合夹具”） 15, 16。
采集配对的多视角图像以进行视图解耦训练 17。
阶段 2: 基础模型训练 (WM & Encoder) - 第 3-4 周
任务：
基于 DINOv2 训练物理特征投影层。
实现并训练 CFM 世界模型。
关键交付： 验证散度计算对“非法/碰撞”状态的识别准确率。
阶段 3: 语义对齐与 VLM 适配 - 第 5-6 周
任务：
训练 $W_{proj}$ 线性映射层，将 $z_t$ 注入 Qwen-2.5-VL 18, 19。
使用 InfoNCE 实现 $s_t$ 与物理变化序列 $(z_t, z_{t+k})$ 的跨模态对齐 4, 5。
关键交付： 验证相同语义指令（如“靠近”）在不同视角下的 $s_t$ 相似度。
阶段 4: PM 训练与全系统集成 - 第 7-8 周
任务：
训练底层 Transformer PM。
在缺少模拟器时，开启 Dyna-style 训练，由 VLM 监控 WM 的“想象偏差” 8, 20。
关键交付： 闭环运行成功率测试，特别是在新场景下的泛化能力测试 21。
5. 给编程 AI 的具体实现提示
WM 实现： 优先参考 torch-cfm 库，并在速度场输入中包含动作 $a_t$。
VLM 适配： 修改 Llama-3.2-Vision 或 LLaVA 的输入嵌入层，将映射后的物理 Token 置于图像 Token 序列之后 18。
切换逻辑：
if div_loss > theta_div or policy_entropy > theta_ent:
    s_t, cot = system_2.reasoning(image, z_t) # 触发 System 2
else:
    a_t = system_1.pm(z_t, s_t) # 维持 System 1 快跑
这个计划将你的原始设想转化为了一个具备自我纠错能力、多模态对齐的鲁棒架构。

