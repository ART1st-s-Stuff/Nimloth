根据你的项目计划、之前讨论的弱点分析以及最新的研究成果（如 FOREWARN、MARVL、HiLAM 等），我为你整理了一份非常具体的细化执行计划。这份计划旨在将你的“System 1/System 2 解耦”构想转化为可工程化的实现。
细化执行计划：具有多模态思维引导的分层隐层世界模型 (HLWM-MTG)
1. 核心架构与数学定义
系统逻辑基于“预见（Foresight）与审思（Forethought）”的解耦 1, 2。
隐层特征 ($z_t$): $z_t = \text{MLP}(\text{DINOv2}(I_t))$，捕获底层物理和空间特征 3。
语义特征 ($s_t$): 取 VLM 在生成思维链 (CoT) 后的最后一个隐藏层向量。它不直接代表状态，而是代表**“预期的状态变化”** 4, 5。
动作策略 ($PM$): $a_t = \pi_\theta(z_t, s_t)$，采用 Transformer 架构，将 $s_t$ 作为 Task Embedding 注入 6, 7。
2. 阶段一：自动化数据采集与标注 (AI2THOR)
实现步骤：
随机探索数据： 利用模拟器随机采样底层动作 $a_t$，收集三元组序列 $\{(I_t, a_t, I_{t+1})\}$。必须包含碰撞和失败轨迹，用于后续 WM 的不确定度校准 8, 9。
全知标注 (Oracle Labeling)： 利用 AI2THOR 的 Metadata 获取物体的 3D 坐标和状态 10, 11。
CoT 标注： 将物理状态变化转化为文本，如：“检测到夹具距离杯柄 5cm，正在靠近”。
语义段划分： 利用 Ramer–Douglas–Peucker (RDP) 算法在动作空间检测拐点，自动划分任务阶段（如：靠近、抓取、移动、放置） 11。
多视角配对： 同时渲染 2-3 个视角的图像，用于训练视角无关的 $z_t$ 编码 12, 13。
3. 阶段二：世界模型 (WM) 训练与校准
实现步骤：
速度场拟合： 基于 Conditional Flow Matching (CFM) 训练 $v_\phi$。
损失函数： $\mathcal{L}{WM} = \mathbb{E}{t, z_0, z_1, a} \| v_\phi(z_t, a_t) - (z_1 - z_0) \|^2$ 3。
不确定度量化 (关键步骤 Q1)：
在推理时，对输入 $z_t$ 注入 $\epsilon \sim \mathcal{N}(0, \sigma^2)$（$\sigma$ 取 $z$ 标准差的 5%）。
计算输出扰动的散度：$\text{Div} \approx \frac{1}{K} \sum_{k=1}^K \|v_\phi(z_t + \epsilon_k) - v_\phi(z_t)\|$ 3。
阈值设定： 在训练集上运行该逻辑，取散度分布的 95% 分位数 作为快速执行的退出阈值 $\theta_{div}$。
4. 阶段三：VLM 接地与 $s_t$ 对齐训练 (核心 Q5)
实现步骤：
物理特征注入 (Adapter)： 建立一个线性投影层 $W_{proj}$，将 $z_t$ 映射为 Qwen-2.5-VL 的词嵌入维度 14, 15。
行为叙述微调 (LoRA)： 冻结 VLM，仅微调投影层和 LoRA 权重，训练 VLM 描述基于 $z_t$ 序列的物理细节 16, 17。
跨模态对齐 (InfoNCE)：
训练一个轻量级编码器 $h_\psi(z_t, z_{t+k})$，将物理空间的**“状态变化”**映射到语义空间 4。
损失函数：$$\mathcal{L}{align} = -\log \frac{\exp(\cos(s_t, h\psi(z_t, z_{t+k})) / \tau)}{\sum \exp(\dots)}$$
这解决了 Q5：$s_t$ 被约束为与未来几帧的物理演化趋势对齐，而非直接等于物理坐标 18, 19。
时序一致性约束： 在同一语义段内，强制 $s_t$ 的变化极小化，通过惩罚 $\text{MSE}(s_t, s_{t+1})$ 来实现 20, 21。
5. 阶段四：策略模型分层蒸馏与集成
实现步骤：
PM 训练： 输入 $z_t, s_t$，通过行为克隆 (BC) 学习动作序列 $a_t$ 6, 22。
Dyna-style 梦境训练 (解决 Q6)：
在模拟轨迹时，WM 生成预测状态 $\hat{z}_{t+k}$。
结果验证 (Outcome Verification)： 让 VLM 扮演审判官，评估 $\hat{z}_{t+k}$ 对应的语义描述是否符合逻辑。如果 VLM 发现 WM “走偏”了（例如生成的语义描述为“杯子破碎”），则停止该轨迹的训练并降低其权重 23, 24。
闭环逻辑：
if WM_div > theta_div or PM_entropy > theta_ent:
    # 触发 System 2 (慢速思考)
    s_t, CoT = VLM.deep_reasoning(image, z_t) 
else:
    # 维持 System 1 (快速反应)
    a_t = PM(z_t, s_t)
6. 给编程 AI 的具体实现提示 (Programmer's Cheat Sheet)
模型选型： Qwen-2.5-VL-8B 的修改重点在于 forward 方法。将物理 Token 拼接在 vision_tokens 之后 14。
负采样策略： 在 InfoNCE 损失中，负样本应选择来自同一视频但不同时间段的轨迹片段，这能迫使 $s_t$ 区分细微的动作意图差异 18, 25。
不确定度触发优化： 为了防止系统在切换点频繁抖动，引入“耐心值”（Patience），只有连续 3 帧散度超标才调用 VLM 26。
场景-视图解耦： 如果条件允许，在 Encoder 输出 $z_t$ 后加一个 Shuffle Loss，确保 scene code 与 view code 正交，解决泛化性问题 (Q7) 27, 28。
这份文档现在具有很强的实操性。如果你同意，你可以直接将其交给编程 AI 启动 数据采集 (Phase 1) 和 世界模型实现 (Phase 2)。

