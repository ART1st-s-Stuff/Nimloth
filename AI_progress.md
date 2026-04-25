# AI项目进展（按Phase）

更新时间：2026-04-25
统计口径：仅记录可由当前仓库文档与任务进展文件验证的内容；未落地项明确标注为”未开始/规划中”。

---

## Phase 1：数据采集与标注（AI2THOR）

### 已完成
- 动作标签升级为连续动作三元组：`move_ahead_distance`、`delta_yaw`、`delta_pitch`。
- 采集配置接入连续动作范围、防撞参数、NavMesh rollout 参数，并完成 Hydra 配置映射。
- AI2THOR 适配器新增连续动作执行、中心深度估计、可达点读取接口。
- 采集器实现 OU 趋势采样、防撞策略、失败动作强制旋转、NavMesh 混合 rollout。
- 随机游走策略从 OU 过程切换为“目标采样 + 匀速小步执行”，并补充目标扰动/速度扰动/单轴动作概率。
- Recover 策略改为连续小角度旋转，支持历史轨迹方向与最小累计转角约束；同时新增恢复与 pitch 控制相关观测字段。
- 采集配置从 `ou_process` 迁移为 `random_walk`。

### 使用模型
- 当前阶段以环境与采样策略为主，无新增可训练大模型训练结论。
- 数据采集后端采用 `ai2thor`（同时保留 `mock` 后端可插拔能力）。

### 使用算法
- Ornstein-Uhlenbeck (OU) 趋势采样（历史方案，已在随机游走主流程中替换）。
- 目标驱动随机游走（目标采样 + 匀速小步执行）。
- 基于中心区域深度估计的防撞/避撞策略。
- 失败动作后的强制旋转恢复策略。
- 基于可达点集合的 NavMesh 混合 rollout 采样。
- 连续小角度恢复与最小累计转角约束。

### 当前状态
- **已完成（核心链路）**：连续动作采集与采样策略重构已落地。
- **进行中（稳定性）**：角落起始位姿下的恢复效率仍需进一步优化。

### 下一步
- 增加角落判定（多区域深度）与双向试探旋转策略。
- 在 Recover 退出条件中加入“深度改善幅度”判据，减少假恢复。
- 做多场景短程回归，统计 recover 触发率、平均卡墙停留步数与无效轨迹占比。

---

## Phase 2：世界模型（WM）训练与校准

### 已完成
- 建立并跑通 CFM 世界模型训练最小闭环。
- 训练数据集升级为 `K` 帧历史序列样本：`z_history`、`action_history`、`z_next`、`gt_action`。
- WM 主干升级为 Transformer 时序建模。
- 新增逆动力学模型模块。
- 训练流程支持 `unsupervised` 与 `semi_supervised` 两种模式，并支持损失权重与梯度裁剪配置。
- 训练流程新增 `fully_supervised` 模式：WM 训练直接使用真实动作序列，不再依赖 IDM 推断动作。
- 阈值校准流程已支持并使用散度 95% 分位数生成 `theta_div`。
- LeWM（Latent Energy World Model）实现，支持因果注意力和 SIGReg 正则。
- SIGReg 正则超参数可配置化：支持 `num_quadrature_points`、`t_min`、`t_max`、`kernel_sigma`。
- LeWM 支持混合编码器配置：`lewm_qwen25vl_8b`、`lewm_dinov2m_qwen25vl_8b`。

### 使用模型
- 编码侧：DINOv2（冻结编码器，按项目文档进行映射使用）。
- 世界模型：Conditional Flow Matching (CFM) 世界模型、Latent Energy World Model (LeWM)。
- 时序骨干：Transformer（双向 for CFM，因果 for LeWM）。
- 动作推断：逆动力学模型（用于无监督/半监督训练范式）。

### 使用算法
- Conditional Flow Matching（速度场拟合）。
- Latent Energy World Model（绝对 latent 预测 + SIGReg 正则）。
- Transformer 时序建模（多帧历史状态与动作条件建模）。
- 逆动力学重构（从状态变化推断动作）。
- 无监督训练范式：使用预测动作驱动 WM 重构下一状态。
- 半监督训练范式：将预测动作映射到标注动作空间进行监督约束。
- 全监督训练范式：直接使用标注动作驱动 WM rollout，跳过 IDM 动作推断分支。
- 基于分位数统计的散度阈值校准（95% quantile）。
- SIGReg（Sketch Isotropic Gaussian Regularizer）：随机投影 + Epps-Pulley 统计量正则化。

### 当前状态
- **已完成（最小闭环）**：训练与校准主链路可运行并产出模型与阈值。
- **已完成（LeWM 扩展）**：SIGReg 超参数可配置，支持混合编码器配置。
- **进行中（对比实验）**：训练和比较 4 种配置：cfm_dinov2m、lewm_dinov2m、cfm_dinov2m_qwen25vl_8b、lewm_dinov2m_qwen25vl_8b。

### 4种配置对比实验状态（2026-04-25）

| 配置 | 训练状态 | 校准状态 | 评估状态 | 目录 |
|------|---------|---------|---------|------|
| cfm_dinov2m | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 | 2026-04-23_14-48-14 |
| lewm_dinov2m | ✅ 已完成 | 🔄 校准中 | 需评估 | 2026-04-24_21-57-26 |
| cfm_dinov2m_qwen25vl_8b | ❌ 训练失败（Qwen encoder未实现） | - | - | - |
| lewm_dinov2m_qwen25vl_8b | ❌ 训练失败（Qwen encoder未实现） | - | - | - |

**注意**: Qwen-based encoders (cfm_dinov2m_qwen25vl_8b, lewm_dinov2m_qwen25vl_8b) 尚未实现，使用 PlaceholderEncoder 占位。需先实现 Qwen2.5-VL 编码器才能训练这些配置。

### 已知评估指标（cfm_dinov2m，来自2026-04-23）
- `wm_mse`: 0.346026
- `latent_fd_mean`: 41.345353
- `latent_cd_mean`: 0.044785
- `divergence_auroc`: 0.780230
- `theta_div`: 3.909910

### 本次训练参数与结果（2026-04-23）
- 训练产物目录：`models/wm/cfm_dinov2m/2026-04-23_00-32-53`
- 校准产物目录：`models/wm/cfm_dinov2m/2026-04-23_01-03-30`
- 关键训练参数：`training_mode=semi_supervised`、`epochs=4`、`batch_size=16`、`rollout_steps=4`、`temporal_stride=1`、`detach_idm_in_wm=true`
- SIGReg相关参数：`sigreg.enabled=false`、`sigreg.weight=0.1`、`sigreg.warmup_steps=1000`（本次未启用SIGReg训练）
- 训练结果（`train_metrics.json`）：
  - `last_loss=14.851808888121294`
  - `last_loss_recon=2.919360613880249`
  - `last_loss_action=11.932448274241043`
  - `last_latent_var_min=0.39067535393704206`
  - `last_latent_mean_norm=21.70720508159735`
  - `last_latent_cov_trace=1100.2921602298052`
- 校准结果（`theta_div.json`）：
  - `theta_div=1.043241354636848e-05`
  - `percentile=95.0`
  - `num_values=39960`

### 下一步
- 训练并比较 4 种配置：cfm_dinov2m、lewm_dinov2m、cfm_dinov2m_qwen25vl_8b、lewm_dinov2m_qwen25vl_8b。
- 使用 `evaluate_wm.py` 评估所有配置的 MSE、FD（ Frobenius Distance）、CD（Cosine Distance）指标。
- 根据对比结果选择最优配置，并调参优化。

---

## Phase 3：VLM接地与语义状态对齐

### 已完成
- 已在方案层明确目标：以 Qwen-2.5-VL-8B 作为默认 VLM，语义状态 `s_t` 作为“状态变化意图”表示。
- 已形成候选训练路径：投影层注入、LoRA 微调、InfoNCE 跨模态对齐、时序一致性约束（文档级设计）。
- 新增 Phase 3 专用数据集 `SemanticAlignDataset`，统一输出 `z_t/z_t_pos/z_t_neg/task_text/env_context/segment_id/view_id` 契约并支持同视频异时段负采样。
- 新增 `src/vlm` 正式模块：`QwenVLMAdapter` 与 `SemanticStateGenerator`，支持真实 Qwen 推理与 fallback 占位模式统一接口。
- 新增语义对齐训练入口 `src/train/train_semantic_align.py`，实现 InfoNCE + 时序一致性联合优化并接入 Hydra/W&B 产物记录。
- 新增语义对齐评估 `src/eval/eval_semantic_align.py`，可输出同意图相似度、异意图分离度与时序平滑指标。
- 新增 `src/train/export_pm_ready_features.py`，导出 Phase 4 可直接消费的 `state={z_t,s_t,env_context}` 数据与接口契约文档。

### 使用模型
- Qwen-2.5-VL-8B（默认规划模型，当前未完成该阶段训练闭环）。

### 使用算法
- 线性投影层（`z_t` 到 VLM 词嵌入空间）【规划中】。
- LoRA 轻量微调【规划中】。
- InfoNCE 跨模态对齐损失【规划中】。
- 语义段内时序一致性约束（如相邻 `s_t` 平滑约束）【规划中】。

### 当前状态
- **已进入工程实现（高完整度）**：Phase 3 已具备数据构造、训练、评估、导出完整代码路径；待在目标机器执行端到端实验产物回归。

### 下一步
- 在 AI2THOR 数据上执行 `train_semantic_align` 与 `eval_semantic_align`，沉淀首组 checkpoint 与指标基线。
- 基于评估结果调节 `positive_k/negative_gap/temperature/temporal_weight`。
- 逐步接入更强监督信号（标注 CoT 或阶段标签）并扩展跨视角一致性测试。

---

## Phase 4：PM训练与全系统集成

### 已完成
- 已形成系统闭环设计：当 WM 散度或 PM 不确定度超阈值时切换到 VLM 深度推理，否则走快速执行路径（设计层）。
- 已新增 PM-ready 离线特征导出脚本与接口契约，可供下一轮 PM 基线直接消费。

### 使用模型
- PM（策略模型，规划为接收 `z_t` 与 `s_t` 的策略网络，具体结构待定）。
- 与 WM、VLM 进行分层协同（当前为设计定义，尚未完成训练集成）。

### 使用算法
- 行为克隆（BC）训练 PM【规划中】。
- 基于 WM 的 Dyna-style 想象训练【规划中】。
- 不确定度触发切换逻辑（WM 散度/PM 熵阈值）【规划中】。

### 当前状态
- **未开始（训练与集成）**：PM 训练本体尚未落地，但数据接口预埋已完成。

### 下一步
- 明确 PM 架构与动作空间对齐方案（与 WM/VLM 输出接口一致）。
- 先完成 PM 的监督学习基线，再接入不确定度切换逻辑。
- 在固定场景完成闭环联调后，再扩展到跨场景泛化测试。

---

## 跨Phase风险与阻塞

- 角落位姿下 Recover 效率仍可能下降，需要角落特化策略。
- AI2THOR 在线采集与全量训练回归尚未执行，当前主要是轻量链路验证。
- Phase 3/4 关键问题（`s_t`监督、PM 结构、跨场景泛化）尚需进入实证阶段。

## 近期优先级（短期）

1. 完成 AI2THOR 采集冒烟 + Phase 2 双范式短训回归。
2. 量化采集端恢复质量指标并迭代 Recover 策略。
3. 启动 Phase 3 最小可运行实验（投影注入 + 小规模对齐）。
4. 定义 PM 基线训练脚本与接口契约，准备 Phase 4 集成。
