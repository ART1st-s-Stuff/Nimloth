# 设计

## 训练目标与身份

Stage 1 继续对完整目标回答执行 teacher forcing。逐 token CE 先按标准 next-token shift 和 answer mask 取有效位置，再只对八个动作编号 token 乘以 8；动作起始、动作结束、EOS 和其他回答 token 为 1，按有效权重和归一化。实现使用能表达“动作编号集合”的名称，避免继续把边界误称为动作 token。

新的 loss 范围写入 objective/checkpoint 身份。旧的“边界及编号共同加权” checkpoint 不能恢复其优化器、调度器或收敛历史；本任务从原始 `hf_actor` 重新初始化。预处理 tensor 中的 input/label 不因 loss 权重而变化，但成功子集改变了数据集合：优先通过已有逐记录 hash/ID 验证复用缓存 tensor，并发布新的成功子集 manifest；现有缓存接口不能证明逐记录等价时，按成功 JSONL 重建缓存，不复制或改写旧 manifest 冒充新集合。

## 严格格式指标

训练期 Stage 1 自由生成同时检查 token 终止和回答 envelope：

1. 从 prompt 后的实际生成 token IDs 判断模型是否生成 EOS；达到长度上限、只有 pad、缺少 EOS 或 EOS 后仍有非 padding 内容都失败。
2. 仅在终止合法后移除生成末尾的终止/padding 表示，将正文交给 Stage 1 的共享严格 parser。
3. 正文必须包含完整非空 CoT、恰好一个合法动作块且无尾随内容。

正式 evaluator 保持无约束生成，不把 `action_end` 配成 stop token，也不补 EOS。训练期与正式评估共享正文解析合同，并用交叉测试保证相同正文结论一致；训练期另外保留终止原因，使“正文正确但未结束”不能被计为格式通过。Stage 2 的 query inject 专用评估保持自己的前缀合同，不继承 Stage 1 的权重或误用完整 CoT 生成检查。

## 数据选择

以已审核的 B prompt 派生数据为唯一文本输入来源，原始 rollout 和现有派生文件只读。转换器已经发布 `sft1_train_success.jsonl`；为本实验建立明确的成功 train 与成功 heldout 视图，按 trajectory-level `success == true` 筛选，不按单步奖励猜测成功。

启动前输出并核验：trajectory 数、assistant turns、八类动作计数、源 record ID/hash、train/heldout 重叠为零、所有图片和目标 token 完整。任一动作缺失或成功子集规模不足以形成有效训练/完整验证时停止，不混入失败 rollout。训练/收敛 LM 只使用成功子集；完整 heldout prompt 只用于严格格式生成检查，不把其失败参考回答加入优化目标。

## 初始化、训练与保存

从 `/mnt/nimloth/checkpoint/hf_actor` 创建新的不可覆盖 BF16 初始化产物：动作起止分别复制 input embedding/output head 的 EOS 行，八个动作编号分别使用既定 forward/backward/right/left/rotate right/rotate left/up/down 语义词行均值。全 tensor、token ID、dtype 和源模型 hash 验证后发布。

训练保持 a100-1 单机 8 GPU FSDP、LoRA r64/alpha128/dropout0.05、batch 1/rank、GA8、LM LR 1e-6、embedding/head LR 5e-6、max length 20000、既有图像尺寸、BF16/FA2、seed 42。训练前运行同拓扑的有限保存恢复门禁；正式运行使用唯一目录和 fresh optimizer/convergence state。

每 10 optimizer steps 发布完整 resume checkpoint。每个 epoch 完成全量成功-heldout LM 验证、严格格式样本和 epoch checkpoint 核验后，清理由该 epoch 覆盖的本次 step checkpoint；失败时不清理。至少完成 2 epoch，以相邻完整验证 LM loss 改善为准，连续 2 轮改善不足 1% 才发布收敛 final。每个运行段到时只在完整边界暂停和续训。

## 评估链

收敛 final 合并导出后，先使用统一 Stage 1 生成器对固定 32 条完整 heldout prompt 做严格原文门禁，保存 token IDs、EOS/length 状态和 parser 结果。至少 31/32 必须生成 EOS 且正文完整匹配；未达标则保留失败证据并停止，不启动已知低效的 120-episode 环境运行。

门禁通过后，调用唯一 Stage 1 success-rate 入口，在 test Base 60 与 Common Sense 60、seed 1..60、最多 20 步、既定无约束生成参数上完成真实环境评估。报告 checkpoint、commit、完成数、严格格式率、Base/Common Sense/overall success、回报与终止原因。任何部分运行只报告实际分母。

## 兼容与回滚

代码修改集中在 Stage 1 loss、训练期格式评估、配置/身份、测试及模块说明；不修改 Stage 2/3/RL 目标。旧 checkpoint、旧缓存、旧评估和暂停的 Stage 2 原样保留。代码回滚通过专用分支提交完成；实验失败不覆盖旧的有效结论，也不自动恢复旧 loss 继续训练。
