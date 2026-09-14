# 修正SFT1符合spec并启动2000来源rollout训练

## 2026-09-11 Stage 2 追加授权
用户要求从 B epoch_005 (step135) 启动 Stage2，a100-1 使用7GPU，训练到收敛；确认验证集总loss=LM CE+DINO Query MSE，权重各1，至少2epochs，连续2轮相对改善不足1%停止。分别记录两项验证loss。复用B的1709/193划分和新prompt；K16/4x4按现有Stage2配置，保留真实CoT与查询插入合同。新建Stage2输出，不恢复SFT1优化器。此前“不启动stage2”被此次授权替代；spec.md不改。
实现范围：Stage2收敛及分项loss，query FSDP checkpoint支持和BF16 projector，epoch完成后的安全暂停，独立真实DINO图像缓存。沿用LR1e-6/embedding5e-6、LoRA64/128/.05、batch1/GA8（7rank有效batch56），每10步保存、已验证epoch后清理其覆盖的step。七卡最大前缀更新/保存/恢复验收后才正式训练。
准备运行根：/mnt/nimloth/outputs/experiments/sft2-rollout2000/20260911T072059Z_epoch5_query_converge。Stage1保持停止；源epoch005和旧数据不覆盖。DINOteacher固定facebook/dinov2-large revision47b73eefe95e8d44ec3623f8890bd894b6ea2d6c，真实缓存，不伪造Qwen缓存父manifest。

## 2026-09-11 Stage 2 K64 覆盖授权
用户明确取消当前K16/4x4正式训练，改为K64/8x8网格。其余已确认合同保持：B epoch_005初始化、a100-1七卡GPU0..6、LM/DINO权重各1、batch1/GA8、学习率和LoRA配置不变、至少2epochs且连续2轮相对改善不足1%停止。K16运行保留为已取消证据，不恢复其优化器；K64使用新的运行根、新的64-slot DINO缓存、完整输入审计和独立七卡保存恢复门禁，不能复用4x4缓存或输出。

## 目标与授权
用户明确要求先修正stage1代码，使之符合src/nimloth/training/sft/spec.md的纯回答LM监督，再在a100-1重新开始stage1。已有任务、专用branch与服务器授权继续有效。此前K1/K16实验方案均被此纠正替代，不启动旧launch。

## 要求
1. Stage1没有query数量K，不插入/生成query块，不把latent query token纳入回答目标；只对记录中的CoT/动作回答进行teacher-forcing LM监督。Stage2仍保留K/query/DINO目标。
2. 修正实际CLI、数据预处理、缓存身份、格式指标与checkpoint身份；旧带query缓存/恢复状态不得静默复用。已有转换JSONL含latent块，需要显式可审计的format-only处理，保留真实CoT、动作与截图，不改写原始文件。
3. 对有query数量或模式覆盖的stage1调用明确报错，不允许更名后偷偷保留K。
4. 修正及回归检查后同步已提交源码，在a100-1用1709训练/193验证（2000来源、98排除）新建一次run；模型/mnt/nimloth/checkpoint/hf_actor。8GPU、训练至收敛、batch1/GA8、LR1e-6、embedding5e-6、LoRA64/128、maxlen20000/pixels100352、FA2、seed42；每个运行段6h，时限暂停可续训，不代表收敛。
5. 每10optimizer steps保存；每个epoch成功且其checkpoint核验后，仅删该epoch已经覆盖的中间step checkpoint。失败保留所有产物。原始数据和旧缓存/失败run不删。

## 验收
- 测试证明stage1实际编码input/labels没有latent query token，纯回答CE；stage2 query接口与行为不回归。
- CLI/缓存/checkpoint/格式指标符合阶段边界，历史query产物不能被错误恢复。
- 真实8GPU训练产生有限loss和optimizer step，并保存完整可追溯启动和交接记录。
- epoch/val/清理未完成时如实报告；val193unique在现有DDP sampler有200采样项，不当作无padding均值。

## 不在范围
除人类明确批准的SFT1动作加权伪代码外，不修改人类spec.md；不启动stage2/WM/rollout评估，不删除其他实验产物，不调整已选训练预算。

## 已确认收敛合同
人类明确拒绝固定1epoch并选定：至少2epochs，以内部验证集LM loss为准，连续两轮相对改善不足1%停止。未收敛不写完成/收敛标记；时限保存并exact resume继续，不把预算当质量结论。预处理必须先完整完成再训练。

## 2026-09-10 FSDP追加授权
用户明确同意将当前DDP改为FSDP分片训练，保留a100-1单机8卡、完整序列、有效batch64、原LM目标、每10步保存及epoch完成清理、验证loss相邻两轮改善不足1%收敛。无旧完整checkpoint，新run初始化原模型并复用已验证format缓存。

## 已审核动作加权伪代码
用户批准 spec.md 中 sft1_step 的唯一修改：目标动作起始/结束/8个编号token赋予大于1的权重，其余有效回答token权重1，按有效权重和归一化token CE。拒绝FP32可训练参数方案；维持既有精度与学习率。具体权重待用户回答，不能默认10已获批准。实现后验证再继续，训练目前停止。

## 2026-09-11 最新覆盖授权：epoch7 / selective LM
用户批准 Stage2/3 spec 伪代码：Stage2 全部轨迹 DINO、成功轨迹 LM；Stage3 全部窗口 WM/value/DINO、成功轨迹起点 LM。成功标记继承完整原始轨迹。Stage2 由新 termination-correct Stage1 epoch_007 初始化，K64/8x8、a100-1 七卡及已确认的收敛/保存设置不变；不恢复旧 epoch5 运行的 optimizer。当前工作是实现已批准规范并完成门禁。
