# 修正SFT1符合spec并启动2000来源rollout训练

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
不修改人类spec.md，不启动stage2/WM/rollout评估，不删除其他实验产物，不调整已选训练预算。

## 已确认收敛合同
人类明确拒绝固定1epoch并选定：至少2epochs，以内部验证集LM loss为准，连续两轮相对改善不足1%停止。未收敛不写完成/收敛标记；时限保存并exact resume继续，不把预算当质量结论。预处理必须先完整完成再训练。

## 2026-09-10 FSDP追加授权
用户明确同意将当前DDP改为FSDP分片训练，保留a100-1单机8卡、完整序列、有效batch64、原LM目标、每10步保存及epoch完成清理、验证loss相邻两轮改善不足1%收敛。无旧完整checkpoint，新run初始化原模型并复用已验证format缓存。
