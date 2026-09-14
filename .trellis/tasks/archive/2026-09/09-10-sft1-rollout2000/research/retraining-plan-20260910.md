# 最新授权（2026-09-10）
用户明确批准先采用修正后的prompt再启动B。仅B，20optimizer updates，不重跑A、不执行C、不自动转正式训练。保留原hf_actor、原数据、旧run；修正数据另存，重建缓存。新action编号行按语义词均值初始化，action起止按已审查B复制EOS；BF16/currentLoRA、LR1e-6/embed5e-6、weight8、8A100、batch1GA8不变。整体最多4小时，失败不自动重试。具体commit/唯一输出/PID启动时补记。

# B 方案追加指令（2026-09-10）
用户授权只做B，不重跑A、不执行C；动作编号初始化源改为语义词：forward/backward/right/left/rotate right/rotate left/up/down。多token语义词组在各自输入embedding和输出head取均值再转换BF16；action_start/end仍按B用EOS初始化。保持embed/head5e-6、LoRA1e-6、weight8；仅20updates的有限B，不自动转正式收敛训练。
用户要求开始前核对原始rollout prompt。发现真实system/user中动作名称输出要求未完整转换；已提出仅纠正输出要求与完整token映射，原始数据不改，并按上一计划的prompt审查点请求用户批准。批准前不启动B。
实现边界：新增独立CPU模型初始化脚本和小矩阵测试；新模型产物供B训练使用，原hf_actor和所有旧run保留。初始化审计明确区分原始参数与派生checkpoint。

---
以下为此前提案，仅上述更新后的B范围有效：

# SFT1 新训练计划（待用户审核，未授权启动）

## 当前证据与目标
现有 weight8 run 已按用户要求请求保存后暂停。epoch2 step54 的验证LM loss 0.87370795、自由生成格式0/32；CPU四样本全部输出普通数字后EOS，56–84tokens，无512截断。单条同参考前缀 action_start概率从1.975e-8到1.226e-6；即便提供action_start，正确action3概率仅1.018e-7。累计head行变化约245–295/2048元素，不把它当逐步舍入比例或单一因果证明。
目标仍是纯SFT1：模型自主生成完整think/action格式。无query/latent/DINO，无强制token掩盖失败。

## 固定项
从原始/mnt/nimloth/checkpoint/hf_actor新建运行，fresh optimizer/history；保留所有旧run。a100-1单机8A100 FSDP；冻结基座+LoRA r64 alpha128 dropout0.05；保留目前BF16 embedding/head及既有LoRA精度，不引入FP32 master或新增FP32 delta。数据1709train/193内部验证，完整CoT/动作/图像、maxlen20000、pixels100352；batch1/rank GA8，有效batch64，seed42，weight8（其余回答及EOS权重1）。完整词表加权回答CE，不做restricted-eight-action CE。

## 实施前检查
核验十个新token均为独立ID；无论模型有无预留行，都按明确方案写入并验证输入embedding和输出head。核验二者是否实际共享，分别从各自来源行初始化；不重新初始化resume权重。新增token行单独参数分组，避免整张head应用较高LR，保证非action行维持既有5e-6。保留LoRA1e-6。以实际8rank优化器更新、保存/重载一致性、梯度非零和非action行更新规则作门禁。
同时审查渲染prompt中的“只能回答动作名”和“必须使用special-token动作块”冲突；先形成逐条diff供审核，不擅改数据或真实CoT。若清理prompt，必须固定为各对照共同输入并重建身份正确的缓存，不能与初始化效应混淆。

## 有限对照（建议数值，待审核）
三组均从原始基座开始，相同数据顺序，每组20个optimizer updates，0/10/20保存诊断；试验合计60updates。它是筛选预算，不是正式epoch上限。
A：保留旧预留行初始化，全部embed/head LR5e-6（同目标基线）。
B：显式初始化，其他设置同A。候选方案：action_start/action_end分别复制各自输入/output矩阵中的EOS行；八个动作编号复制其对应普通数字0..7的行（必须验证数字原有编码为单token，否则先报告再调整方案）。这是待检验的起点假设，不保证格式成功，不改变EOS ID或停止规则。
C：同B，但十个action行LR5e-5，其他embedding/head仍5e-6、LoRA1e-6。比较B/A分离初始化效果，C/B分离学习率效果；不无界扫参。
预估3组含初始化、检查及生成约2–4小时，8卡；总4小时硬预算，预算到达保存并报告，不自动重提或扩展。实际预算需启动前按预检吞吐核实。

## 对照证据与选型
固定同一32条内部验证首轮prompt，各组在0和20步保存自由生成完整原文、token IDs、EOS/长度停止原因，128token主指标不变；对达到128上限的案例另做512诊断，不混合统计。第10步只做固定4样本诊断与参考边界概率、token分组CE及行delta。验证LM仍使用完整193条（报告现有分布式padding口径），同时报告动作起始/编号/结束的分项CE。禁止把同一次实验调参用过的内部验证称作未见测试。
选型先看自由格式改善，再看未加权验证LM与重复/胡言乱语/动作塌缩；不得仅因较小weighted loss选择。若三组仍0/32或收益不明确，停在审查点，提供输出而不自动开始正式训练。正式验收建议完整193条首轮自由生成>=95%，这是拟议内部格式目标，非环境成功率保证，需用户批准。

## 正式训练
只在对照结果经审核后，从原始hf_actor按选中配置fresh启动，不继承pilot optimizer。先处理/核验缓存再训练。沿用用户收敛规则：至少2epoch，未加权验证LM连续2轮相邻改善不足1%停止；按收敛训练，不设置固定epoch终点。每epoch保存32条可对照原文，收敛后193条完整格式评估；loss收敛但格式不达标则标记未通过，不自动无限延长。每10步checkpoint，epoch提交核验后删除本run被覆盖中间ckpt；保留epoch/best/final。6小时段，安全暂停续训；意外错误停下诊断。

## 审核边界
此文只提出计划，不修改人类spec.md、训练实现或数据，不启动pilot/正式运行。初始化来源、5e-5新行LR、三组20步/总4h预算及95%内部格式验收均为待审核提案。
