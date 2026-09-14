# 设计

使用独立 codex/sft2-dino-information-test worktree，基于63e8cc29090788f565bbb198d78170892f0e72ab。复用 Stage2 trainer、DINO cache 和完整轨迹 collator，分别归约成功LM回答和全部DINO回答。

先修复 _capture_last_hidden 对远端 Transformers4.49不支持 logits_to_keep 的调用；保持一次前向及数值目标，不能仅删参数而忽视全序列logits显存。以远端实际版本回归测试及真实最长序列门禁验证。

导出epoch14时保持原始adapter精度与独立embedding/head，迁移a100-2后校验hash。核对数据(prompt/termination)、教师身份、K64缓存、train/val轨迹不重叠，不能直接假定epoch7数据仍适用。

评估使用同一固定验证划分；基线均值仅由训练DINO构建。MSE/余弦先按slot、再按观察、最后按轨迹归约，并报告轨迹bootstrap区间；随机打乱对应关系与训练前模型作对照。主结论针对8x8 DINO目标的信息恢复，不等价于原始高分辨率DINO全信息恢复。

失败时保存日志并诊断，在授权预算内修复验证后重试，不盲目重提。监控仅报告变化；已退出且不再推进时结束监控，不能持续空查询。

## 2026-09-12 用户要求加速准备流程

实现可验证的完整审计复用：以数据、图像/缓存输入清单、tokenizer/chat template/image processor、K/grid/长度/像素限制及相关源码身份绑定审计。只更换模型权重不使输入审计失效。旧报告若缺少上述身份不直接升级为可复用证据。迁移依赖（包括缓存引用的源JSONL）在导出前验证完整。当前远端审计已完成1225/1902，保持其已固定源码和运行继续，禁止为部署优化再次重跑；优化代码先本地验证。

使用方式：新准备合同必须显式提供 input_root（包含data与dino_cache的已迁移目录）；可选 reuse_input_audit 指向之前由新版prepare_test生成的完整报告。身份不匹配时执行完整审计，旧报告不支持直接复用。仍校验输入文件字节，不再重复图像collation；目前源码指纹保守覆盖src/nimloth，代码变化会失效。当前运行使用旧commit，不改合同。

## 训练性能优化（用户授权）

跳过失败轨迹的交叉熵计算而非末尾屏蔽；成功回答归一化和全部回答DINO保持不变，验证全失败/混合批次损失与梯度。使Stage2已配置的CPU加载进程生效，使用安全worker启动方式和确定性顺序，保留Stage1默认行为。当前测试不中途热替换源码；切换须验证完整恢复身份及原始6小时预算，不能重置测试。
