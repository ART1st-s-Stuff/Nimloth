# 设计

使用独立 codex/sft2-dino-information-test worktree，基于63e8cc29090788f565bbb198d78170892f0e72ab。复用 Stage2 trainer、DINO cache 和完整轨迹 collator，分别归约成功LM回答和全部DINO回答。

先修复 _capture_last_hidden 对远端 Transformers4.49不支持 logits_to_keep 的调用；保持一次前向及数值目标，不能仅删参数而忽视全序列logits显存。以远端实际版本回归测试及真实最长序列门禁验证。

导出epoch14时保持原始adapter精度与独立embedding/head，迁移a100-2后校验hash。核对数据(prompt/termination)、教师身份、K64缓存、train/val轨迹不重叠，不能直接假定epoch7数据仍适用。

评估使用同一固定验证划分；基线均值仅由训练DINO构建。MSE/余弦先按slot、再按观察、最后按轨迹归约，并报告轨迹bootstrap区间；随机打乱对应关系与训练前模型作对照。主结论针对8x8 DINO目标的信息恢复，不等价于原始高分辨率DINO全信息恢复。

失败时保存日志并诊断，在授权预算内修复验证后重试，不盲目重提。监控仅报告变化；已退出且不再推进时结束监控，不能持续空查询。
