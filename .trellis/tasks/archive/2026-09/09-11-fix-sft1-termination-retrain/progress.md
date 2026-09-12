# 当前进度

## 已完成

- 代码提交 `2a8ac7122c8e1f389e01cd360cb592cb9d308d80`：严格 EOS/全文格式合同、仅八个动作编号权重 8、成功子集准备、统一 Stage 1 格式门禁及评估入口。
- a100-1 数据审计：B 源 train 1709 条中成功 1149 条/9012 assistant turns；heldout 193 条中成功 134 条/972 assistant turns。train/heldout 的 record ID、`(eval_set, seed)`、裸 seed 重叠均为 0，图片缺失为 0。
- 全集 `rotate left` 共 21 次/19 条轨迹；成功 train 中 3 次，成功 heldout 中 0 次。人类明确成功子集不要求八类动作全覆盖；实现、任务需求和测试已同步，18 个聚焦测试通过。
- 8-rank BF16/FSDP 机制门禁通过：输出 `/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T114820Z_termination_fix_fsdp_gate`，返回码 0；精确保存恢复、生成和 epoch 导出均通过。该门禁使用小模型，仅证明机制。
- 正式运行已通过数据派生和 BF16 语义初始化。初始化核验 825 个 tensor；动作边界来自 EOS，动作编号来自既定语义词；Stage 1 `latent_token_count=null`。

## 运行中

- 主机：a100-1（直接 SSH，无 Slurm）。
- 运行目录：`/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T115012Z_success_only_termination_w8`。
- 控制目录：同名加 `_controller`；完整合同为 `launch_contract.json`，workflow PID 为 `616605`，日志为 `workflow.log`。
- 截至 2026-09-11 11:52 UTC 正在构建全新的成功子集 preprocess cache，尚未进入 GPU 训练。
- 监控：Codex heartbeat `sft1` 每 10 分钟检查；无实质变化时不通知，失败时不自动重启。

## 后续门禁

- 缓存完成后核验 manifest、tensor 数、无 latent/截断以及标签以 `<|action_end|><|im_end|>` 结束。
- 进入训练后核验 8 ranks、BF16、FSDP/LoRA、步级 loss、NaN/OOM 和每 10 步完整 checkpoint。
- 只在完整成功-heldout 验证 LM loss 满足既定规则后接受收敛 final；随后执行固定 32 条严格格式门禁，至少 31/32 才运行统一 Base60 + CommonSense60 环境评估。
