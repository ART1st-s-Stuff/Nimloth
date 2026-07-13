# E0019：SFT1 format eval 不能要求模型生成被 mask 的 latent query tokens

错误：`evaluate_format()` 从 user prompt 直接自由生成，并用正则要求输出完整 k-token latent block；但正式 SFT1 设置 `mask_latent_query_labels=true`，这些 token 的 labels 全为 `-100`，模型没有学习“生成 query token”的 CE 信号。因此 `format_correct_rate=0` 不能说明 action-format 训练失败。

正确做法：masked query 语义下，评估器应在 `</think>` 后由框架注入 k 个 latent query tokens，再评估 action block；或者将该指标明确标记为不适用。Best checkpoint 仍按 val loss 选择，不能用当前 format rate 排序。
