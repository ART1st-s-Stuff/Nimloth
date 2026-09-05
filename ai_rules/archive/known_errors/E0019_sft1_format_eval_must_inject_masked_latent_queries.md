# E0019：SFT1 label mask、推理协议与 format eval 必须一致

错误：同时设置 `mask_latent_query_labels=true`，又从 user prompt 自由生成并要求模型输出完整 latent query block。Masked query 仍可通过后续 action CE 学习输入 embedding/hidden，但没有 CE 教它生成这些 token，所以该组合的 `format_correct_rate=0` 无法判断 action 学习。

正确做法取决于运行协议：若模型必须像旧 SFT1/VAGEN 一样自主输出完整格式，SFT1 应监督 query token labels；若框架会确定性注入 query block，才应 mask labels，并让 evaluator 使用同样的注入协议。不能把其中一种训练方式配上另一种推理检测。Best checkpoint 不能依赖与训练协议不匹配的 format rate。
