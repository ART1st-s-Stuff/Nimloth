# Representation Ablation

配置驱动的 world-model representation 消融基础设施。

当前阶段已实现两条离线评估路径：

- `qwen_latent` representation：`<|latent_state|>` hidden 经 `StateProjector` 得到单个 WM state vector。
- `qwen_multi_latent` representation：把每个 `<|latent_state|>` 展开成 K 个相邻 latent markers，提取最后一组连续 latent hidden，形成 `(B, K, D)` token set。
- 离线评估入口：`python -m nimloth.representation_ablation.eval --config <yaml>`。
- token-set 训练入口：`python -m nimloth.training.representation_ablation.train --config <yaml>`（当前支持 `qwen_multi_latent`、direct `qwen_vision_tokens` 与 `compressed_vision_tokens` predictor diagnostic）。
- single latent 支持 predictor one-step / multi-step 诊断、value head top-k/ranking/calibration、可选 simple decoder reconstruction。
- multi latent 支持 `token_transformer` predictor 与 `pooled_mlp` value head 的训练和离线 value/predictor 指标；token-set reconstruction 和 environment fast path 尚未实现。
- vision-token predictor diagnostic 使用 Qwen vision encoder 的直接输出，不使用 `<|image_pad|>` 的 LLM hidden states；目前支持 1-step 与 4-step autoregressive predictor 训练配置。
- compressed-vision predictor diagnostic 使用 attention/Perceiver-style compressor 将 raw Qwen vision tokens 压到 K 个 tokens，再用 LeWM-style predictor loss + SIGReg 训练 compressor/predictor；offline eval 支持 one-step 与 multistep token metrics；不使用 reconstruction loss。RCDM direct-concat visualization 尚未接入。

Phase 2 基础模块：

- `qwen_tokens.py`：多 latent marker 展开、最后一组连续 latent token hidden 提取。
- `token_set.py`：token-set predictor/value head，输入输出形状为 `(B, K, D)`。

未实现的 representation/predictor/value head 类型在正式入口中仍会显式报错，避免把未接入模块误当成完整实验实现。
