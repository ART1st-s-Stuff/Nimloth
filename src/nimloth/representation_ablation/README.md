# Representation Ablation

配置驱动的 world-model representation 消融基础设施。

当前阶段已实现 Phase 1 的 single Qwen latent baseline：

- `qwen_latent` representation：`<|latent_state|>` hidden 经 `StateProjector` 得到单个 WM state vector。
- 离线评估入口：`python -m nimloth.representation_ablation.eval --config <yaml>`。
- 支持 predictor one-step / multi-step 诊断、value head top-k/ranking/calibration、可选 simple decoder reconstruction。

Phase 2 的基础模块已开始落地，但尚未接入正式训练/eval 主路径：

- `qwen_tokens.py`：多 latent marker 展开、最后一组连续 latent token hidden 提取。
- `token_set.py`：token-set predictor/value head，输入输出形状为 `(B, K, D)`。

未实现的 representation/predictor/value head 类型在正式入口中仍会显式报错，避免把未接入模块误当成完整实验实现。
