# Representation Ablation

配置驱动的 world-model representation 消融基础设施。

当前阶段只实现 Phase 1 的 single Qwen latent baseline：

- `qwen_latent` representation：`<|latent_state|>` hidden 经 `StateProjector` 得到单个 WM state vector。
- 离线评估入口：`python -m nimloth.representation_ablation.eval --config <yaml>`。
- 支持 predictor one-step / multi-step 诊断、value head top-k/ranking/calibration、可选 simple decoder reconstruction。

后续 Phase 会在同一 config/factory 接口下加入 multi latent tokens、compressed vision tokens 和 raw vision tokens。未实现的 representation/predictor/value head 类型会显式报错，避免把 placeholder 误当成真实实现。
