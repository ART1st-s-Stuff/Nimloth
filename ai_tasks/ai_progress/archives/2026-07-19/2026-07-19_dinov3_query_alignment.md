# 2026-07-19 DINOv3 query-state alignment

## 任务目标

参考 DeepSight（arXiv:2605.10564），在独立分支为 Nimloth 的 latent query state 增加 DINOv3 语义特征对齐；本阶段不提交训练、评估或 Slurm 实验。

## 分支与工作区

- 分支：`feat/dinov3-query-alignment`
- worktree：`/workspace/remote2/nimloth-feat-dinov3-query-alignment`
- 起点：`dev` commit `5628cc5`

## 已完成

- 阅读项目行为、记忆、代码、实验和进度规则。
- 阅读论文全文并检查论文官方代码仓库 `hotdogcheesewhite/DeepSight`。
- 确认官方实现并非直接对齐少量全局 query state：它为 5 张 256×256 BEV 图像创建 `5 × (1 CLS + 4 register + 256 patch) = 1305` 个 BEV query token，跳过 4 个 register token后，用线性层把对应 Qwen hidden 从 2048 投影到 DINOv3 ViT-L/16 的 1024 维特征，以 MSE 对齐 5 张未来 BEV 的 CLS+patch features；训练总 loss 中该项权重为 2。
- 确认 Nimloth 当前使用 k=8 latent query hidden，经 `StateProjector` flatten 后成为单个 1024 维 WM state；现有 rollout 数据只有逐步 RGB observation 路径，没有 DeepSight 所需的未来 BEV 图像或 1305 个 query slots。

## 人类确认的方案

- 保留现有 k=8 query state 与逐步 RGB rollout 数据。
- 对齐目标使用选择 action 时可见的 current RGB observation。
- 整组 query hidden 先经过现有 `StateProjector` 得到 1024 维 WM state，再直接对齐 frozen DINOv3 ViT-L/16 final CLS feature。
- 不增加 trainable alignment head；state 与 target 维度不一致时立即报错。

## 已完成实现

- 新增 `FrozenDINOv3Encoder`：默认显式加载 `facebook/dinov3-vitl16-pretrain-lvd1689m`，固定 eval/frozen，输出 detached final CLS；加载失败时不回退 DINOv2 或其他模型。
- 新增 current-query-state ↔ current-RGB DINOv3 CLS MSE，并按 `lambda_dinov3` 接入 train/validation/CSV/W&B metrics。
- compact/legacy preprocess cache collator 均恢复传播 `current_image_path`，旧 cache 不需要为此重建。
- 新增专用 config `configs/training/sft2/latent_wm_value_k8_dinov3.yaml`：k=8、ViT-L/16、CLS、`lambda_dinov3=1.0`；原 k8 config 保持关闭，避免改变旧流程。
- checkpoint 的 HF config 与 training invariants 记录 DINO source/feature/lambda；alignment resume 拒绝缺少 invariants 或配置不一致的 checkpoint。DINO teacher 权重不复制进 Nimloth checkpoint。
- 更新 backbone/training/config 索引与单元测试。

## 主要文件修改

- `src/nimloth/backbone/dinov3.py`
- `src/nimloth/training/sft2/{loss,trainer,evaluate,preprocess_cache,checkpoint,cli}.py`
- `src/nimloth/training/common/config.py`
- `configs/training/sft2/latent_wm_value_k8_dinov3.yaml`
- `tests/test_dinov3_encoder.py`
- `tests/training/sft2/{test_sft2_loss,test_cli_config,test_preprocess_cache}.py`

## 验证

- RED：新增测试最初因 `nimloth.backbone.dinov3` 不存在而 collection fail。
- `python3 -m compileall -q src/nimloth tests`：通过。
- `git diff --check`：通过。
- DINO/config/loss/cache targeted tests：`24 passed`。
- SFT2 + latent/query/dataset related suite（排除下述 dev 基线坏测试）：`75 passed`。
- 未提交或启动训练、评估、Slurm、GPU 或远程实验。

## 已知基线问题

- 未修改的 `tests/training/sft2/test_trajectory_prefix_encoding.py::test_two_step_prefix_tokenization_is_stable` 在 dev 原始代码中先使用局部变量 `token_id_map`、后赋值，单独导致 `UnboundLocalError`；确认同一错误存在于 `dev`，本任务未越界修复。

## 风险 / 尚未验证

- 未下载受许可控制的真实 DINOv3 ViT-L/16 权重，也未做 GPU forward；真实 checkpoint access、Transformers 版本兼容、显存/吞吐和 loss 数值范围均未验证。
- DINOv3 CLS 与 SIGReg/WM/value 联合优化的效果只能通过后续经人类确认的实验判断。
