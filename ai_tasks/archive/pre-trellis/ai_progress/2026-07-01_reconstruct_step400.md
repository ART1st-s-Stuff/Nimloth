# 2026-07-01 reconstruction from SFT2 lejepa step400

## 任务目标
- 使用用户指定 checkpoint：`/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/ckpt_step400_preserved`
- 先导出完整 HF model
- 再复用现有 post-hoc reconstruction 训练代码启动 decoder 训练，并使用现有 trainer 的 train + val/eval 流程

## 当前计划
1. 服务器上准备干净 worktree。
2. 提交 hold job 占住 1 张 GPU。
3. 导出 step400 LoRA+vision checkpoint 为完整 HF model。
4. 启动 reconstruction decoder 训练。
5. 监控直到确认健康启动。
6. 更新进度文件与 `AI_branch_progress.md`。

## 已完成步骤
- 已阅读 AGENTS / ai_rules / server / slurm / experiment-start 规则。
- 已核实用户给定 checkpoint 路径存在，且为 LoRA adapter checkpoint（含 `state_proj.pt`、`wm_predictor/`、`vision_full_state.pt`）。
- 已核实现有 reconstruction trainer 需要完整 HF `--model`，因此必须先导出 merged model。
- 已核实现有 trainer 为：训练 `WMImageDecoder`，冻结 Qwen / `StateProjector` / `LatentWMPredictor`，并在训练过程中记录 W&B 图像与按 epoch 做 val reconstruction eval。
- 已向用户说明实验入口、冻结/训练模块、输出目录、resume 机制、资源预估；用户已确认执行。
- 已在服务器创建干净 detached worktree：`/project/peilab/atst/nimloth/.worktree/recon-step400-20260701`，commit `13ea39d71e19b57c1eea6fe60d2204f8a5b222c2`。
- 用户补充允许占用 `dgx-12` 空闲 GPU 后，先提交了 `preempt / dgx-12 / gpu:7` 的 job `463506`；但它长时间 `PENDING (Priority)`，预计开始时间一度推迟到 `2026-07-04T16:19:02`。
- 按用户新指示，已取消 `463506`，切换到 4GPU 空闲节点方案。
- 首次 `dgx-29 / gpu:4` job `463508` 与重提 `463509` 都在 1 秒内失败：
  - 原因 1：batch 脚本 README heredoc 中反引号触发 shell command substitution；
  - 原因 2：脚本在 `set -e` 下 `source /etc/profile`，该 source 直接使 shell 退出。
- 已修正远程 batch 脚本并重新提交 `463510`：
  - 脚本：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_sft2_lejepa_align_fn1_step400_dgx29g4/launch.slurm`
  - 输出目录：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_sft2_lejepa_align_fn1_step400_dgx29g4`
  - 资源：`normal / dgx-29 / gpu:4`（按用户要求预留 4 卡，实际训练只用 1 卡）
  - 流程：先导出 `export_best_hf`，再跑现有 `nimloth.training.reconstruction.cli`；若已有 `step_*` checkpoint 会自动 `--resume`。
- `463510` 的 export 已完成，但训练阶段失败：远程 worktree 的 `external/le-wm` 未初始化，报错 `ImportError: LeWM file missing: .../external/le-wm/module.py`。
- 已在远程执行 `git -C /project/peilab/atst/nimloth/.worktree/recon-step400-20260701 submodule update --init external/le-wm`，并用 `.venv-vagen-main` 验证 `from nimloth.training.reconstruction.trainer import train_reconstruction_decoder` 成功。
- 已重提 job `463525`，单卡 trainer 曾健康运行于 `dgx-29`：
  - `slurm-463525.out` 显示 trainer 成功加载 Qwen shard、完成 W&B 登录并创建 run `e7c2vd5m`
  - W&B URL：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/e7c2vd5m`
  - `wandb_val_preview_logged` 在 `step=0` 已记录 5 个 preview
  - `train_step_log.csv` 已写入首条训练日志：`epoch=1, step=500, loss=0.16091777384281158`
- 用户随后明确要求 4 卡并行，因此已在本地实现 reconstruction trainer 的 DDP 支持：
  - 训练集改为 `DistributedSampler`
  - 只对 `WMImageDecoder` 做 DDP；Qwen / StateProjector / WM predictor 仍为每 rank 冻结副本
  - 只由 rank0 做 W&B / val eval / checkpoint / CSV logging
  - metadata 记录 `world_size`、`per_rank_batch_size`、`effective_batch_size`
- 本地代码提交与同步：
  - `f73ccac feat(reconstruction): add decoder DDP training`
  - `2d96ff7 tune(reconstruction): add decoder lr warmup`
- 远程 worktree 当前已切到 `2d96ff758ac5a492fb0db86f00312950ea2181c1`。
- 已取消单卡 job `463525`，随后尝试 `dgx-29` 的 4GPU DDP job `463532` 与 `dgx-18` 的 4GPU DDP job `463536`；两者都因调度优先级未立即启动。
- 为争取 backfill，已把 `dgx-18` 版 DDP job 时间限制改为 `8h`，重提为 `463537`。该 run 曾健康运行并在 SFT2 `ckpt_step1000_preserved` source checkpoint 上创建 W&B run `yohr0763`，但用户根据图像纯色现象要求暂停当前训练并尝试调参。
- 用户随后澄清“step1000”指的是 SFT2 checkpoint `/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/ckpt_step1000_preserved`，不是 reconstruction decoder 的 step checkpoint。
- 因此已取消 `463585`，并基于同一个 SFT2 step1000 source checkpoint 发起保守调参版：
  - 新输出目录：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_sft2_lejepa_align_fn1_step1000_dgx12g4_preempt_lr5e5_wu1500`
  - 新 job：`463623`
  - 资源：`preempt / dgx-12 / gpu:4 / 4h`
  - 调参内容：`lr 1e-4 -> 5e-5`，新增 `lr_warmup_steps=1500`，其余保持不变
  - 为避免重复消耗，复用了旧 step1000 run 已导出的 full HF model：`...step1000_dgx18g4_ddp4/export_best_hf`
- 当前有效 job：`463623`，已在 `dgx-12` 健康启动：
  - 4 个 rank 已成功加载 Qwen shard
  - W&B run 已创建：`bpfk4266`
  - W&B URL：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/bpfk4266`
  - `wandb_val_preview_logged` 在 `step=0` 已记录 5 个 preview
  - 当前需要继续跟踪更早期 loss 与图像是否比暂停前 run 更快脱离纯色解

## 关键路径
- 导出脚本：`experiments/training/sft2/export_lora_visionfull_checkpoint.py`
- 训练入口：`python -m nimloth.training.reconstruction.cli`
- 数据：`experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format/{train_all,val_all}.jsonl`

## 待确认/风险
- 服务器现有 `.worktree/dev` 不够干净，因此本任务持续使用独立 detached worktree，避免脏工作区干扰。
- 交互式 `srun --overlap` 检查作业时会被 cgroup 限制只看到部分 GPU；是否 4 卡并行以 torchrun 4 个 rank 进程和 trainer 日志中的 `world_size=4` 为主要证据。
- 当前 tuned run `463623` 已确认创建 W&B run 并写出 `step=0` preview；下一步重点是观察前几个 step logging 点和 preview 图像是否较上一版更快摆脱纯色。
- 这次调参只测试了更保守的优化（`lr=5e-5`, `warmup=1500`）；如果图像仍快速塌成纯色，再考虑改 decoder 容量或更换诊断目标。

## 2026-07-02 更新：decoder 修复后全量 reconstruction 仍不成功

### 已完成
- 确认 LeWM repo 本身没有显式 image decoder 配置；此前将 decoder 具体结构归因于 LeWM 没有足够证据。
- 修复 `WMImageDecoder` 并提交：`a3eab99 fix(reconstruction): replace broken cross-attn decoder with self-attn ViT decoder`。
  - 旧 decoder：patch queries cross-attend 到单个 memory token，单样本 overfit 卡在均值图。
  - 新 decoder：state expand 到 patch tokens + positional embedding + self-attention blocks + patch RGB head。
- 单样本 overfit job `463770` 完成：
  - 输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/overfit_test2`
  - final_loss=8.84e-06，pred_std=0.1620 接近 target_std=0.1625
  - 结论：新 decoder 可以完全拟合单样本，decoder 本体的均值坍缩问题已修复。
- 全量 reconstruction job `463782` 完成：
  - 输出：`/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-07-01/reconstruct_decoder_fix_sa_step1000_dgx06g4_lr5e5_wu1500`
  - 状态：`COMPLETED 0:0`，运行 05:27:51，`preempt / dgx-06 / gpu:4`
  - W&B：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth/runs/irtwhtxd`
  - 配置：4 epochs，DDP world_size=4，per-rank batch_size=1，L1，`lr=5e-5`，`lr_warmup_steps=1500`

### 结果
- 用户观察：50k+ step 后预览仍为明暗色块，不是可识别图像。
- val metrics：
  - epoch1：pred_mse=0.04056，oracle_mse=0.03617，copy_mse=0.03634
  - epoch2：pred_mse=0.06088，oracle_mse=0.03471，copy_mse=0.03493
  - epoch3：pred_mse=0.07217，oracle_mse=0.03366，copy_mse=0.03396
  - epoch4：pred_mse=0.09397，oracle_mse=0.03390，copy_mse=0.03410
- 初步结论：新 decoder 解决了单样本 overfit 问题，但全量数据上的 oracle reconstruction 仍只接近 copy baseline，pred reconstruction 还随训练变差；继续同配置加长训练不太可能解决。

### 后续建议
1. 跑 train subset / val subset 的 offline samples，对比是否 train 也只学成色块。
2. 做 latent-image 检索或简单 probe，检查 `state_proj(Qwen <latent_state>)` 是否携带可恢复视觉外观。
3. 若继续做“看图”诊断，优先考虑输入换成 Qwen visual patch tokens / earlier hidden states，或改 spatial bottleneck + CNN/VAE-style decoder；Diffusion/Flow Matching 暂不优先。
