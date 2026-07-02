# 2026-07-02 LeWM OGBench-Cube decoder reproduction

## 目标

验证 Nimloth 自己写过的 `WMImageDecoder` 在适配 LeWM 原文 latent 规模后，能否从官方 LeWM OGBench-Cube encoder 的 192-dim `[CLS]` 表示重建原图。

本任务目标已由用户明确更改：不是验证 LeWM 原文 decoder 本身，而是验证 Nimloth decoder 在 LeWM 官方 latent 上的重建能力。

## 当前计划

1. 使用官方 `external/le-wm` repo、HF `quentinll/lewm-cube` checkpoint 和 `quentinll/lewm-cube` dataset。
2. 构造独立实验脚本：加载官方 LeWM encoder，冻结 encoder，训练 `WMImageDecoder(emb_dim=192, image_size=224, patch_size=16, ...)`。
3. 输出原图/reconstruction 对比、指标和 checkpoint。
4. 昂贵下载/GPU 训练前按实验规则向用户确认。

## 已完成步骤

- 创建本地分支 `nimloth-lewm-repro`。
- 创建 worktree `/workspace/remote2/nimloth-nimloth-lewm-repro`。
- 初始化官方 `external/le-wm` submodule。
- 已确认 LeWM paper Appendix D 的 visualization decoder 是 `[CLS]` 192-dim → image 的后训练诊断；但本任务改为测试 Nimloth decoder。
- 已检查 Nimloth `WMImageDecoder`：当前是 state vector 线性展开到所有 patch tokens + self-attention + RGB patch head；可通过 config 适配到 `emb_dim=192, image_size=224, patch_size=16`。

## 文件修改

- 新建本进度文件。
- 新增 `experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py`：加载官方 LeWM Cube checkpoint，冻结 encoder，训练 Nimloth `WMImageDecoder(emb_dim=192, image_size=224, patch_size=16)`。
- 新增 `experiments/lewm_decoder_repro/README.md`：记录目标、边界、命令和输出。
- 新增 `experiments/lewm_decoder_repro/setup_remote_env.sh`：安装 `stable-worldmodel[train]`、`transformers==4.55.4`、`hdf5plugin` 等依赖。
- 新增 `experiments/lewm_decoder_repro/run_smoke.slurm`：单 GPU smoke launcher。

## 验证命令和结果

- `python -m py_compile experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py`：通过。
- `bash -n experiments/lewm_decoder_repro/run_smoke.slurm experiments/lewm_decoder_repro/setup_remote_env.sh`：通过。
- 已提交：`8ccfebe feat: add lewm cube decoder repro`。
- 后续提交：`dea79f6 docs: track lewm decoder repro setup`、`7864edd docs: summarize lewm decoder repro branch`、`0fdc9b5 fix: allow lewm env setup without uv`。
- 已 push 到 `origin/nimloth-lewm-repro`，并在服务器创建/同步 worktree `/project/peilab/atst/nimloth/.worktree/nimloth-lewm-repro`。
- job `464080` 首次 smoke 已启动但失败于训练前：HF archive 解压出 `$STABLEWM_HOME/cube_single_expert.h5`，脚本按 LeWM config-style dataset name 期待 `$STABLEWM_HOME/ogbench/cube_single_expert.h5`，因此 `FileNotFoundError`。无有效 metrics/checkpoint。
- job `464086` 失败于训练前：`stable_worldmodel` 实际在 `$STABLEWM_HOME/datasets/` 下解析 dataset；同时 transformers 5.x 的 ViT key 与官方 HF weights 不匹配。无有效 metrics/checkpoint。
- job `464090` 失败于训练前：HDF5 reader 未自动注册且缺 `hdf5plugin`。无有效 metrics/checkpoint。
- 已修复：
  - `prepare_dataset()` 创建 `$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5` symlink。
  - setup pin `transformers==4.55.4`，确保官方 LeWM weights exact load；脚本现在默认拒绝非 exact checkpoint load。
  - 显式注册 HDF5 reader，并安装 `hdf5plugin`。
- job `464091` 完成：`/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/2026-07-02/nimloth_decoder_cube_smoke_464091`。
  - 资源：preempt / dgx-39 / 1 GPU，运行 00:01:55，COMPLETED 0:0。
  - 数据：官方 Cube HDF5，random_split 0.9 后 smoke subset `train_limit=2048`, `val_limit=256`；4 frames/sequence，因此 val metrics 覆盖 1024 images。
  - 最终 epoch5：train_loss=0.0377573，val_l1=0.0376028，val_mse=0.00990812。
  - preview `previews/best.png`：粗场景布局、光照块、机械臂/方块区域可见，但仍明显 blocky，尚未达到论文展示质量。

## 当前新阶段：full train + W&B

- 用户要求直接开始全规模训练，每 1k step eval 并上传 W&B。
- 已新增 step-eval / W&B / checkpoint 支持：
  - `--eval-every-steps` 每 N optimizer steps 跑 eval。
  - `eval_log.csv` 记录 step-level eval。
  - `step_*` checkpoint 保存 decoder + optimizer + global_step，可用 `--resume-checkpoint` 恢复（dataloader 会从 epoch 开头重启）。
  - `--wandb-project/--wandb-name` 上传 train/eval metrics 和 preview。
- 新增 `experiments/lewm_decoder_repro/run_full_wandb.slurm`。
- 计划启动 full train：`train_limit=0` 使用完整 train split；`epochs=1`；`eval_every_steps=1000`；`val_limit=4096`（每次 eval 16,384 张图，避免每 1k step 做完整 10% val split 的巨大开销）。

## 待确认问题

- 若需要“每次 eval 都覆盖完整 val split”，需另行确认；这会非常昂贵。
