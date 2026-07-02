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
- 新增 `experiments/lewm_decoder_repro/setup_remote_env.sh`：按 LeWM README 安装 `stable-worldmodel[train,env]` 等依赖。
- 新增 `experiments/lewm_decoder_repro/run_smoke.slurm`：单 GPU smoke launcher。

## 验证命令和结果

- `python -m py_compile experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py`：通过。
- `bash -n experiments/lewm_decoder_repro/run_smoke.slurm experiments/lewm_decoder_repro/setup_remote_env.sh`：通过。
- 已提交：`8ccfebe feat: add lewm cube decoder repro`。
- 后续提交：`dea79f6 docs: track lewm decoder repro setup`、`7864edd docs: summarize lewm decoder repro branch`、`0fdc9b5 fix: allow lewm env setup without uv`。
- 已 push 到 `origin/nimloth-lewm-repro`，并在服务器创建/同步 worktree `/project/peilab/atst/nimloth/.worktree/nimloth-lewm-repro`。
- job `464080` 首次 smoke 已启动但失败于训练前：HF archive 解压出 `$STABLEWM_HOME/cube_single_expert.h5`，脚本按 LeWM config-style dataset name 期待 `$STABLEWM_HOME/ogbench/cube_single_expert.h5`，因此 `FileNotFoundError`。无有效 metrics/checkpoint。
- 已更新远程失败输出 README 和 `outputs/experiments/lewm_repro/progress.md`。
- 已修复 `prepare_dataset()`：若 root fallback h5 存在，自动创建 `ogbench/cube_single_expert.h5` symlink。

## 待确认问题

- 下载 HF Cube dataset 约 46GB，训练 decoder 需要 GPU；启动前需向用户确认资源与输出目录。
