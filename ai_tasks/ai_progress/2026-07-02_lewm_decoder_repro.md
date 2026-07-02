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

## 验证命令和结果

- 尚未运行实验。

## 待确认问题

- 下载 HF Cube dataset 约 46GB，训练 decoder 需要 GPU；启动前需向用户确认资源与输出目录。
