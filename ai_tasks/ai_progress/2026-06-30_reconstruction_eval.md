# 2026-06-30 Reconstruction eval implementation

## 目标
为 Nimloth SFT2/RL checkpoint 添加 post-hoc reconstruction diagnostic，用于评估 WM latent / predictor 是否学到可解码的信息。

## 当前计划
1. 添加 `WMImageDecoder`。
2. 添加 reconstruction decoder 训练入口。
3. 添加 offline reconstruction eval 入口与 metrics。
4. 添加基础单元测试。
5. 运行轻量测试。

## 已完成步骤
- 已阅读 LeWM paper 摘要与相关章节：reconstruction 是 visualization-only diagnostic，不是 LeWM 主训练目标。
- 已检查当前 SFT2/RL WM loss：latent MSE + value loss，无 pixel decoder。
- 已实现第一版 post-hoc reconstruction 代码，未接入 SFT2/RL loss。

## 文件修改
- 新增 `src/nimloth/wm/reconstruction.py`：`WMImageDecoder` 与 checkpoint save/load。
- 修改 `src/nimloth/wm/__init__.py`、`src/nimloth/wm/README.md`：导出/记录 decoder。
- 新增 `src/nimloth/training/reconstruction/`：decoder 训练 CLI/trainer/README。
- 修改 `src/nimloth/training/README.md`：登记 reconstruction 训练包。
- 新增 `src/nimloth/eval/reconstruction.py`：offline reconstruction eval。
- 修改 `src/nimloth/eval/README.md`：登记 reconstruction eval。
- 新增 `configs/training/reconstruction/defaults.yaml` 和 `eval.yaml`。
- 新增 `tests/test_wm_reconstruction_decoder.py`、`tests/eval/test_reconstruction_metrics.py`。

## 验证
- `PYTHONPATH=src python -m compileall -q src/nimloth/wm/reconstruction.py src/nimloth/eval/reconstruction.py src/nimloth/training/reconstruction tests/test_wm_reconstruction_decoder.py tests/eval/test_reconstruction_metrics.py` 通过。
- `python -m pytest ...` 未运行成功：当前容器没有 `pytest`。
- 无法执行 torch runtime smoke test：当前容器没有 `torch`。

## 运行进展（2026-06-30 晚）
- reconstruction full decoder 训练已多次 resume，当前有效运行切换到 `dgx-03`，Slurm job `462610`。
- 因用户指出不能使用 `dgx-56`，已取消 `dgx-56` 上 reconstruction step；`dgx-56` 只剩 hold job，不再跑本训练。
- 当前从 `/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-06-30/reconstruct_decoder_sft2_full_4epoch/step_000021000` resume。
- 最新确认日志已继续输出到约 `step=21450`，W&B project 强制为 `nimloth`。
- W&B preview logging 已改为同时上传 `reconstruction/val_preview_table` 与独立 chart keys `reconstruction/preview_00` ... `reconstruction/preview_04`；val previews 按不同 rollout record 均匀取样。

## 待确认问题
- 是否要继续把 reconstruction metrics 接入 SFT2/RL 主 validation logging。
- 是否要继续实现多步 open-loop rollout reconstruction。
