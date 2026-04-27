# dev 目录说明

`dev` 目录仅用于开发期调试与实验脚本，不承载正式训练入口。

## 目录结构

- `smoke/`: 快速冒烟脚本（短时验证）
- `debug/`: 定位问题的调试脚本
- `benchmark/`: 性能与时延测试脚本
- `experiments/semantic_align/`: 语义对齐相关实验脚本
- `experiments/joint_rl/`: 联合训练相关实验脚本
- `webui/`: 可视化与交互测试脚本
- `_shared/`: dev 内可复用公共函数
- `artifacts/`: 开发脚本产物（避免与脚本混放）

## 脚本收敛

- 原 `test_eval_*` 系列已收敛为 `experiments/semantic_align/eval_smoke.py`（通过 `--mode` 切换）。
- 原 `test_dataloader*` 系列已收敛为 `smoke/dataloader_smoke.py`（参数化控制 encoder / 迭代模式）。

## 使用建议

- 日常快速验证优先使用 `smoke/` 脚本。
- 临时脚本优先放入 `debug/`，验证完成后及时清理或沉淀到 `_shared/`。
