# 2026-07-05 Latent Representation Ablation 执行进度

## 任务目标

在 `exp/latent-repr-ablation` 分支上按配置驱动方式实现 representation ablation 的基础设施，使后续能通过不同 YAML 配置启动不同实验，而不是为每个实验改代码。

## 当前计划

1. Phase 0：阅读并冻结接口/schema，确认可复用模块。
2. Phase 1：实现 baseline single-latent 的配置驱动离线评估链路。
3. 本地单元测试通过后，提交代码。
4. 需要服务器 smoke 时，启动指定模型 `openai-codex/gpt-5.5` 的子 agent 执行；smoke 不上传 W&B。

## 已完成步骤

- 已在 `ai_tasks/latent_repr_ablation_plan.md` 补充分 Phase 和配置驱动要求，并提交 `2e20222`。
- 已阅读现有 SFT2 dataset、WM predictor/value head、reconstruction evaluator、checkpoint helper。
- 已实现 Phase 1 single `qwen_latent` baseline 的配置驱动离线评估基础设施：严格 YAML schema、Phase-1 validator、module loader、value/predictor metrics、config-driven eval CLI。
- 已修正 eval 汇总逻辑：value/ranking/calibration 和 one-step predictor 指标按全体 encoded transitions 汇总，避免 batch size=1 时 calibration/AUC 失真。
- 已新增 `init.sft2_checkpoint` 便利配置；设置标准 SFT2 checkpoint dir 后可自动推导 qwen/state_proj/wm_predictor/value_head 路径。
- 根据 code-review 子 agent 的 blocking finding，已修复 value head checkpoint 缺失时可能静默使用随机初始化权重的问题：Phase-1 validator 和 loader 现在要求 `value_head_checkpoint/value_head.pt` 真实存在，否则报错。
- 根据 server-smoke 子 agent 结果，已修复 `test_metrics.py` 对 float32 结果精确相等的问题，改用 `pytest.approx`。
- 根据 server-smoke 子 agent 结果，已把 `LatentWMPredictor` / `StateProjector` / `ValueHead` / decoder 导入移入 loader 函数，避免 `import nimloth.representation_ablation.eval` 在未初始化 `external/le-wm` 时失败；真正评估加载 predictor 时仍会要求真实依赖存在。
- 已新增 baseline A 的两个 eval config 模板：value/predictor 与 reconstruction strips。

## 文件修改

- `src/nimloth/representation_ablation/README.md`
- `src/nimloth/representation_ablation/__init__.py`
- `src/nimloth/representation_ablation/config.py`
- `src/nimloth/representation_ablation/modules.py`
- `src/nimloth/representation_ablation/metrics.py`
- `src/nimloth/representation_ablation/eval.py`
- `src/nimloth/eval/representation_ablation.py`
- `configs/eval/representation_ablation/a_qwen_latent_value_predictor.yaml`
- `configs/eval/representation_ablation/a_qwen_latent_reconstruction.yaml`
- `tests/representation_ablation/test_config.py`
- `tests/representation_ablation/test_metrics.py`
- 本进度文件。

## 验证命令和结果

- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，3 passed。
- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python - <<'PY' ... load_ablation_config(...)`：两个新增 YAML 模板均可解析。
- `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py`：通过。
- `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/eval/representation_ablation.py`：通过。
- 修正 eval 汇总后再次运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/eval.py`：通过。
- 修正 eval 汇总后再次运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，3 passed。
- 新增 `init.sft2_checkpoint` 后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，4 passed。
- 新增 `init.sft2_checkpoint` 后两个 eval YAML 模板仍可解析。
- 修复 value head checkpoint 缺失风险后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 修复 value head checkpoint 缺失风险后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/config.py src/nimloth/representation_ablation/modules.py tests/representation_ablation/test_config.py`：通过。
- 修复 server smoke 问题后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 修复 server smoke 问题后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py src/nimloth/eval/representation_ablation.py`：通过。
- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation -q`：本地失败在 torch import，原因是当前本地环境缺少 `libstdc++.so.6`，不是测试断言失败；需要服务器/可用 torch 环境 smoke。

## 待确认问题

- 暂无。若后续需要启动超过 3 分钟的训练/评估，会按实验规则再次向人类确认。
