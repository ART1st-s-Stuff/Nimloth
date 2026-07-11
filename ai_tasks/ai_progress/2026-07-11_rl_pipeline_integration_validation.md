# 2026-07-11 RL 管线修复合并与验证

## 任务目标

- 将 `d6e1c1f`、`e05bb53` 中已经完成的 RL/FSDP 安全修复合并到 `feat/rl`。
- 保留当前工作区已有的 encoding `max_length=999999` 未提交修改。
- 先完成本地静态检查和单元测试；在满足实验规则后，再进行真实 GPU/环境管线 smoke 验证。

## 当前计划

1. 保存并恢复当前未提交修改。
2. Cherry-pick 两个 RL 安全修复提交，解决仅与旧 merge 记录有关的冲突。
3. 审查合并结果，修复发现的直接阻塞问题。
4. 运行 Python/shell 静态检查与 RL 单元测试。
5. 明确真实 GPU smoke 的入口、数据 split、checkpoint、训练模块、输出和资源后执行或请求确认。
6. 更新进度文件并提交阶段性成果。

## 已完成步骤

- 已确认当前位于规范 worktree `/workspace/remote2/nimloth-feat-rl`，分支为 `feat/rl`。
- 已确认开始时 `feat/rl` 与 `origin/feat/rl` 的 HEAD 均为 `5ac33bc`。
- 已保存并恢复工作区原有 encoding `max_length=999999` 修改，没有覆盖该修改。
- 已审查目标提交并完成 cherry-pick：
  - `afff665`：对应原 `d6e1c1f`，提供 FSDP/JSONL/advantage 安全修复和测试。
  - `212a6a7`：对应原 `e05bb53`，补充本地 WM 模块同步与 JSONL CLI 安全检查。
- 两次 cherry-pick 都只在当前分支不存在的旧 `ai_tasks/merge_dev.md` 上产生 modify/delete 冲突；已保持该旧 merge 记录不进入本分支，其余 RL 代码按原提交合入。
- 本地静态检查通过；本地 Python 环境没有 torch/pytest/transformers/yaml，无法执行 import/pytest runtime 验证。

## 文件修改

- Cherry-pick 修改：`src/nimloth/training/rl/`、`experiments/training/rl/README.md`、`tests/training/rl/`。
- `src/nimloth/training/rl/trainer.py`：保留任务开始前已有的 encoding `max_length=999999` 修改。
- `ai_tasks/ai_progress/2026-07-11_rl_pipeline_integration_validation.md`：本任务实时进度。

## 验证命令和结果

- `python -m py_compile src/nimloth/training/rl/*.py tests/training/rl/*.py experiments/training/rl/*.py`：通过。
- `bash -n experiments/training/rl/*.slurm experiments/training/rl/*.sh`：通过。
- `PYTHONPATH=src python -m pytest -q tests/training/rl/test_rollout_jsonl.py`：未执行，本地环境缺少 pytest 和 torch。
- `PYTHONPATH=src python -m nimloth.training.rl.cli --help`：未执行到 CLI，本地环境缺少 torch（package import 时由 checkpoint 模块导入）。

## 待确认问题

- 真实 GPU smoke 预计需要加载 Qwen/SFT2 checkpoint 并连接 AI2-THOR env，预计超过 3 分钟；提交前需要按实验规则向人类说明资源、数据、checkpoint、输出和 resume 策略。
