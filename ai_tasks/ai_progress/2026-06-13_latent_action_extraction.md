# latent action extraction 进度

## 任务目标

- 根据 `DESIGN_DOCS.md`，实现每一步从 Qwen/transformer 输出中提取 latent state 和 action prior。
- latent state：取 CoT `</think>` 后的 `<|latent_state|>` token 位置的最后层 hidden state。
- action prior：取 `<|action_start|>` 位置的 causal LM logits，用于预测后一个位置的第一个 action token，并限制到显式 action token 集合。

## 当前计划

1. 检查现有 Nimloth/参考 elen/VAGEN 代码，确认 token 格式和是否已有可复用实现。
2. 新增独立、可测试的提取模块，避免耦合训练/rollout 框架。
3. 添加 README 和单元测试说明语义边界。
4. 运行轻量测试和 IDE diagnostics。

## 已完成步骤

- 已阅读 `AGENTS.md`、`ai_rules/`、`AI_branch_progress.md`、`DESIGN_DOCS.md`、当前任务文件。
- 已确认当前仓库已有外部 VAGEN/elen 参考，但 Nimloth 主路径尚无 `src/` 代码。
- 已新增 Nimloth 独立 latent extraction 模块，未修改外部 VAGEN/elen 代码。
- 已按当前语义区分：latent state 取 `<|latent_state|>` 位置；action prior logits/probs 用 causal LM 的 `<|action_start|>` 位置 logits 预测首个 action token。不再把 action token 位置 hidden state 命名为 prior。

## 文件修改

- `src/nimloth/__init__.py`
- `src/nimloth/latent/__init__.py`
- `src/nimloth/latent/extraction.py`
- `src/nimloth/latent/README.md`
- `tests/test_latent_extraction.py`
- `ai_tasks/ai_progress/2026-06-13_latent_action_extraction.md`

## 验证

- 已通过：本地 `python -m py_compile src/nimloth/__init__.py src/nimloth/latent/__init__.py src/nimloth/latent/extraction.py tests/test_latent_extraction.py`
- 已通过：服务器 `/project/peilab/atst/nimloth` 中 `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_latent_extraction.py`，结果 `5 passed in 3.43s`。
- 已通过：服务器 `.venv` 中 `PYTHONPATH=external/VAGEN/verl .venv/bin/python -m pytest -q external/VAGEN/verl/tests/workers/rollout/test_latent_action.py`，结果 `1 passed in 13.61s`。
- 已通过：服务器 `.venv` 中 py_compile 检查：
  - `external/VAGEN/verl/verl/workers/rollout/latent_action.py`
  - `external/VAGEN/verl/verl/workers/fsdp_workers.py`
  - `external/VAGEN/verl/verl/trainer/ppo/ray_trainer.py`
  - `external/VAGEN/verl/verl/workers/config/rollout.py`
  - `external/VAGEN/verl/tests/workers/rollout/test_latent_action.py`
  - `src/nimloth/latent/extraction.py`
  - `tests/test_latent_extraction.py`
- 已通过：服务器 fake causal LM 端到端 smoke test，`LatentActionExtractor.extract_from_model` 成功提取 latent state 与 action token logits。
- 已通过：VS Code diagnostics 检查 `src/nimloth/latent/extraction.py`、`src/nimloth/latent/__init__.py`、`tests/test_latent_extraction.py`，无诊断问题。
- 备注：服务器默认 `/usr/bin/python` 环境不可用于本任务；验证均使用仓库 `.venv/bin/python`。

## 2026-06-13 纠错记录

- 人类指出前一版理解基本错误：需求是在所有后端支持 `<|latent_state|>` 的 attention embedding 提取，以及 `<|action_start|>` 下一个位置的 action logits。
- 已纠正 Nimloth 独立工具的命名/语义：action logits 从 `<|action_start|>` 位置 logits 读取，用于预测下一个位置的 action token；不再把 action token 位置 hidden state 命名为 prior。
- 已新增 VAGEN/verl 侧统一提取工具 `external/VAGEN/verl/verl/workers/rollout/latent_action.py`。
- 已新增配置开关 `actor_rollout_ref.rollout.extract_latent_action`，默认 `False`。
- 已在 FSDP actor-rollout worker 中接入生成后提取，并在 PPO trainer 中添加兜底，使 sync/async rollout 生成路径（包括 hf/vllm/sglang 输出）在配置打开时统一调用 actor forward 提取。
- 当前未完成：Megatron actor 后端的 `<|latent_state|>` attention embedding 提取。原因是现有 mcore non-fused forward 只将 logits/log_probs 暴露给 post-process，hidden embedding 没有通过当前 `MegatronPPOActor.forward_backward_batch` 返回；需要进一步修改 mcore forward/post-process 接口后才能真实支持。

## 待确认问题

- 人类已确认：默认路径是 FSDP，Megatron 可先不修改。本轮不继续改 Megatron/mcore forward。
