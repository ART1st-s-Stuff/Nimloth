# AI_branch_progress.md — Nimloth 当前进展

本文件记录当前阶段的计划、进展、重要决策和失效记忆。每个 AI 会话开始后应阅读本文件。

---

## 当前阶段：项目初始化 / memory skill

日期：2026-06-10

### 已确认

- 项目名称：Nimloth
- 项目目标：World Model Agent
- 技术栈：Python 机器学习
- 目录位置：当前目录 `/workspace/remote/superpod-csejzhang/nimloth`
- 参考项目：`../flower`
- 当前重点：建立 AI 友好的轻量 memory/task 管理方式。
- Memory 设计原则：短小、可搜索、由人类审批、以文件段 evidence 为依据，不写长篇总结。

### Memory skill/CLI 已创建

- `.agents/skills/memory/SKILL.md`：memory skill 操作协议，已添加 Agent Skills frontmatter。
- `.agents/skills/memory/bin/memory.py`：无第三方依赖 Python CLI。
- `.agents/skills` 是 canonical skill 目录；`.skills` 已废弃并移除。
- `.claude/skills` 是指向 `../.agents/skills` 的兼容 symlink；`.codex`、`.cursor`、`.opencode`、`.pi` 项目 skill 目录已移除，因为这些工具可使用 `.agents/skills`。
- `./skill`：仓库根目录唯一 skill wrapper，支持：
  - `./skill memory add <title> <content>`
  - `./skill memory set <id> <field=value> ...`
  - `./skill memory search <keyword-regex>`
  - `./skill memory get <id>`
  - `./skill memory upvote <id>`
  - `./skill memory human-verify <id>`
  - `./skill human memory-approve`（人类专用）
- 已移除根目录 `./memory` 和 `./verify-ai-memory`，避免根目录随 skill 增多而混乱。
- `.memory/memories.jsonl`：CLI 管理的结构化记忆存储，AI 不应手动编辑。

### Memory 规则要点

- AI 创建的记忆默认 level 为 `pending-human-verification`。
- AI 不得声称 pending memory 是人类已确认记忆。
- 人类审批界面中输入非 `a/r/s/q` 的文本会作为 `human_suggestions` 附加到 pending memory；AI 必须按 suggestion 修改后再请求审批；approve 后 suggestions 自动删除。
- evidence 必须是 JSON list，元素格式为 `{ "filename": str, "line_start": int, "total_lines": int }`。
- tags 必须是 JSON string list。
- 使用定义为：Agent 先验证 evidence，验证后发现该记忆对当前任务有用，才运行 `./skill memory upvote <id>`。
- lazy archive：verified memory 若 7 天没有 triggered verification，或 14 天没有 upvote/use，会自动进入 `archived`。

### 当前 memory 状态

- 已创建并审批 verified memory `M0001`，记录 memory skill/CLI 的存在。

### 已纠正

- 人类指出当前无需创建代码结构；已移除先前创建的代码/实验空目录。

### 待人类确认

1. 是否继续创建对应的 `task` skill/CLI？
2. 是否保留旧 `AI_branch_progress.md` / `AI_issues.md` / `ai_tasks/` 作为过渡，还是逐步迁移到 skill/CLI？

---

## 失效/注意

- 旧项目 flower 中的实现只能作为参考；不能默认视为 Nimloth 的目标实现。
- 当前阶段不要擅自创建业务代码结构、训练脚本或实验目录。

---

## 2026-06-13：latent state/action prior 提取工具

### 已完成

- 根据人类当前 prompt 和 `ai_tasks/latent_action_extraction.md`，进入代码阶段，实现每一步 latent state/action prior 的基础提取工具。
- 新增 `src/nimloth/latent/extraction.py`：
  - 管理 Nimloth special tokens：`<|latent_state|>`、`<|action_start|>`、`<|action_end|>`、8 个 navigation action tokens。
  - 定位单步序列中的 `<|latent_state|>`、`<|action_start|>`、首个 action token。
  - 从 HF-style model output 中提取 final hidden state。
  - 从 `<|latent_state|>` 位置提取 latent state。
  - 用 causal LM 的 `<|action_start|>` 位置 logits 计算 action token 子集上的 logits/log_probs/probs，用于预测后一个位置的首个 action token。
  - 提供 `LatentActionExtractor` 包装类，便于对 Qwen/transformer 模型逐步调用。
- 新增 `src/nimloth/latent/README.md` 和 `tests/test_latent_extraction.py`。
- 未启动训练、评估、rollout、数据采集或 Slurm 任务。

### 验证

- 本地 `python -m py_compile src/nimloth/__init__.py src/nimloth/latent/__init__.py src/nimloth/latent/extraction.py tests/test_latent_extraction.py` 通过。
- 服务器 `.venv` 中 `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_latent_extraction.py` 通过：`5 passed in 3.43s`。
- 服务器 `.venv` 中 `PYTHONPATH=external/VAGEN/verl .venv/bin/python -m pytest -q external/VAGEN/verl/tests/workers/rollout/test_latent_action.py` 通过：`1 passed in 13.61s`。
- 服务器 `.venv` 中 fake causal LM 端到端 smoke test 通过，确认 `LatentActionExtractor.extract_from_model` 可正确提取 latent state 与 action token logits。
- VS Code diagnostics 对新增 Python 文件无报错。
- 备注：服务器默认 `/usr/bin/python` 环境不可用于本任务；验证使用 `/project/peilab/atst/nimloth/.venv/bin/python`。

### 2026-06-13 纠错

- 人类指出先前方案误解需求：目标是在所有后端支持 `<|latent_state|>` 的 attention embedding 提取，以及 `<|action_start|>` 下一个位置的 action logits。
- 已确认普通 PPO 默认使用 FSDP：`ppo_trainer.yaml` 默认加载 `dp_actor`，其 `strategy: fsdp`；Megatron 仅在 `ppo_megatron_trainer.yaml` 或显式 strategy override 下启用。
- 人类确认 Megatron 可先不修改，本轮不继续触碰 Megatron/mcore forward。
- 已纠正 Nimloth 独立工具语义，不再把 action token 位置 hidden state 称为 action prior。
- 已新增 VAGEN/verl 侧统一提取工具，并通过 `actor_rollout_ref.rollout.extract_latent_action` 配置开关启用。
- 已在 FSDP actor-rollout worker 和 PPO trainer 生成后兜底路径接入提取。该路径覆盖 FSDP actor worker 下 hf/vllm/sglang sync/async rollout 的生成结果。
- 当前未完成且暂不处理：Megatron actor 后端的 `<|latent_state|>` attention embedding 提取。现有 mcore non-fused forward 只将 logits/log_probs 暴露给 post-process，hidden embedding 没有通过 `MegatronPPOActor.forward_backward_batch` 返回；不能声称 Megatron 已支持。

## 失效/注意（追加）

- 虽然旧进展曾说“当前阶段不要擅自创建业务代码结构”，本次是人类当前 prompt 明确要求按 VS Code 中任务执行实现，因此创建了最小 `src/nimloth/latent/` 主路径。
- 当前仓库初始化时尚未检测到 git 仓库。

---

## 2026-06-13：memory/event 规则接线

### 已完成

- `AGENTS.md` 明确：任务过程中可以随时通过 memory SKILL 使用和更新记忆，具体协议见 `.agents/skills/memory/SKILL.md`，不得手动编辑 `.memory/memories.jsonl`。
- `ai_rules/02_memory_and_progress.md` 将长期记忆入口从旧 `ai_notes/` 指向 memory SKILL，并加入事件规则索引。
- 新增/填充事件规则：
  - `ai_rules/events/on_progress.md`：取得阶段性进展时，添加新的 durable memory，并评估本任务中使用过的 memory。
  - `ai_rules/events/on_experiment_start.md`：实验开始前，查询 memory、核验证据，并阅读执行 `ai_rules/03_experiments_and_data.md`。
  - `ai_rules/events/on_experiment_end.md`：实验结束/失败/暂停后，更新实验说明文档、结果分析、resume 信息和相关进度。
- `ai_rules/README.md` 已要求触发事件时阅读对应 `events/` 规则，并把规则优先级中的长期记忆入口改为 memory skill。

### 待人类确认

- 无。先前用于规则索引的 pending memory 已不存在；此类信息后续应以规则文档为准，不重复写入 memory。

---

## 2026-06-13：memory 使用规范收紧

### 已完成

- `.agents/skills/memory/SKILL.md` 明确 memory 是从项目实际工作中提取的短小有效经验，不是规则、进度、实验说明或源码文档的重复副本。
- `ai_rules/02_memory_and_progress.md` 增加 memory 使用规范：进度文件记录过程和状态，memory 只记录未来可复用的一句话经验；若信息已清楚存在于规则、实验 README、代码注释或进度文件中，不创建冗余 memory。
- `ai_rules/events/on_progress.md` 同步收窄：只有产出可复用、短小、非重复的项目经验时才添加 memory。
- 本次没有新增 memory；规范本身已由规则文档承载，重复创建 memory 不符合新规范。
