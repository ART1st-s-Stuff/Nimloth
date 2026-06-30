# AI_branch_progress.md — Nimloth 当前进展

本文件记录当前阶段的计划、进展、重要决策和失效记忆。每个 AI 会话开始后应阅读本文件。

---

## 2026-06-30：fix/fsdp — RL FSDP safety refactor（方案 A 实现完成）

### 已完成

- `src/nimloth/training/rl/trainer.py`:
  - 分布式 guard：`world > 1` + `EnvRolloutCollector` → `RuntimeError`，清晰报错
  - 确定性 batch：per-iteration generator (`seed+iteration`) 代替全局 RNG
  - PPO advantage: `std(unbiased=False)` 避免 batch size=1 NaN

- `src/nimloth/training/rl/rollout.py`:
  - `JSONLRolloutCollector` 重写：支持 `sources: list[Path]`（文件/目录）
  - 首次加载 shuffle，轮转循环，所有 rank 确定性相同结果
  - 空源/无效路径报错

- `src/nimloth/training/rl/cli.py`:
  - 新增 `--jsonl-sources` (nargs="+")
  - 非 `--env-url` 且非 `--vagen-config` 时默认走 JSONL

- `src/nimloth/training/rl/loss.py`:
  - `compute_advantages()`: `std(unbiased=False)`

- `tests/training/rl/test_rollout_jsonl.py`: 新增 10 项测试
  - JSONL 加载、轮转、目录展开、多文件、确定性、空源报错
  - advantage 单样本 NaN、多样本 normalization

- 文档更新: `ai_tasks/merge_dev.md`、`experiments/training/rl/README.md`、`src/nimloth/training/rl/README.md`

- 提交: `d6e1c1f` on `fix/fsdp`

### 验证

- `py_compile` 全部通过 (src/nimloth/training/rl/*.py, experiments/training/rl/*.py, tests/training/rl/*.py)
- `bash -n` 全部通过
- 本地无 torch/pytest 环境，无法运行测试

### 风险

- `JSONLRolloutCollector` shuffle 使用固定 seed(42)，不同 run 数据顺序相同（可改）
- `world > 1` 时 small modules (state_proj, wm_predictor, value_head) 仍不跨 rank all-reduce — 梯度可能分叉。短期方案：这些模块参与 FSDP optimizer 但梯度在每 rank 上独立更新。对训练稳定性影响未知，需要真实多 GPU 训练验证。
- 未实现 vLLM rollout backend（按计划允许）

---

## 2026-06-21：Qwen2.5-VL packed-forward monkey patch probe

### 已完成

- 新增实验性 monkey patch：`src/nimloth/training/sft2/qwen_monkey_patch.py`
  - 目标是 HF `Qwen2_5_VLTextModel.forward` 的 multimodal/no-cache/3D-position decoder 路径。
  - 当前 patch 仅在 `sdpa`/`eager` 且无 sliding-attention 层时生效。
  - patch 做法：绕过 HF mask-builder 的 skip path，直接向 decoder layers 传显式 4D 下三角 causal mask。
- `experiments/training/sft2/validate_trajectory_once_2step.py` 新增参数：
  - `--qwen-monkey-patch {none,force_explicit_causal_mask}`
  - 保留 `--attn-implementation` 方便与 `sdpa`/`flash_attention_2` 区分。
- `experiments/training/sft2/validate_2step.slurm` 新增 `QWEN_MONKEY_PATCH` 透传与日志。
- 本地 `py_compile` 通过，并已同步到服务器 `/project/peilab/atst/nimloth`。
- 服务器验证：
  - 首次 job `457378` 失败，暴露 patch 与当前 HF `capture_outputs`/layer return 形式的兼容性问题：patched loop 假设 decoder layer 返回 tensor，实际需要兼容 tuple。
  - 修正后重新提交 job `457380`（`sdpa` + `QWEN_MONKEY_PATCH=force_explicit_causal_mask`）并完成。
- `457380` 结果：
  - text `0/0`
  - synthetic image step0 `0.4375`, step1 `0`
  - real record step0 `10.0`, step1 `7.375`
- 结论：**强制显式 4D causal mask 也不能恢复** Qwen2.5-VL packed-forward image prefix 的等价性；与未 patch `sdpa` 相比只出现很小波动，不能视为修复。

### 进一步定位（decoder probe）

- 新增诊断脚本：`experiments/training/sft2/probe_qwen_decoder_prefix_invariance.py` 与对应 Slurm。
- 服务器 job `457410`（`sdpa`）比较了两条路径的逐层 prefix/full hidden diff：
  1. 正常 wrapper：`Qwen2_5_VLForConditionalGeneration.forward`
  2. decoder-only：手工准备 decoder 输入后直接调用 `model.model.language_model`
- 结果：synthetic image 与真实 record 上，两条路径的 **36 层逐层 diff 完全一致**，且都从 **第 0 层（第一层 decoder layer 输出）** 就开始出现非零差异。
- 但后续更细 probe（job `457430`）显示：
  - `position_ids_prefix_max_diff = 0.0`
  - 第一层 `cos/sin` prefix diff 也为 `0.0`
  - **可是在进入第一层 attention 之前，手工准备的 `inputs_embeds` prefix 已经非零不同**：
    - synthetic image: `0.25`
    - real record: `0.39453125`
  - 第一层 `attn_input` / `q_proj` / `k_proj` / `v_proj` 的非零 diff 与此一致地继续传播。
- 这把当前怀疑点进一步收窄为：
  - 不是 `position_ids` / rotary tables；
  - 也还不能把锅直接甩给第一层 attention backend；
  - **更像是 multimodal `inputs_embeds` 组装/替换本身在 prefix vs full 下已经不一致**（即 image placeholder → image embeds 注入后的结果已经变了）。
- 再进一步的 scatter probe（job `457475`）表明：
  - `text_embeds` prefix 完全一致（`max_diff = 0.0`）
  - `image_mask` prefix 也完全一致，替换位置没有漂移
  - 但 `image_embeds` prefix 本身非零不同：
    - synthetic image: `0.25`
    - real record: `0.39453125`
  - `scattered_embeds` 的 diff 只出现在 image placeholder 覆盖的那些 token 位置上；text token 位置不变
- 这意味着当前最强证据指向：
  - **真正先发生不一致的是 image embeds 本身，而不是 text embeds、placeholder mask、position ids 或第一层 rotary tables。**
  - 也就是说，prefix/full mismatch 的最早可观测源头已被收窄到 `get_image_features()` 输出或其 flatten/concat 组织方式。
- 再进一步的 image-feature structure probe（job `457479`）给出更明确证据：
  - 在服务器当前 HF 环境里，`model.model.get_image_features(...)` 返回的不是带 `.pooler_output` 的对象，而是一个 tuple：prefix case 长度 1，full case 长度 2。
  - 将 full case 返回的前一张图特征与 prefix case 的单图特征直接比较，仍然非零不同：
    - synthetic image: `0.25`
    - real record: `0.39453125`
- 因此，当前最早已确认的分叉点就是：
  - **同一张前缀图片在“单图调用 get_image_features”与“和后续图片一起 batched 调用 get_image_features”时，输出特征本身就不同。**
- 继续深入到 vision tower 内部层（job `457482`）后，结论进一步收紧：
  - `patch_embed` 前缀完全一致（synthetic/real 都是 `0.0`）
  - 但 **第 0 个 vision block 的输出就已经开始非零分叉**：
    - synthetic: block0 diff `0.0078125`
    - real: block0 diff `0.125`
  - 后续 block diff 持续放大，最终传到 merger / pooler 输出。
- 这说明：
  - 问题不在 patchify / patch embedding；
  - **问题最早进入点在 vision transformer 的第一个 block 内部**（attention / norm / MLP / window/full routing 之一），而不是更后面的 merger 或 text decoder。
- 这也意味着我需要修正更早的一个判断：在当前服务器/HF 路径上，不能再说“vision feature extraction 已被排除”；相反，最新证据显示它正是目前最早能观测到的不等价来源，而且已经缩到 **vision block 0**。
- 候选 per-image vision cache patch 验证：
  - 新增 `experiments/training/sft2/validate_per_image_vision_cache.py` / `.slurm`，验证“每张图独立提 vision features，再 scatter 到 full trajectory 后只跑一次 text decoder”。
  - job `457504` (`sdpa`) 与 `457506` (`flash_attention_2`) 均显示：per-image vision cache 能把 `inputs_embeds` 和 `position_ids` 的 prefix diff 打到 `0.0`，但 image case 的 hidden/latent diff 仍不为 0。
  - synthetic image：`sdpa` latent diff step0 `1.375`；FA2 latent diff step0 `0.625`。
  - real record 的 latent index 诊断还暴露当前 real-case index 对齐需进一步核查，但 synthetic case 已足够说明：仅修 vision features 不足以恢复 full/prefix 等价。
- 因此最新状态：
  - full packed 失败至少包含两个层面：1) batched vision feature 非不变；2) 即使将 vision/input_embeds/position_ids 对齐，image-style multimodal decoder full forward 仍非 prefix-invariant。
  - 目前还**不能**确信可以写出一个“单次 full forward + fast attention”的正确 patch。

### 风险 / 当前判断

- 该 patch 只是定向试探，结果为否；还不能据此声称根因已精确定位到某一行 mask-builder 代码。
- 当前没有把该 patch 接入默认 trainer 主路径；也不应默认启用。

## 当前阶段：项目初始化 / memory skill

日期：2026-06-10

### 已确认

- 项目名称：Nimloth
- 项目目标：World Model Agent
- 技术栈：Python 机器学习
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

---

## 2026-06-13：远程网络异常处理经验

- 人类确认：如果多次 SSH 重试失败，应判断可能存在网络问题，停止继续重试并让人类处理。

## 2026-06-13：SFT1 VAGEN baseline rollout 采集启动

### 已完成

- 根据 `ai_tasks/sft1_exp.md` 准备第一阶段 SFT rollout 数据采集。
- 已核验 checkpoint 来源：`experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2`。
- 已核验该 run 的 `validation/43.jsonl` 到 `validation/50.jsonl` 均为 `120/120` 成功；选择最新最高成功率 checkpoint `global_step_50`。
- checkpoint 路径：`.../checkpoints/global_step_50/actor/huggingface`；脚本会拒绝使用非 step 50 的 latest checkpoint。
- 已核验 split 语义：train 使用 `base_train/common_sense_train`，baseline val 使用 `base/common_sense`，test-like heldout 使用 `complex_instruction/visual_appearance/long_horizon`。
- 已确认 `dgx-09` 作业 `451917` 只作为外部 AI2-THOR env server，env URLs 为 `http://10.23.0.77:8400` 与 `http://10.23.0.77:8401`；模型 rollout 使用独立 Slurm allocation，避免抢占 env GPU。
- 新增 rollout-only Slurm 脚本：`experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`。
- 新增实验说明：`experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test_README.md`。
- 已提交 Slurm job `451995`：`sft1-rollouts-vagen50`，preempt 分区，1 node x 8 GPU，当前 pending reason 为 `Priority`。

### 采集设计

- 使用 VAGEN trainer 的 `trainer.val_only=True` 和 `trainer.val_before_train=True` 作为 rollout-only 入口；只生成 validation trajectories，不做 actor/critic update。
- 输出目录：`experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test/validation/{train,val,test}`。
- 计划数量：train 4800、val 600、test 540，总计 5940。
- 图片保存已启用：`trainer.log_image.enable=True`，图片位于 `validation/<split>/image_50/images_<sample_idx>/<turn_idx>.png`。
- Resume 方式：Slurm `--requeue`；每个 split 如果已有非空 `50.jsonl` 则跳过，不截断已有输出。

### 注意

- `ai_tasks/sft1_exp.md` 未明确定义 test split；本次将没有 `_train` 对应的三个 navigation heldout categories 记录为 test-like split，并已写入 README。
- VAGEN JSONL 保存 decoded multi-turn 文本和 image placeholders，截图单独按 sample/turn index 存盘；后续 SFT 转换若需要严格 `{role, content, screenshot_path}` 结构，需要做一次转换/重组。

### 2026-06-13 纠错：VAGEN navigation 无正式 train/val/test 三分法

- 人类询问 VAGEN 是否定义 train-test-val 后，重新核查 `external/VAGEN/vagen/envs/navigation` 与官方 examples。
- 结论：VAGEN navigation 明确区分的是 train scenes 与 eval scenes；examples 使用 `DATASET_TRAIN`/`DATASET_VAL`，但代码/assets 没有正式独立的 `test` split。
- 训练集证据：`base_train/common_sense_train/long_horizon_train` 来自 train scenes。
- eval/validation 证据：`base/common_sense/long_horizon` 等 60-task eval sets；官方 `examples/evaluate/navigation/config.yaml` 将 `base/common_sense/long_horizon` 作为 evaluation configs。
- 因此，先前将 `complex_instruction/visual_appearance/long_horizon` 记为 `test-like` 是假设，不是 VAGEN 定义。已取消 Slurm job `451995`，避免继续消耗资源跑 split 语义不稳的采集。
- `experiments/navigation_baseline/runs/sft1_rollouts_vagen50_train_val_test_README.md` 已标注 paused/cancelled 与该纠错。

### 2026-06-13 SFT1 rollout split 方案按 VAGEN train/test 边界修正并重启

- 人类确认：先按照 VAGEN 的方式划分 train/test，然后再在 train 里划分 val。
- 已修正 `experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`：
  - `train`: VAGEN train-scene assets `base_train/common_sense_train/long_horizon_train`，每类 seeds `1..1080`，`val_kwargs.n=1`，共 3240 rollouts。
  - `val`: 从相同 VAGEN train-scene assets 留出 seeds `1081..1200`，`val_kwargs.n=2`，共 720 rollouts。
  - `test`: VAGEN eval-scene assets `base/common_sense/complex_instruction/visual_appearance/long_horizon`，每类 seeds `1..60`，`val_kwargs.n=7`，共 2100 rollouts。
  - 总计划 6060 rollouts。
- VAGEN seed 规则已核查：`[min, max, 1]` 是 inclusive range，且每个 seed 最多出现一次；因此 train/val seed 范围在每个 train asset 内不重叠。
- 新实验名/输出目录：`sft1_rollouts_vagen50_vagen_train_val_test`。
- 仍为 rollout-only：`trainer.val_only=True`，actor/critic 不训练，初始化 checkpoint 仍为 `global_step_50`。
- env server guard 通过：`dgx-09` job `451917` running，URL 文件 ready，checkpoint latest=50。
- 已提交新的 Slurm job `451998`。

### 2026-06-13 SFT1 rollout checkpoint 加载方式修正

- job `451998` 在 checkpoint load 阶段失败，未产生 rollout 输出。
- 失败原因：脚本用单节点 8GPU/world_size=8 启动，但原训练 checkpoint 的 FSDP shards 是 `model_world_size_16_rank_*.pt`，加载器寻找 `model_world_size_8_rank_*.pt` 而失败。
- 已确认 `global_step_50/actor/huggingface` 中存在完整 HF export（4 个 safetensors shard + config/tokenizer）。
- 已修正 `experiments/navigation_baseline/sft1_rollouts_vagen50_valonly.slurm`：
  - `actor_rollout_ref.model.path` 改为 `global_step_50/actor/huggingface`。
  - `trainer.resume_mode=disable`，避免加载 world_size=16 的 FSDP shards。
  - `trainer.default_local_dir` 指向新 run 下的空 `no_resume_checkpoint_dir`。
  - rollout 仍是 `trainer.val_only=True`，actor/critic 不训练。
  - 因禁用 resume，validation dumps 预期写为 `0.jsonl` 和 `image_0/`；模型来源仍记录为 `global_step_50`。
- README `sft1_rollouts_vagen50_vagen_train_val_test_README.md` 已同步该修正。

### 2026-06-13 按人类要求转换 best checkpoint world size

- 人类纠正：需要把 best checkpoint 转换为目标 world size，而不是绕开 FSDP resume 直接 HF 冷启动 rollout。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size8.py`：初始化 8-GPU VAGEN/verl actor-rollout workers from `global_step_50/actor/huggingface`，然后调用原生 `_save_checkpoint()`，只做 checkpoint conversion，不 rollout、不训练。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size8.slurm`：单节点 8GPU，输出目标 `experiments/navigation_baseline/runs/vagen50_world_size8_from_hf/checkpoints/global_step_50/actor/model_world_size_8_rank_*.pt`。
- 已恢复 rollout 脚本为 resume 方式：`trainer.default_local_dir` 指向转换后的 checkpoint dir，`trainer.resume_mode=auto`，预期输出仍为 `50.jsonl`/`image_50/`。
- 已提交转换 Slurm job `452016`。转换成功并验证 rank shards 后，再启动 SFT rollout。

### 2026-06-13 Slurm GPU 资源查询脚本

- 人类纠正：以后查询资源时，应告诉每个分区、每个节点具体还剩多少资源，而不只给汇总。
- 已新增 `experiments/navigation_baseline/slurm_gpu_resources.py`，解析 `scontrol show nodes`，输出每个 GPU 节点的 partition/node/state、free/allocated/total GPU、free/allocated/total CPU、free/real memory，并附 partition 汇总。
- 使用示例：`python3 experiments/navigation_baseline/slurm_gpu_resources.py --only-free-gpu`。

备注：上方“HF 冷启动 rollout”方案已被人类纠正，不作为当前方案。当前方案以 `2026-06-13 按人类要求转换 best checkpoint world size` 为准：先产出 world_size=8 FSDP resume checkpoint，再启动 rollout。

### 2026-06-13 改用 dgx-26 碎片资源：world_size=2 + 1env2Qwen rollout

- 人类指示：使用 `dgx-26`，先导出 `world_size=2` checkpoint，然后用 1 卡 env + 2 卡 Qwen 进行并行 rollout。
- 已取消 pending 的 8GPU conversion job `452016`。
- 已新增 `experiments/navigation_baseline/convert_vagen50_to_world_size2_dgx26.slurm`：`normal` 分区，`--nodelist=dgx-26`，`--gres=gpu:2`，从 best checkpoint HF export 保存 `model_world_size_2_rank_*.pt`。
- 已新增 `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm`：`normal/dgx-26/gpu:3`，第 1 张 GPU 启本地 AI2-THOR env server，后 2 张 GPU 用作 Qwen/Ray，resume `vagen50_world_size2_from_hf/checkpoints/global_step_50`。
- 已提交 world_size=2 conversion job `452020`。转换成功验证后启动 rollout。

### 2026-06-13 world_size=2 conversion 成功并启动 dgx-26 rollout

- `convert_vagen50_to_world_size2_dgx26.slurm` 经数次修正后成功完成 job `452050`。
- 验证通过：`vagen50_world_size2_from_hf/checkpoints/latest_checkpointed_iteration.txt=50`，并存在 `global_step_50/actor/model_world_size_2_rank_{0,1}.pt`、`extra_state_world_size_2_rank_{0,1}.pt`、`fsdp_config.json`、`data.pt`。
- 已提交 rollout job `452052`：`normal/dgx-26/gpu:3`，GPU0 local AI2-THOR env，GPU1-2 Qwen/Ray，resume converted world_size=2 checkpoint，输出目录 `runs/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen/validation/{train,val,test}`。

## 2026-06-13 15:36 UTC - SFT1 VAGEN rollout retry with shard resume

- Human corrected rollout robustness requirement: future experiments must be resumable, but progress saving should be coarse enough to avoid wasting too much compute on checkpoint/output overhead.
- Cancelled rollout job `452052` after observing `data.val_batch_size=6` made train split too slow and split-level resume would lose all unfinished train work.
  - `sacct`: `452052 CANCELLED by 3738`, elapsed `01:39:30` on `dgx-26`.
  - No completed `validation/{train,val,test}/50.jsonl` existed, so no partial split output was reused.
- Updated `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm`:
  - Increased `VAL_BATCH_SIZE` from `6` to `24`.
  - Increased `AGENT_NUM_WORKERS` and `AGENT_MAX_CONCURRENT_TRAJECTORIES` from `2` to `8`.
  - Replaced split-level resume with seed-shard resume.
  - New output paths are `validation/{split}/shard_*/50.jsonl`.
  - Shard plan: train seeds `1-1080` as six 180-seed shards (`540` rollouts each), val seeds `1081-1200` as one shard (`720` rollouts), test seeds `1-60` as one shard (`2100` rollouts).
  - Resume skips any shard with an existing non-empty `50.jsonl`; failed/requeued jobs rerun only the currently incomplete shard.
- Validation before submit:
  - `bash -n experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_dgx26_1env2qwen.slurm` passed.
  - `dgx-26` had `0/8` free GPUs after cancellation because other users/jobs occupied it; `normal` only showed `dgx-10` with `1/8` free GPU.
- Submitted updated rollout job `452075`:
  - `normal`, `dgx-26`, `gres:gpu:3`.
  - Current state at submit: `PENDING (Priority)`.
  - Purpose/data/checkpoint semantics unchanged: rollout-only VAGEN baseline from converted `world_size=2` `global_step_50`; 1 GPU local AI2-THOR env, 2 GPUs Qwen/Ray.

## 2026-06-13 16:48 UTC - SFT1 rollout moved to 2-node fragmented GPUs with external env

- Human approved switching from waiting for `dgx-26` 3GPU to using fragmented available nodes.
- Cancelled pending `452075` (`sft1-ws2-dgx26`) before it started.
- Created `experiments/navigation_baseline/sft1_rollouts_vagen50_ws2_2node_externalenv.slurm`:
  - `normal`, `--nodelist=dgx-[10,16,21]`, `--nodes=2`, `--gres=gpu:1`, `--mem=60G` per node.
  - Uses existing external AI2-THOR env URLs from job `451917`: `http://10.23.0.77:8400` and `http://10.23.0.77:8401`.
  - Keeps converted `world_size=2` checkpoint `global_step_50`, rollout-only `trainer.val_only=True`, shard-level resume, and `VAL_BATCH_SIZE=24`/8 agent workers.
  - Performs external env health checks from the allocated head compute node before starting rollout.
- Submitted job `452090`; it started on `dgx-[10,21]`.
- Runtime checks:
  - External env health succeeded from `dgx-10` for both `8400` and `8401`.
  - Ray head/worker logs created for `dgx-10` and `dgx-21`.
  - Job entered `train/shard_001_180` and VAGEN config showed `n_gpus_per_node=1`, `nnodes=2`, `val_only=True`, validation dir `validation/train/shard_001_180`.
  - Model initialization is in progress; next monitor step is to confirm converted FSDP checkpoint load (`model_world_size_2_rank_*.pt`) and first validation generation batches.
- Operational note: SSH control plane again returned `Connection closed by UNKNOWN port 65535`; SSHFS log reads continued working.

## 2026-06-13 17:10 UTC - SFT1 rollout monitoring: first shard completed

- Job `452090` remains running on `dgx-[10,21]` with external env `dgx-09`.
- Confirmed actual checkpoint resume in `sft1_rollouts_vagen50_ws2_2node_externalenv.log`:
  - actor rank 0 loaded `actor/model_world_size_2_rank_0.pt`.
  - actor rank 1 loaded `actor/model_world_size_2_rank_1.pt`.
  - critic rank 0/1 also loaded world_size=2 files.
- First train shard completed:
  - `validation/train/shard_001_180/50.jsonl` exists with `540` lines, as expected.
  - `validation/train/shard_001_180/image_50` exists and contains PNG image dumps.
- Job automatically advanced to second train shard `shard_181_360` and again loaded actor world_size=2 rank0/rank1 checkpoint files.
- Current log counts around this monitor point:
  - `validation generation end`: 24
  - `test_gen_batch meta info`: 25
  - `Traceback`: 2, both from optional nvcc/colorama extension warnings while generation continued.
  - `OOM`: 0
  - `ERROR`: 0

## 2026-06-14 - VAGEN continue training moved to normal 4env/16train

- SFT1 rollout collection completed via resumed job `452120` after `452090` failed from external env timeout:
  - train shards: 6 x 540 = 3240 lines.
  - val shard: 720 lines.
  - test shard: 2100 lines.
  - total rollout JSONL records: 6060.
  - output root: `experiments/navigation_baseline/runs/sft1_rollouts_vagen50_ws2_2node_externalenv/validation`.
- Human requested moving the VAGEN continue-training task from `preempt` to `normal` using `4env 16train`.
- Cancelled old pending preempt continue-training job `451918` (`vagen-resume-16g-extenv-100`).
- Added `experiments/navigation_baseline/env_normal_4gpu_resume_retry2.slurm`:
  - `normal`, `dgx-12`, `gres:gpu:4`, 4 external AI2-THOR env servers on ports `8400..8403`.
  - Control dir: `external_env_normal_4gpu`.
- Added `experiments/navigation_baseline/resume_retry2_train_from50_normal_4env16train_external_env.slurm`:
  - `normal`, `dgx-32,dgx-37`, 2 nodes x 8 GPU = 16 train GPUs.
  - Reads 4-env URL file from `external_env_normal_4gpu/env_urls.txt`.
  - Continues from original run checkpoints with `trainer.resume_mode=auto`, latest checkpoint expected `global_step_50`, total target `trainer.total_training_steps=100`.
  - Trains actor and critic via VAGEN PPO/GAE; external env job only provides AI2-THOR environments.
- Submitted env job `452235`; it is running on `dgx-12` and ready with URLs:
  - `http://10.23.0.101:8400`
  - `http://10.23.0.101:8401`
  - `http://10.23.0.101:8402`
  - `http://10.23.0.101:8403`
- Submitted train job `452236`; it is running on `dgx-[32,37]` and passed health checks for all 4 external env URLs before launching VAGEN training.

## 2026-06-14 - Server resource handling preference

- Human instructed that future server-resource work should first submit a placeholder/hold job to reserve the target resources, then connect to the allocated node(s) for interactive operation, instead of relying on resources remaining available while preparing commands.

## 2026-06-14 - Continue-training retries and current status after resource/debug cycle

- Continued monitoring VAGEN resume training from `global_step_50` on normal `4env 16train`.
- Confirmed external env backend `452235` remains healthy on `dgx-12` with 4 URLs:
  - `http://10.23.0.101:8400`
  - `http://10.23.0.101:8401`
  - `http://10.23.0.101:8402`
  - `http://10.23.0.101:8403`
  - Env health checks from training nodes returned `{"ok":true,...}`; current blockers are training initialization, not env server failure.
- `452263` was cancelled after it remained RUNNING but idle:
  - checkpoint stayed at `50`, no `global_step_51`.
  - stdout/log mtime stopped around actor/worker initialization.
  - 16 train GPUs stayed ~0% util and ~1.7GB memory.
- Patched `experiments/navigation_baseline/resume_retry2_train_from50_normal_4env16train_external_env.slurm` to disable fused kernels after diagnosing Torch/inductor-style initialization stalls:
  - `actor_rollout_ref.model.use_fused_kernels=False`.
- `452269` was cancelled after it still spawned many `torch/_inductor/compile_worker` processes and stalled in `actor_rollout_init_model`.
- Added explicit torch compile disables and retry:
  - `actor_rollout_ref.actor.use_torch_compile=False`
  - `actor_rollout_ref.ref.use_torch_compile=False`
  - `TORCHINDUCTOR_DISABLE=1`
  - `TORCH_COMPILE_DISABLE=1`
- `452285` still spawned compile workers, so it was cancelled.
- Added deeper compile disables and retry:
  - `actor_rollout_ref.actor.fsdp_config.use_torch_compile=False`
  - `actor_rollout_ref.ref.fsdp_config.use_torch_compile=False`
  - `critic.model.fsdp_config.use_torch_compile=False`
  - `TORCHDYNAMO_DISABLE=1`
  - `TORCHINDUCTOR_COMPILE_THREADS=1`
  - startup cleanup now kills stale `torch/_inductor/compile_worker` along with Ray/SGLang leftovers.
- `452287` initially had `compile_worker=0` and nonzero GPU util, but later regressed/stalled; it was cancelled.
- `452295` progressed further than previous retries:
  - compile workers stayed `0` after stronger disables.
  - actor/critic initialization reached `After critic FSDP` and `reference model: Qwen/Qwen2.5-VL-3B-Instruct`.
  - then stalled in `WorkerDict.actor_rollout_init_model` during reference model/FSDP initialization with GPU util back to 0%, checkpoint still `50`, no `global_step_51`.
  - Ray logs showed no explicit hidden fatal errors; `py-spy` was blocked by ptrace permissions; `/proc` showed workers waiting in `ep_poll`.
- Added FSDP initialization workaround for the next retry, relying on checkpoint load to restore actual weights:
  - `actor_rollout_ref.actor.fsdp_config.sync_module_states=False`
  - `actor_rollout_ref.ref.fsdp_config.sync_module_states=False`
  - `critic.model.fsdp_config.sync_module_states=False`
- Submitted `452310`; it later started and is currently RUNNING on `dgx-[32,37]` at the last resource check.
  - Latest known checkpoint remains `50` until post-start monitoring confirms otherwise.
  - Need monitor whether `452310` passes the prior `reference model`/`actor_rollout_init_model` stall and reaches validation/training or `global_step_51`.
- Current resource snapshot from `slurm_gpu_resources.py`:
  - normal total free: `10/176` GPUs.
  - normal free nodes: `dgx-14` 3 free, `dgx-16` 3 free, `dgx-26` 1 free, `dgx-54` 3 free.
  - no additional full 8-GPU normal node was free while `452310` occupied `dgx-32,dgx-37`.
  - preempt total free: `15/208` GPUs; `dgx-31` had 7 free, `dgx-34` showed 8 free but `DOWN+NOT_RESPONDING`.
- Human instructed future server-resource workflow:
  - first submit a placeholder/hold job to reserve target GPUs/nodes,
  - then connect to the allocated node(s) for interactive operation,
  - do not rely on queried free resources remaining available.
- Added pending memory `M0005` for that resource workflow preference; human approval still required via memory skill approval flow.

## 2026-06-14 - normal 2env + 3x4GPU train resume from step 50

- Human requested continuing VAGEN training on `normal` with `2 GPU env + 3 nodes x 4 GPU train` from `global_step_50`.
- Source checkpoint is `world_size=16` at `global_step_50`; new layout requires `world_size=12` conversion before resume.
- Added `experiments/navigation_baseline/convert_vagen_checkpoint_to_world_size.py` (generic HF->FSDP shard converter).
- Added `experiments/navigation_baseline/convert_vagen50_to_world_size12.slurm`: `normal`, 3 nodes x 4 GPU, writes `model_world_size_12_rank_*.pt` into original run `checkpoints/global_step_50`.
- Added `experiments/navigation_baseline/resume_retry2_train_from50_normal_2env12train_external_env.slurm`: `normal`, 3 nodes x 4 GPU, reads 2-env URLs from `external_env_dgx09_2gpu`, `train_batch_size=96`, `ppo_mini_batch_size=24`, `total_training_steps=100`, `resume_mode=auto`.
- Reused existing `env_dgx09_2gpu_resume_retry2.slurm` for 2 external AI2-THOR env servers.
- Cancelled old 4-env job `452235` (`vagen-env-normal-4gpu` on `dgx-12`).
- Submitted:
  - `452345` env: `dgx-09`, `gres:gpu:2`
  - `452346` convert: `dgx-[12,16,54]`, `gres:gpu:4` x 3 nodes
  - `452347` train: pending `(Dependency)` on `afterok:452346`
- Monitoring `452345/452346/452347` retry cycle:
  - Fixed Hydra `+sync_module_states` overrides and convert script cwd/path issues.
  - Fixed convert dummy dataset `n_envs: 12` to satisfy `drop_last=True` with `train_batch_size=12`.
  - Current active jobs after retries:
    - `452345` env: `dgx-09` RUNNING, 2 env URLs ready (`http://10.23.0.77:8400/8401`).
    - `452355` convert: `dgx-[12,16,54]` RUNNING ~19m; passed critic FSDP, reached `Before build_rollout` on all ranks, then log stopped ~19:11 HKT with GPU util ~0%.
    - `452356` train: pending `afterok:452355`.
  - Convert still has `0/12` ws12 actor shards; possible SGLang rollout init stall (same class as prior resume retries).

---

## 2026-06-18：LeWM 清理 + training/experiments 结构优化

### 已完成

- LeWM：`wm/_vendor_lewm.py` 最小 vendoring；移除 `wm/model.py`、pixel JEPA pretrain；`LatentWMPredictor` 在 `wm/predictor.py`。
- WM 模型组件迁入 `wm/`：`state_proj.py`、`value_head.py`、`collate.py`；新增 `wm/README.md`。
- SFT2 训练逻辑下沉 `training/sft2/`：`trainer.py`（主循环）、`step.py`、`checkpoint.py`、`evaluate.py`、`dataset.py`、`qwen_latent.py`。
- 跨 phase 工具：`training/common/dist.py`、`qwen_batch.py`。
- `experiments/training/sft2/train.py` 改为薄入口（调用 `nimloth.training.sft2.trainer`）。
- 文档同步：`ai_tasks/sft2_phase2_plan.md`、`CHANGELOG.md`、`configs/training/README.md`、`experiments/training/README.md`。

### 第二轮拆分（2026-06-18）

- `qwen_tuning` / `vision_ema` → `src/nimloth/backbone/`；新增 `backbone/README.md`。
- 离线指标 `val_rollout_success_rate` → `src/nimloth/eval/rollout.py`；`training/sft2/metrics.py` 仅保留 batch 内指标。
- 测试迁至 `tests/backbone/`、`tests/eval/`。

### 目录约定（SFT2）

- **骨干 / 调参**：`src/nimloth/backbone/`
- **模型 / 数据**：`src/nimloth/wm/`
- **离线评估**：`src/nimloth/eval/`
- **训练编排**：`src/nimloth/training/sft2/`
- **实验入口**：`experiments/training/sft2/`（Slurm/submit 不变）

### 2026-06-18 审阅后修正

- 人类确认 `AGENTS.md` 变更由人类本人修改，无需回退。
- 人类确认项目从未使用内置 LeWM 实现训练；已删除 `LatentWMPredictor.load_checkpoint` 中旧 LeWM `model.pt` warm-start fallback。
- `_vendor_lewm.py` 不再导入/导出 `JEPA`、`SIGReg`，仅保留 SFT2 predictor 需要的 `ARPredictor`、`Embedder`、`MLP`。
- `ai_tasks/sft2_phase2_plan.md` 与 `CHANGELOG.md` 已同步：SFT2 predictor 仅支持自身 `predictor.pt` checkpoint 或随机初始化，不支持旧 JEPA checkpoint warm-start。

### 2026-06-18 baseline 实验目录迁移

- 分支：`refactor/experiments-training-baseline`
- 新增规范入口：`experiments/training/baseline/`（通用 Slurm + submit，无节点/retry 命名）
- 配置：`configs/training/baseline/{train,val,defaults}.yaml`
- 远程已初始化：`outputs/experiments/training/baseline/`（`README.md`、`progress.md`、`slurm/`、`runtime/`）；旧 `outputs/experiments/navigation_baseline/` 保留
- 参考最新有效 VAGEN RL run：legacy `retry2`，`global_step_93`
- `experiments/navigation_baseline/` 标记为冻结遗留，勿新增脚本

### 2026-06-18 SFT1 脚本迁移

- 规范入口：`experiments/training/sft1/` + `configs/training/sft1/`
- 远程已初始化：`outputs/experiments/training/sft1/`（README、progress.md、slurm/）
- legacy runs 暂留 `experiments/navigation_baseline/runs/`（`SFT1_RUNS_ROOT` 可覆盖）
- SFT2 合并脚本路径更新为 `experiments/training/sft1/merge_lora_ckpt.py`
- SFT2 默认 `TRAIN_OUT` 迁至 `outputs/experiments/training/sft2/<date>/<name>/`（`common_env.sh`）


## 2026-06-18：SFT2 DDP resume correction should live in local repo

- Human corrected workflow: local repo is the source of truth; server-side code may be overwritten. The SFT2 DDP/checkpoint resume fixes originally committed on the server must be reflected locally.
- Local repo now carries the relevant code changes in `src/nimloth/training/sft2/trainer.py`: non-reentrant Qwen gradient checkpointing, DDP `find_unused_parameters=False`, and full HF checkpoint resume reloading `best/` before optimizer construction.
- Remote run status at the time of correction: `sft2_latentwm_default_8gpu` resumed from `best/` (`start_epoch=2`, `global_step=855`) and progressed to at least `global_step=876` without the prior DDP ready-twice error.

## 2026-06-19：SFT2 action token mismatch 修正与重启

- 发现 SFT2 使用 `nimloth.latent.add_special_tokens()` 时仍会添加旧 `<|act_moveahead|>...` action tokens；实际 VAGEN/Nimloth SFT 数据和 parser 使用 `<|action_(0)|>...<|action_(7)|>`。
- 已在本地修正并提交：`src/nimloth/latent/extraction.py` 改为 `<|action_(idx)|>`；嵌套 submodule `external/VAGEN/verl/verl/workers/rollout/latent_action.py` 默认 action tokens 同步改为 `<|action_(idx)|>`。
- 本地提交：root `47b3295`；VAGEN submodule `b7420be`；verl nested submodule `d8e52104`。本地 pytest 环境不可用，`python -m py_compile` 通过；远程 `.venv` 验证 `tests/test_latent_extraction.py` 与 `external/VAGEN/verl/tests/workers/rollout/test_latent_action.py` 均通过。
- 远程同步并提交：root `f58d6fcd2114a6c56967c4278d18ed3825d43787`；VAGEN submodule `6cbb529`；verl nested submodule `8bc3f7f0`。
- 已停止污染的远程实验 `outputs/experiments/training/sft2/2026-06-18/sft2_latentwm_default_8gpu`，并在该目录 `README.md` 记录失败原因：旧 token 被加入 tokenizer，checkpoint vocab/metadata 被污染，不应作为最终 SFT2 结果。
- 已用 fresh output 重启 SFT2：`outputs/experiments/training/sft2/2026-06-19/sft2_latentwm_default_8gpu_tokenfix`，复用 hold job `456005`，从干净 SFT1 merged checkpoint 初始化，LLM freeze、vision full+EMA，训练 state_proj / LatentWMPredictor / ValueHead。
- 重启健康检查：新 run `add_special_tokens` 对 SFT1 tokenizer 返回 `added=0` 且无旧 `<|act_*>`；日志未出现 new embeddings/lm_head resize warning；`train_step_log.csv` 已写到至少 `global_step=5`。

## 2026-06-19：SFT2 CE last-span 调整（dev worktree）

- 按人类要求在 `../nimloth-dev` worktree 修改，尚未同步服务器。
- `training/common/qwen_batch.py` 的 SFT2 CE mask 现在只覆盖 prefix 中最后一个 assistant span，避免 transition 展开后重复监督早期 assistant turns。
- SFT2 next-prefix WM target forward 与 validation latent forward 会移除 `labels`，避免不使用 CE 时仍让 Qwen 计算 loss。
- 已新增 `tests/training/common/test_qwen_batch.py` 覆盖 last-span 行为；本地依赖不完整，`python -m py_compile` 通过，`pytest` 因当前 Python 无 pytest、手动导入测试因缺 PIL 未能运行。

## 2026-06-19：SFT2 慢速后续优化（dev worktree）

- 在 `../nimloth-dev` 继续修复慢速诊断剩余项，未启动新的服务器训练。
- `training/common/qwen_batch.py` 增加 per-process chat-template、token offset 与 RGB image decode LRU cache（图片 cache 限制为 8192，避免过高 CPU 内存占用）；同时 last-span 计算只渲染最后 assistant 相关 prefix，减少 transition 展开后的重复模板渲染/图片打开/offset tokenization 开销。
- `training/sft2/qwen_latent.py` 改为通过 Qwen final norm forward hook 捕获 last hidden，不再用 `output_hidden_states=True` 返回所有层 hidden states；新增 `tests/training/sft2/test_qwen_latent.py` 覆盖该行为。
- `training/sft2/trainer.py` 在 gradient accumulation 非同步 micro-step 上对 DDP 模块使用 `no_sync()`，只在 accumulation 边界或 epoch 尾部同步梯度。
- SFT2 配置/CLI 增加性能旋钮：YAML 可设置 `attn_implementation`、`gradient_checkpointing`；CLI 的 `--gradient-checkpointing/--no-gradient-checkpointing` 可切换；默认配置改为 `flash_attention_2` 且保持 gradient checkpointing 开启以降低 OOM 风险。
- 验证：`python -m py_compile` 覆盖相关源码和新增测试通过；当前本地 Python 缺 `pytest`，手动导入测试因缺 `PIL` 未能运行。

## 2026-06-19：SFT2 dev 分支同步服务器并重跑

- 本地 `../nimloth-dev` 已提交并推送 dev 分支：`682448d Optimize SFT2 training throughput`，随后修正 VAGEN submodule 指针到服务器已有/已推送的 tokenfix commit：`80d65a0 Use pushed VAGEN tokenfix submodule commit`。
- 服务器 `/project/peilab/atst/nimloth` 已切到 `dev` 并 reset 到 `origin/dev` commit `80d65a05c36620d3ab9e0eaa6e879a93d20b2d95`；服务器工作区清洁。
- 服务器验证通过：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/training/common/test_qwen_batch.py tests/training/sft2/test_qwen_latent.py` -> `3 passed in 63.94s`。
- 已取消旧慢速 SFT2 hold/train job `456285`，保留 hold job `456454`（`dgx-28`）用于重跑。
- 新 run 输出目录：`outputs/experiments/training/sft2/2026-06-19/sft2_latentwm_default_8gpu_tokenfix_opt`；README 记录 commit、数据、init checkpoint、训练/冻结模块与监控项。
- 第一次 launcher 因 login shell 未 load Slurm module 未启动；第二次 launcher 误设 `EVAL_TAG_PREFIX=alltrain_8gpu_lora_cache_opt`，被及时 kill，未进入训练。第三次使用正确 `EVAL_TAG_PREFIX=alltrain_8gpu_lora_cache` 启动。
- 当前新 run 已健康启动：从 SFT1 `epoch_004/hf_merged` 初始化，`train_step_log.csv` 已写到至少 `global_step=13`；最近 10 step 中位约 `6.36s/step`（旧 run 最近 200 step 中位约 `7.55s/step`），GPU 显存约 47GB/80GB。

## 2026-06-19：SFT2 speedup 续作（batch 默认 + smoke + P4 原型）

- 人类批准增大 batch_size：默认 yaml/CLI 改为 `batch_size=2`, `grad_accum=4`（8 卡 effective batch 仍为 64）。
- P4 原型：`trajectory_forward.py` + `forward_qwen_last_hidden()`；**尚未接入 trainer**。
- 新增 GPU smoke：`experiments/training/sft2/smoke_speedup.py` + `smoke_speedup.slurm`。
- `train_vagen79_default.slurm` 支持 `PREPROCESS_CACHE_DIR`、`STEP_TIMING` 环境变量。
- 服务器 smoke（dgx-28）：P1 cache/micro_loss 通过；P4 trajectory latent 等价性在真实数据上未通过（max_abs_diff≈14），暂不集成 packed forward。

## 2026-06-20：trajectory-once packed forward review/fix

- Review 远程 probe 日志 `outputs/experiments/training/sft2/smoke/once_probe_456667.{log,err}`：3 条真实 train trajectory 的 trajectory-once full forward 均未通过 legacy per-prefix 等价，`latent_max_diff≈15.75-16.25`，`total_diff≈0.09-0.18`。
- 结论：full-trajectory single forward 对当前 Qwen-VL navigation 多图轨迹不是可接受的默认 SFT2 语义等价优化；可能由未来图片/多模态 position/vision 批处理等实现细节导致，不能把该路径产物当作默认 SFT2 结果。
- 安全修复：`trainer.py` 现在在 `--packed-forward` 未同时传 `--allow-approx-trajectory-once` 时直接报错，防止误用非等价路径；Slurm wrapper 仅在 `ALLOW_APPROX_TRAJECTORY_ONCE=1` 时附加 override。
- 文档同步：`experiments/training/sft2/README.md` 记录 once_probe_456667 的失败结论，并说明 packed-forward 仅可用于 research/profiling，生产默认不得启用。
- 本地验证：相关 Python 文件 `py_compile` 通过；本地环境缺 torch，无法运行 pytest/import 级测试。

## 2026-06-20：trajectory-once 编码修复 + 2-step GPU 验收（进行中）

- **根因（debug job 456704）**：debug 合成数据 messages 未交错 assistant（`[u0,u1,a1]` 缺 a0）；真实数据须用 `expand_record_transitions` 结构。VL 多图时 standalone prefix encode 与 full encode 的 token prefix 可能不一致（prefix instability）。`find_step_latent_indices_in_full`（char span）在 Qwen-VL 上不可靠。
- **编码修复（已完成，本地+服务器已同步）**：
  - `encode_full_trajectory` + `verify_prefix_tokenization`（text 级 + token 级 prefix 检查）
  - `find_step_latent_indices()`：`find_latent_index_in_last_assistant_span()`（VL offset 失败时回退 `find_last`，与 legacy 一致）；不再用 `find_all(full)[:N]`
  - `forward_trajectory_once` 改用上述索引；forward 前 `reset_model_rope_state`（`qwen_latent.py`）
  - `trajectory_forward.py` full encode 改为用 `steps[-1]` 的 prefix
  - 修复 `test_trajectory_prefix_encoding.py` 的 `max_length` 变量 bug
- **新增**：`experiments/training/sft2/validate_trajectory_once_2step.py` + `validate_2step.slurm`；synthetic 2-step 必须通过；real record 前 2 step 若 prefix verify 失败则单独报告（不 silent fallback）。
- **服务器 job 456711 失败**：`find_step_latent_indices_in_full` 在 synthetic step0 找不到 latent（verify 已通过）；已改为 `find_step_latent_indices()`。
- **GPU 验收 job 456802**（最终，dgx-21）：
  - `synthetic_2step_text`：latent diff **0/0** ✅ → **encoding 修复 GPU 验证通过**
  - `synthetic_2step_image`：prefix verify ✅；step0 **0.406** / step1 **0** → index/encoding OK，**VL 一次 forward hidden 仍不等价**
  - `real_record .../000000` 前 2 step：prefix verify ✅（span 定位；step0 曾有 3 个 token-id 误匹配 `[293,520,600]`）；latent diff **10 / 7.5** → 同为 forward 语义问题
- **结论**：encoding/index 层修复完成；navigation 多图 **trajectory-once 单 forward 不可当作 legacy 等价**；**不得默认 `PACKED_FORWARD=1`**

## 2026-06-20：trajectory-once 多图不等价定位到 Qwen-VL forward 语义

- 按人类要求改用空闲/可抢占节点重跑验证；取消 normal pending job，提交并完成：`456832`（preempt `dgx-36`, validate）与 `456833`（preempt `dgx-47`, debug）。
- `validate_trajectory_once_2step.py` 增加 alignment 诊断：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、Qwen `position_ids`、以及 `get_image_features()` 前缀 diff。
- 结果：
  - synthetic 2-step text：diff `0/0`，且 position ids 对齐。
  - synthetic 2-step image：step0 latent diff `0.40625`，step1 `0`；但 step0 的 input ids、attention mask、pixel values、image grid、position ids 全部与 full 前缀一致，`image_features_prefix_max_diff=0.0`。
  - real record `train/shard_001_180/000000` 前 2 step：diff `10.0/7.5`；两步的输入、图片张量、grid、position ids、image features 前缀也全部对齐。
- `debug_trajectory_once.py` 修复 synthetic 2-step 构造为真实 user/assistant 交错轨迹；GPU job `456833` 证实 text 2-step hidden/logits/latent 都完全一致，而 fake image step0 即使所有编码/图片/position 对齐，prefix region hidden max diff 仍为 `2.0`、logits diff `0.5`、latent diff `0.625`。
- 结论更新：早期 mismatch/index 问题是实现 bug，现已排除；剩余多图不等价发生在 Qwen-VL language-model full sequence forward 中，而不是 tokenization、latent index、vision feature 或 position id 对齐错误。trajectory-once/full-trajectory 对当前多图 SFT2 仍不得作为默认语义等价优化。

## 2026-06-20：SFT2 语义安全 trajectory-aware batching 原型

- 根据 trajectory-once 不等价结论，改为实现不改变语义的 batching 优化：新增 `src/nimloth/training/sft2/trajectory_sampler.py`，按 record 将连续 step indices 放入同一 micro-batch，但每个 prefix 仍是 DataLoader batch 中的独立 row，Qwen 仍执行 legacy per-prefix forward，不做 full-trajectory single forward。
- `trainer.py` 新增 `--trajectory-aware-batching` 路径：非 packed-forward 时可使用 `TrajectoryAwareBatchSampler`；DDP 下按 batch index 切分并补齐，使各 rank micro-batch 数一致；每 epoch 调用 `set_epoch()` 保持确定性 shuffle。
- `cli.py` 增加 `--trajectory-aware-batching/--no-trajectory-aware-batching`；`train_vagen79_default.slurm` 支持环境变量 `TRAJECTORY_AWARE_BATCHING=1` 传参。
- 新增 `tests/training/sft2/test_trajectory_sampler.py` 覆盖连续 step 分组和 DDP rank 切分。
- 本地验证：`python -m py_compile` 覆盖新 sampler、trainer、cli、测试与 slurm 相关改动通过；本地缺 torch/pytest，不能运行 pytest。尝试同步服务器做 .venv pytest/smoke 时 SSH banner exchange timeout，尚未完成远程验证。

## 2026-06-20：trajectory-aware batching 远程 smoke 验证

- 服务器 SSH 恢复后，已同步 `src/nimloth/training/sft2/*.py`、相关 common 文件、Slurm 与 tests 到 `/project/peilab/atst/nimloth`；远程 `.venv` 验证：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/training/sft2/test_trajectory_sampler.py tests/training/sft2/test_cli.py` -> `4 passed`。
- 恢复并强化 packed-forward 安全阀：`--packed-forward` 必须同时传 `--allow-approx-trajectory-once`，Slurm 只有在 `ALLOW_APPROX_TRAJECTORY_ONCE=1` 时才追加 override；避免同步过程中丢失 guard。
- 新增 1-GPU smoke 脚本 `outputs/experiments/training/sft2/smoke/trajectory_batch_smoke.slurm`（服务器临时脚本），对比 `TRAJECTORY_AWARE_BATCHING=0/1`，使用 `max_train_records=2`、`batch_size=2`、`grad_accum=1`、`vision_tune=freeze`、无 EMA，验证训练 loop 可跑通。
- smoke 结果：
  - baseline job `456857`（preempt `dgx-05`）：跑完 1 epoch，`global_step=19`，val 正常输出。
  - trajectory-aware job `456858`（preempt `dgx-36`）：跑完 1 epoch，`global_step=20`，val 正常输出。
  - 两者均非 packed-forward，仍执行 legacy per-prefix Qwen forward；初步 step timing 显示 trajectory-aware 当前 forward 累计均值更低，但该对比跨节点且样本顺序/step 数不同，只能说明功能可用，不能作为严格速度结论。
- 下一步若要决定是否默认启用，应在同一 8GPU/同一配置下做 A/B：`TRAJECTORY_AWARE_BATCHING=0/1` + `STEP_TIMING=1`，最好配合 preprocess cache 与相同 max_records，比较 epoch wall time、current_forward、next_forward、batch_prep。

## 2026-06-20：8GPU trajectory-aware batching A/B 与缓存终端 batch bug 修复

- 在 8GPU preempt 节点上做了实际 A/B 执行。
- 过程中发现 `trajectory-aware-batching + preprocess cache + DDP` 暴露一个真实 bug：某些 rank 收到 terminal-only cached micro-batch 时，`compute_step_wm_loss()` 的 dummy next-forward 回退访问 `items[0]["messages"]`，但 cached items 之前未保留 `messages`，导致 `KeyError: 'messages'`。已在 `preprocess_cache.py` 修复：`CachedTransitionDataset.__getitem__()` 现在把当前 `messages` 注入 entry，`collate_cached_transition_batch()` 也把它传入 items。
- 修复后，8GPU no-checkpoint A/B job `456886`（warm cache partially unfair）与更公平的 cache-hit A/B job `456888`（同节点 `dgx-47`，shared cache hit）均跑通。
- 公平对比 job `456888` 配置：`max_train_records=8`, `batch_size=2`, `grad_accum=4`, `llm_tune=freeze`, `vision_tune=full`, `vision_ema=true`, 8GPU DDP，checkpoint monkeypatch 为 no-op 以避免保存时间污染。
- `456888` 结果：
  - off: `elapsed=59s`
  - on (`--trajectory-aware-batching`): `elapsed=57s`
  - 两者 preprocess cache 均为 hit。
  - 从 `train_step_log.csv` 看，首个 optimizer step 前启动/加载阶段：off 约 `43.6s`，on 约 `40.2s`；首个 step 到 val 结束的活跃训练阶段：off 约 `8.44s`，on 约 `10.04s`。
- 当前结论：trajectory-aware batching **功能正确且 DDP/cached 路径已修复**，但在这组小规模 8GPU cache-hit A/B 中 **没有明确训练阶段加速，甚至活跃训练阶段略慢**；end-to-end 仅有约 `2s` 改善，更像启动抖动而非稳定吞吐提升。暂不建议默认启用，应视作可选实验开关。

## 2026-06-20：vLLM prefix-invariance probe 跑通

- 目标：验证 Qwen2.5-VL full trajectory 中“同一 image prefix 单独前向 vs 作为 full trajectory 前缀部分”输出不一致，是否只存在于 HF `transformers`。
- 远程 probe 路径：`outputs/experiments/training/sft2/smoke/probe_vllm_prefix_invariance.py`。
- 解决 vLLM 启动环境问题：将 HOME/cache 重定向到项目目录；加载 `nvhpc-hpcx-cuda12/23.11`；设置 `CUDA_HOME=/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/cuda/12.3`；最终使用系统 `gcc/g++` 作为 `CC/CXX`，避免 flashinfer/Triton 编译器冲突。
- job `456934` 跑通，日志：`outputs/experiments/training/sft2/smoke/vllm_prefix_456934.log`：
  - `2step_text`: `input_ids_prefix_match=true`, `max_abs_prompt_logprob_diff=0.0`。
  - `2step_image`: `input_ids_prefix_match=true`, `max_abs_prompt_logprob_diff=0.280426025390625`, `mean_abs_prompt_logprob_diff=0.049253354532205314`。
- 控制实验 job `456937` 关闭 vLLM prefix caching 后结果不变，日志：`outputs/experiments/training/sft2/smoke/vllm_prefix_nocache_456937.log`：
  - text 仍为 `0.0`；image 仍为 `0.280426025390625`。
- 结论：该 prefix non-invariance 不是 HF `transformers` 独有；vLLM 的 prompt logprob 层面也复现了 image prefix 非不变性。它也不像 vLLM prefix cache 造成。默认 SFT2 仍不能启用 full-trajectory/packed-forward 近似路径，除非另有严格等价证明。

## 2026-06-21：SFT2 no-packed epoch_001 rollout eval 明显低于 baseline

- 训练 run：`outputs/experiments/training/sft2/2026-06-20/sft2_latentwm_default_8gpu_vllm_nopacked`。

## 2026-06-21：FA2 不能修复 SFT2 packed-forward 多图不等价

- 为了验证 `sdpa` 是否是 packed-forward 不等价的主因，给 `validate_trajectory_once_2step.py` 与 `validate_2step.slurm` 增加了 `--attn-implementation` / `ATTN_IMPLEMENTATION` 参数化，允许直接对比 `sdpa` 与 `flash_attention_2`。
- 已同步到服务器 `/project/peilab/atst/nimloth`，并运行 preempt jobs：`457345`（`sdpa`）与 `457346`（`flash_attention_2`）；pending normal jobs `457343/457344` 已取消。
- 结果：
  - `sdpa` (`457345`): text `0/0`; synthetic image step0 `0.40625`, step1 `0`; real record `train/shard_001_180/000000` step0 `10.0`, step1 `7.5`。
  - `flash_attention_2` (`457346`): text `0/0`; synthetic image step0 `0.78125`, step1 `0`; real record step0 `9.625`, step1 `7.375`。
- 两个 job 的 alignment 诊断完全一致且全部通过：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、`position_ids_eq_full_prefix=true`、`image_features_prefix_max_diff=0.0`。
- 结论：把 attention backend 从 `sdpa` 切到 `flash_attention_2` **不能恢复** Qwen2.5-VL packed-forward 的 prefix-equivalence；问题不只是 `sdpa` 的已知精度/实现问题。FA2 在 synthetic image 2-step 上甚至更差，真实 record 也仍保持很大的 latent diff。
- 按人类要求在超过 1 epoch 后停止；停止时训练已到 `epoch=2`, `global_step=959`，但用于对比的是已完整落盘的 `epoch_001` checkpoint。
- `epoch_001` rollout eval 最终通过复用 env job `456981` 的 external env 跑通；结果文件：
  - `outputs/experiments/training/sft2/2026-06-20/sft2_latentwm_default_8gpu_vllm_nopacked/eval_rollouts/sft2_eval_nopacked_epoch_001/summary_0.json`
- `epoch_001` 结果：
  - val: `14/360`, `success_rate=0.03888888888888889`
  - test: `15/300`, `success_rate=0.05`
- baseline 对比使用同流程下先前 `init` 评估（SFT1 merged init，来自 `2026-06-19/sft2_latentwm_default_8gpu_tokenfix/eval_rollouts/sft2_eval_tokenfix_init/summary_0.json`）：
  - val baseline: `0.3277777777777778`
  - test baseline: `0.22333333333333333`
- 结论：当前这条 SFT2 no-packed run 在 `epoch_001` 时 rollout success rate **显著低于 baseline**：
  - val 下降 `0.2888888888888889`
  - test 下降 `0.17333333333333334`
- 备注：为了拿到该结果，经历了多次 env server 失败/不可达；最终可用的是 baseline 任务 `456981` 对应 external env。结论本身有效，但本次 eval 基础设施不稳定，后续最好固定一个可复用的健康 env 入口。

## 2026-06-21：SFT2 value gamma 可配置 + LLM LoRA/vision-full pair2 训练健康启动

- 将 SFT2 value target 的折扣因子改为可配置：
  - `src/nimloth/wm/dataset.py`: `DEFAULT_VALUE_GAMMA` 从 `0.99` 改为 `1.0`；`expand_record_transitions()` / `iter_transitions_from_jsonl()` / `TransitionJsonlDataset` 接受 `value_gamma`。
  - `src/nimloth/training/sft2/dataset.py`: `TransitionQwenDataset` 透传 `value_gamma`。
  - `src/nimloth/training/sft2/cli.py`: 新增 `--value-gamma`，默认 `1.0`。
  - `src/nimloth/training/common/config.py`: 支持 YAML `loss.value_gamma`。
  - `configs/training/sft2/latent_wm_value.yaml` 与 profiling config 显式设置 `value_gamma: 1.0`。
  - `tests/test_wm_transition_dataset.py` 更新默认 target 期望，并新增显式 `value_gamma=0.9` 覆盖。
- 本地 `python -m py_compile` 通过相关 Python 文件；远程 pytest 由于环境/导入耗时卡住未拿到完整结果，需后续补跑。
- 为继续保留 vision full tune 同时打开 Qwen LLM LoRA，修复 PEFT LoRA 在当前环境中误走旧 `torchao=0.9.0` dispatcher 的问题：在 `configure_qwen_tuning()` 内让 `dispatch_torchao` 返回 `None`，绕过不兼容 torchao 分支。
- 尝试 `llm_tune=lora + vision_tune=full` 单卡/8DDP replica 时发生 OOM；随后启用实验性 pair2：`NGPUS=4`, `NIMLOTH_DDP_GPU_STRIDE=2`，每个 DDP rank 的 Qwen 副本通过 HF `device_map=auto` 分到两张 GPU。
- pair2 smoke 已证明可跑多个 optimizer step，无 OOM/通信崩溃；随后启动正式 1 epoch 训练：
  - job: `457209` on `dgx-47`
  - output: `outputs/experiments/training/sft2/2026-06-21/sft2_latentwm_llmlora64a128_vfull_pair2_ep1`
  - config: `latent_wm_value_epoch1.yaml`（未显式写 `value_gamma`，但已同步代码默认 `--value-gamma=1.0`）
  - settings: `llm_tune=lora`, `vision_tune=full`, `lora_r=64`, `lora_alpha=128`, packed-forward off, trajectory-aware batching off, `NGPUS=4`, `NIMLOTH_DDP_GPU_STRIDE=2`。
  - 健康启动证据：日志显示 LoRA 注入、`qwen_pair_parallel=true`, `rank0_pair=[0,1]`, vision EMA `shadow_params=582`；`train_step_log.csv` 已写到至少 `global_step=20`，无 OOM/ChildFailedError。
- 注意：一次后提交的重复 job `457216` 因资源 pending 被取消；实际健康运行的是 `457209`。

## 2026-06-21：FA2 不能修复 SFT2 packed-forward 多图不等价

- 给 `validate_trajectory_once_2step.py` 与 `validate_2step.slurm` 增加了 `--attn-implementation` / `ATTN_IMPLEMENTATION` 参数化，允许直接对比 `sdpa` 与 `flash_attention_2`。
- 已同步到服务器 `/project/peilab/atst/nimloth`，提交 normal jobs `457343/457344` 后又补提 preempt jobs `457345`（`sdpa`）与 `457346`（`flash_attention_2`）以加速验证；normal pending jobs 随后已取消。
- 结果：
  - `457345` (`sdpa`): text `0/0`; synthetic image step0 `0.40625`, step1 `0`; real record `train/shard_001_180/000000` step0 `10.0`, step1 `7.5`。
  - `457346` (`flash_attention_2`): text `0/0`; synthetic image step0 `0.78125`, step1 `0`; real record step0 `9.625`, step1 `7.375`。
- 两个 job 的 alignment 诊断都通过：`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values`、`position_ids_eq_full_prefix=true`、`image_features_prefix_max_diff=0.0`。
- 结论：把 attention backend 从 `sdpa` 切到 `flash_attention_2` **不能恢复** Qwen2.5-VL packed-forward 的 prefix-equivalence；问题不只是 `sdpa` 的已知精度/实现问题。FA2 在 synthetic image 2-step 上反而更差，真实 record 上也仍保持很大的 latent diff。

## 2026-06-30：squash 合并 feat/rl 与 feat/reconstruct 到 dev

- 按人类要求审查并 squash 合并 `feat/rl` 与 `feat/reconstruct` 到当前本地 `dev`（本地未发现 `feat/dev` 分支和 `Jun21-SFT2` tag）。
- 合并提交：
  - `240f6bc feat(rl): squash online RL training pipeline`
  - `e95757f config: squash RL and baseline launch parameters`
  - `b78b3ee feat(reconstruction): squash decoder training and diagnostics`
  - `c9291a4 config: add reconstruction training parameters`
  - `c617da8 docs: record merge review issues`
- 参数/config 变更已拆为 RL/baseline 与 reconstruction 两个 config 提交；代码与诊断/文档变更分别在 RL、reconstruction squash 提交中。
- 审查发现的问题已记录到 `ai_tasks/merge_dev.md`，本轮未修复。
- 合并冲突处理：保留当前 dev 的 `.memory/memories.jsonl` 与 `AI_branch_progress.md` 主体，避免覆盖当前进度/手工合并 memory；`external/VAGEN` 指向 feature 分支 pointer `93c1124aeaa7850098f46f2b708ee224ba894861`；`qwen_tuning.py` 同时保留 torchao workaround 与 RL 的空 `modules_to_save`。
- 验证：`python -m py_compile` 覆盖新增 RL/environment/agent/reconstruction/wm/eval 源码与相关测试文件通过；`bash -n` 覆盖新增/修改 shell/slurm 脚本通过；pytest 未通过环境验证（系统 Python 无 pytest，`.venv` torch import 缺 `libstdc++.so.6`）。
- 归档进度文件：`ai_tasks/ai_progress/archives/2026-06-30/merge_feat_reconstruct_rl.md`。
