# AI_branch_progress.md — Nimloth 当前进展

本文件记录当前阶段的计划、进展、重要决策和失效记忆。每个 AI 会话开始后应阅读本文件。

---

## 当前阶段：项目初始化 / memory skill

日期：2026-06-10

### 已确认

- 项目名称：Nimloth
- 项目目标：World Model Agent
- 技术栈：Python 机器学习
- 当前重点：建立 AI 友好的轻量 memory/task 管理方式。
- Memory 设计原则：短小、可搜索、由人类审批、以文件段 evidence 为依据，不写长篇总结。

### 2026-06-21：本地/仓库记忆分流与 .local 迁移

- 已将服务器说明移至 `.local/SERVER.md`，并更新 `AGENTS.md` / 其他引用。
- memory CLI 现支持 repo/local 两个存储：repo 记忆仍在 `.memory/`，环境相关记忆写入 `.local/memory/`。
- 新增 `.local/scripts/query-resources.sh`，用于展示 superpod 各节点剩余资源。

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

## 2026-06-21：baseline env reproduction 改动整理到 fix/env-reproduction

- 人类确认 baseline/env reproduction 相关新改动应进入 `fix/env-reproduction` worktree，而不是 `dev`；`dev` 保持 SFT2 旧 checkpoint 测试状态，暂不同步 prompt_format 新修正。
- 已在 `external/VAGEN/verl` 提交 `869ff12b Improve navigation val wandb metrics`，用于 wandb 中展开 navigation validation 指标。
- root `fix/env-reproduction` 同步 baseline train/val prompt_format 修正、vLLM rollout/backend 脚本、checkpoint pruning policy、wandb val watcher/launcher、以及 VAGEN submodule 指针。
- 已从 `../nimloth-dev` 回退此前误放入的 baseline/env reproduction 文件；dev 仅保留原有 SFT2 speedup 相关未提交改动。

## 2026-06-21：同步 AGENTS 合规的未提交工作区改动

- 按人类要求，将当前工作区中符合 AGENTS 规则、且与 baseline env reproduction 工作流相关的剩余改动同步到 `fix/env-reproduction` worktree。
- 同步内容：`ai_rules/03_experiments_and_data.md` 昂贵任务阈值澄清、`experiments/README.md` VAGEN 论文超参数说明、`experiments/training/baseline/launch_hold_train_resume.sh` 的 hold-job 复用/保留逻辑。
- 未同步 `.memory/memories.jsonl`，因为 AGENTS 规定不得手动编辑 memory 存储；该改动仅保留在 archive 分支。
- 其余未整理的大批 baseline 脚本改动已在 `archived/2026-06-21-baseline-unsorted` 分支保留，不进入本 worktree 主线提交。

## 2026-06-21：VAGEN Nimloth 改动移植到 upstream vagen-legacy

- 人类指出 VAGEN main 分支无法复现原文效果，应使用 upstream `vagen-legacy`。
- 已在 `external/VAGEN` 从 `mll-lab-nu/VAGEN` fetch `vagen-legacy`，创建本地分支 `nimloth/vagen-legacy`。
- 已移植近期 Nimloth/VAGEN 关键改动到 legacy 代码结构：
  - 新增 `vagen/env/navigation/nimloth_format.py`，使用 `<|latent_state|><|action_start|><|action_(idx)|><|action_end|>`。
  - `vagen/env/utils/parse_utils.py` 支持 `prompt_format=nimloth` 与 `nimloth_wm`。
  - `vagen/env/navigation/prompt.py` 支持 Nimloth 单动作 prompt，并禁止 Nimloth 格式多动作。
  - `vagen/env/navigation/env.py` 同时接受 legacy action 名与 Nimloth action 名（如 `move_forward`）。
  - `vagen/trainer/ppo/ray_trainer.py` 加入稳定 env uid、validation metadata dump/sort、可选 validation composition guard。
  - legacy `.gitmodules` 恢复 `verl` submodule，指向 `ART1st-s-Stuff/verl` 的 `nimloth/main`（当前 gitlink `869ff12b`）。
- 已提交 VAGEN 子仓库 commit：`01c4566 Port Nimloth changes to VAGEN legacy`。
- 验证：`python -m py_compile vagen/env/navigation/nimloth_format.py vagen/env/utils/parse_utils.py vagen/env/navigation/prompt.py vagen/env/navigation/env_config.py vagen/env/navigation/env.py vagen/trainer/ppo/ray_trainer.py` 通过。未运行训练/rollout。
- 注意：root 仓库仍有大量既有未提交改动；本次只在 `external/VAGEN` 子仓库提交，root submodule 指针显示 modified，尚未提交。

## 2026-06-21：VAGEN legacy reproduction 适配准备

- 已将 `fix/env-reproduction` rebase 到 `origin/main`；解决 `train_resume.slurm` 冲突时保留 `main` 的 `cd ${BASEDIR}` Hydra 修复和 fix 分支的 shared CLI includes。
- 已在 VAGEN legacy 子仓库新增提交 `acc0e75 Add wm alias for legacy navigation format`：legacy navigation parser/prompt 支持 `prompt_format=wm` 作为 `worldmodeling` alias。
- 新增 legacy reproduction 专用配置：
  - `configs/training/baseline/legacy_train.yaml`：navigation `base` + `common_sense`，`prompt_format=wm`，`max_actions_per_step=1`，`use_state_reward=false`，每类 `train_size=10000`。
  - `configs/training/baseline/legacy_val.yaml`：同 split 的 held-out validation，每类 `test_size=60`。
- 新增 legacy 运行入口：
  - `experiments/training/baseline/run_legacy_reproduction.sh`：调用 `python -m vagen.env.create_dataset` 生成 parquet，并调用 legacy `python -m vagen.trainer.main_ppo`，设置 `algorithm.adv_estimator=bi_level_gae`、`rollout_manager.use_multi_turn_reward=True`、`use_loss_mask=True`、`use_gae_mask=True`、`reward_model.enable=False`、`rollout_manager.max_turns=20`。
  - `experiments/training/baseline/legacy_preempt_reproduction.slurm`：2 个 preempt 节点，每节点 2 GPU env / 4 GPU train，在 head node 启 legacy `vagen.server.server`，再启动 Ray + legacy PPO。
- 验证：`bash -n` 通过；未启动昂贵 Slurm 训练，等待人类确认资源/输出方案。

## 2026-06-30：feat/reconstruct post-hoc reconstruction diagnostic 初版

- 在 `feat/reconstruct` worktree 开始实现 WM reconstruction diagnostic；目标是 post-hoc 评估，不接入 SFT2/RL 主 loss。
- 新增 `src/nimloth/wm/reconstruction.py`：`WMImageDecoder` patch-query cross-attention decoder，输入 WM state embedding，输出 RGB image。
- 新增 `src/nimloth/training/reconstruction/`：独立 decoder 训练 CLI/trainer/README。训练时冻结 Qwen、`StateProjector`、`LatentWMPredictor`，只训练 decoder。
- 新增 `src/nimloth/eval/reconstruction.py`：offline eval，比较 `oracle=decoder(s_next)`、`pred=decoder(wm_predictor(s_t,a_t))`、`copy=decoder(s_t)`、`shuffled_action`，输出 CSV/summary/sample images。
- 新增 `configs/training/reconstruction/{defaults,eval}.yaml` 和基础单元测试文件。
- 验证：当前容器无 `torch`/`pytest`，无法运行 pytest；已运行 `PYTHONPATH=src python -m compileall -q ...` 通过。
- 未启动训练、评估、rollout、Slurm 或任何昂贵任务。

## 2026-07-01 Reconstruction full decoder training paused

- User observed reconstruction visual quality is very poor and requested pausing the reconstruction training.
- Stopped Slurm job `462610` on `dgx-03`; confirmed no `nimloth.training.reconstruction.cli` process remains.
- Latest preserved checkpoints in output dir: `step_000068000`, `step_000068500`.
- Output dir: `/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/2026-06-30/reconstruct_decoder_sft2_full_4epoch`.
- Remote output README and `outputs/experiments/training/reconstruction/progress.md` were updated with pause status, command/config summary, latest checkpoint, and suggested diagnosis areas.
- Next step before any resume: diagnose why reconstructions are poor; likely areas include decoder target/input normalization, Qwen image hidden extraction alignment, state_proj / WM predictor latent scale, and whether predicted next-state latent is the correct decoder input.
