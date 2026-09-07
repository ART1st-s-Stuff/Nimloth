# 旧数据训练及评估验证（准备中，未提交）

## 目的与授权

人类已纠正“只评估旧权重”的误解，要求从此前 checkpoint 启动 SFT stage 1、stage 2，再评估新结果，用于验证近期实现改动；不作为后续正式训练的默认起点。2026-09-07 人类要求继续。原实现、测试与远程实验授权持续有效。

执行顺序：初始化 checkpoint → stage 1 一轮 → 显式合并导出 → stage 2 一轮 → 显式合并导出 → 分别对两个阶段导出策略进行 direct rollout。训练一次，不自动失败重提；两个阶段均完整消费旧训练集。不用旧权重直接评估来替代训练。

## 输入与当前证据

远程根目录 `/project/peilab/atst/nimloth`。初始化模型为 `experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface`，是 VAGEN PPO 初始化权重，不是新 SFT stage 1 产物。四个 safetensors shard 与 index 已核验存在。阶段间必须使用本次显式导出的完整 HF checkpoint。

旧数据根 `outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c`。manifest 记录生成策略为旧 VAGEN step 60、strict valid 筛选；不得与当前仍在生成的新版 step60 数据混淆。

- `train_success.jsonl`：613 条成功轨迹、7309 个回答，SHA256 `f7df535a8b11a1e19670ffd57a84c3f18db1741d82080f5cb50eee88d32dd2b2`。
- `val_all.jsonl`：355 条轨迹、6054 个回答，SHA256 `ae47dee83ea449e8068791f2146ea5273369668ad0324c36d2ea03c7583b8661`。
- DINO cache：`outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe`。原始 image path 覆盖训练 7922、验证 6409 张图，无缺项。canonical loader 的实际 shard 与有限值检查仍在执行。
- DINO 身份：`facebook/dinov2-large`，revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`，processor `7d65a7de8788e87d`，4×4 网格、1024 维；不更新 teacher。

2026-09-07 以固定 VAGEN assets 和 `_sync_reset` 的 `seed % len(tasks)` 恢复任务，按完整任务 JSON 比较：train/val 存在一个相同任务，分别是 long_horizon_train seed 1080 与 1096，场景 FloorPlan425。因此原 val 仅作旧数据离线诊断，不能宣称完全独立 held-out；保留人类指定原数据，不悄悄去重改变实验。

独立 rollout 使用 base/common_sense 各 seeds 1–60，共 120 个不同任务。与训练任务内容及场景交集均为零。每阶段 120 episodes，共 240；贪心，20 个环境步、512 个回答 token、TP1。两条评估使用独立环境服务，或完全串行，禁止无证明共享有状态服务。此验证不含 WM rollout，因为当前训练只到 stage 2。

## 参数、资源与边界

准备参数：每阶段 1 epoch，LoRA r64/alpha128，batch 1、gradient accumulation 8、world size 8，学习率 1e-6、embedding 学习率 5e-6，max_length 12000、max_pixels 100352。Stage 1 为 K1 generate、回答 CE；stage 2 为 K16 inject、CE 与 query/DINO alignment 权重各 1。实际 trainable parameter 清单需要在启动前记录，不根据 LoRA 名称推断视觉部分冻结。

拟申请 normal/normal_qos、peilab、单节点 8 GPU，6 小时上限。提交前刷新资源，明确 CPU/内存、rank 映射、端口、运行目录及最终命令。使用 `.venv-vagen-main/bin/python3 -m torch.distributed.run`；2026-09-07 实测 torch 2.8.0+cu128、transformers 4.55.4、peft 0.19.1。无 W&B；保留训练逐步指标、导出检查及 rollout 汇总。

工作区 `/workspace/remote2/nimloth/.worktree/codex-unify-sft`，分支 `codex/unify-sft-training-evaluation`；远程 `.worktree/unify-sft-training-evaluation`。当前基准 commit `ed7a1f1630f200730f1345602a3e95f62044c57a`，启动器尚未完成、最终 commit 尚未固定，因此本记录不能单独放行提交。

## 剩余启动门禁

- 完成启动器、CPU 回归及独立审核，记录完整命令和最终固定 commit。
- 核验 stage 2 trajectory 展开后的真实显存规模；不能用截断/子采样隐藏问题。
- 核验实际 PEFT 导出和阶段间词表/metadata 兼容、DDP trainable graph。
- 完成缓存实际加载检查和 checkpoint 必需 keys 检查。
- 固定唯一空输出目录、子进程清理、渲染 preflight、结束与失败标记、监控交接。
- 只操作本次精确作业，不干扰现有 556034/556035 等其他任务。

保存每轮 checkpoint；当前恢复未保存完整 RNG，不能声称逐位忠实恢复。失败时保留产物与日志，先定位全部后续阶段，再决定新身份重跑；本次不自动延长预算。

## 启动前修复和已完成检查

canonical stage2 现在先建立回答索引再采样，batch1 为一个完整回答前缀，完整覆盖 7309 个训练回答；DDP sampler 为对齐 rank 会补齐最多 world-1 个样本。world8/GA8 下约 115 optimizer steps；stage1 仍按 613 条轨迹，约 10 steps。旧 query checkpoint 的 epoch 数字不代表相同消费边界，本次仅 fresh start。回答尾部即使 query 完整，也不得截断动作；collator 现编码完整前缀并对超长输入报错。格式诊断仍使用原轨迹 dataset，保持原诊断语义。

本地 CPU SFT/SFT1 回归 159 passed；包含每个回答的历史、标签和 DINO 对齐，以及 query 完整但动作被截断的拒绝测试。独立审核定位的问题已据此修复；真实 GPU 路径仍待执行。

远程 canonical DINO loader 校验所有 shard 格式与 manifest，并读取全部 14331 张相关图像的特征验证有限值，通过；cache fingerprint `b50d261e2b533f3e`。checkpoint 四个 shard 的 825 个 index keys 均存在。远程 meta model + 实际 PEFT 配置核验：默认 suffix 也命中视觉 MLP gate/up/down 的 LoRA；原始模型基础权重冻结，视觉 MLP 和语言模型 LoRA、完整 embedding/head 可训练，stage2 另加 projector。原始 padding 词表下 698 个 trainable tensors、770940928 个参数；注册/调整词表后最终精确数量以训练日志为准。不得声称视觉分支完全冻结。

资源细化为 normal/normal_qos，1 node / 8 GPUs / 96 CPUs / 600G / 6h。解释器、源码 pins、数据路径和模型路径显式写入 `experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh`。每阶段完整 offline val；另有原格式诊断32条，不与120条独立rollout混同。训练和导出串行，评估两臂各用独立环境 GPU 与 policy GPU，剩余卡闲置直至任务结束。子进程 HOME 仅指向本作业临时 AI2-THOR cache，release 只读链接，不能写共享运行时。

远程 torch2.8.0/transformers4.55.4/peft0.19.1 的真实 tiny Qwen 检查已通过：默认 auto adapter 保存、扩词表32→36、merge、HF重载，输入与输出权重逐元素一致且独立存储、generate协议回退正确。仅processor序列化使用隔离替身，不能替代本次真实tokenizer/大checkpoint运行证据。显式 `save_embedding_layers=True` 的外部adapter重复别名仍是已知限制，本次默认auto路径未触发，不扩大修复。

最终启动器5项CPU合同测试、完整Ruff、shell语法和diff检查通过；独立源码复审通过。远程Vulkan/cache/tool路径可用，/project剩余约510GiB；本次输出预计低于100GiB，启动器在剩余空间低于100GiB时拒绝启动。
