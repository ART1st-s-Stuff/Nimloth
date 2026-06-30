# feat/reconstruct + feat/rl 合并审查记录

## 合并说明

- 目标分支按本地实际情况使用 `dev`：本地没有 `feat/dev` 分支，也没有发现 `Jun21-SFT2` tag。
- 两个 feature 分支都以 `b96f66d` 附近的旧 dev 为 merge base；合并时使用三方 squash，保留当前 dev 上较新的 SFT2 文件，避免把 feature 分支里缺失的 SFT2 后续优化误删。
- 参数/配置变更已尽量拆到单独提交：RL/baseline 参数一组，reconstruction 参数一组。
- 审查发现的问题本次只记录，不修复。

## 冲突处理记录

- `.memory/memories.jsonl`：`feat/rl`/`feat/reconstruct` 与当前 dev 的 memory id/内容冲突。为避免手工合并 memory 存储文件，保留当前 dev 版本；分支中的 memory 变动未并入。
- `AI_branch_progress.md`：保留当前 dev 版本，避免用 feature 分支较旧的大段进度覆盖当前 SFT2 进度。
- `external/VAGEN`：采用 feature 分支指向的 submodule pointer `93c1124aeaa7850098f46f2b708ee224ba894861`。本地 submodule 未初始化，未验证该对象内容。
- `src/nimloth/backbone/qwen_tuning.py`：合并保留当前 dev 的 `dispatch_torchao` workaround，同时采用 RL 分支的 `modules_to_save=[]`，避免 FSDP+LoRA 与 `modules_to_save` 冲突。
- 第二次合并 `feat/reconstruct` 时，RL 文件冲突保留已并入的较新 `feat/rl` 版本；只加入 reconstruction 增量。

## 审查发现的问题（暂不修）

1. `src/nimloth/training/rl/trainer.py` 的多 GPU 训练同步语义不完整。
   - `world > 1` 时只把 Qwen 包成 FSDP，但 `state_proj`、`wm_predictor`、`value_head` 仍是普通本地模块，并且参与同一个 optimizer（约在 `trainer.py:481-499`）。这些模块的梯度不会跨 rank all-reduce，可能导致各 rank 参数分叉。
   - 同一训练循环中每个 rank 都会调用 `collector.collect()`（约 `trainer.py:535-543`），没有只在 main rank 收集并广播，也没有明确按 rank 分片环境/输出目录；多 rank 可能重复或竞争环境与 rollout 输出。

2. `src/nimloth/training/rl/trainer.py` 的 PPO advantage 标准化在 batch 很小时有 NaN 风险。
   - `advantages.std()` 默认是 unbiased 估计；当 batch size 为 1 时结果为 NaN（约 `trainer.py:608-610`）。若配置或可用 transition 数导致单样本 batch，会污染 actor loss。

3. `src/nimloth/training/reconstruction/trainer.py` 的 resume 不保留历史 best validation 状态。
   - `_maybe_resume()` 只恢复最近 `step_*` checkpoint 的 decoder/optimizer/step（约 `trainer.py:83-105`），主循环随后把 `best_val` 重置为 `inf`（约 `trainer.py:327`）。恢复后可能用较差 epoch 覆盖 `best/`。

4. `src/nimloth/training/reconstruction/trainer.py` 的固定 preview 选择会遍历整个 val dataset。
   - `_fixed_val_preview_items()` 为了挑样本遍历 `range(len(val_ds))` 并逐条 collate（约 `trainer.py:106-126`）。大 validation jsonl 下启动成本可能较高。

5. 合并后的 `AGENTS.md`/`ai_rules/*` 包含 feature 分支的规则调整。
   - `AGENTS.md` 标记为人类编写；本次是按“合并 feature 分支”执行而带入这些修改，但未单独判断这些规则改动是否仍符合当前人类意图。若需要更严格保护，可单独审阅这些规则变更。
