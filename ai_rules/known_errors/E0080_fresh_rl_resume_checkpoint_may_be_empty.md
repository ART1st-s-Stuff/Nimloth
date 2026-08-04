# E0080：fresh RL 的 resume checkpoint 可以为空

错误：正式 RL 从 SFT2 checkpoint 以 global step 0 全新启动时，batch 使用
`${INITIAL_RESUME_CHECKPOINT:?}` 检查变量，导致已定义但为空的合法值在 GPU allocation
内立即退出。

原因：shell 的 `:?` 同时拒绝未定义和空字符串，但 full controller 明确用空字符串表示
不读取旧 RL optimizer/checkpoint。preflight 只分别检查了变量存在和 controller 语义，没有执行
exact batch 的 fresh-init 门禁。

正确做法：batch 使用 `${INITIAL_RESUME_CHECKPOINT?}`，只拒绝未定义变量；正式重提前必须执行
覆盖空字符串的 batch 回归。没有完整 RL checkpoint 的失败任务必须换新实验 ID、W&B identity
和空输出目录。

证据：`ai_tasks/ai_progress/2026-08-04_rl_sft2_restart_id123.md`记录的job `505716` stderr；
`experiments/training/rl/run_vllm_online_ppo_full.sh`的 fresh/resume 分支；
`tests/training/rl/test_slurm_allocation.py`的 fresh-init batch 回归。
