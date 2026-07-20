# RL k>1 inject + WM/ValueHead 连续动作实现进度

日期：2026-07-20

## 目标

在本地 feature worktree 中实现：

1. RL 全链路 metadata-driven `k>1 + inject`，保留 k=1；
2. 独立 `qwen | wm_value` rollout policy；
3. `wm_value` 从 Qwen GT state 开始，segment 内连续使用 WM predicted state + greedy ValueHead，segment 结束后从真实 observation 重同步；
4. 连续 trajectory window 的多步 dynamics loss；
5. WM policy transition 不进入 Qwen PPO。

## 人类边界

- worktree：`/workspace/remote2/nimloth-feat-rl-kgt1-wm-multiaction`
- branch：`feat/rl-kgt1-wm-multiaction`
- 首期仅支持 `inject`，不支持 `generate`。
- 首期只使用 greedy，不做 beam search。
- `fast_path_horizon` 与 multi-step loss horizon 配置化，默认2。
- 暂时不运行 smoke，不提交任何服务器任务。

## 当前计划

1. RED：协议、schema、fast-path state machine、multi-step loss、PPO ownership 测试。
2. GREEN：metadata-driven k-token prompt/extraction/projector/checkpoint。
3. GREEN：`wm_value` rollout 与 JSONL schema。
4. GREEN：多步 dynamics loss 与 actor source mask。
5. REFACTOR：配置、实验入口、README。
6. 仅运行本地单元测试、compile 和静态检查。

## 已完成

- 创建新 worktree/branch；未启动实验。
- 人类确认 inject-only、greedy、连续 WM predicted segment 和 Qwen/WM policy ownership。
- RL protocol 改为从 checkpoint metadata 读取任意正整数 k，并全链路使用完整 inject query block、显式 token IDs 和 k-aware StateProjector。
- 新增 `qwen | wm_value` rollout policy。`wm_value` 在 segment 内只递归使用 predictor state，达到 horizon 后从真实 observation 经 Qwen 重同步。
- JSONL 新增 policy/state/fast-step/protocol/behavior-logprob metadata；WM action 不保存伪造 Qwen log-prob。
- Qwen rollout 与 PPO forward 共用同一 prompt 和真实 observation history；temperature/top-p 后的实际采样分布用于 old/new log-prob。旧 JSONL 无语义版本时自动排除出 PPO。
- 新增连续 trajectory window 的递归多步 dynamics loss，并用 mask 处理短 window。
- ValueHead 对 WM fast-path transition 使用从 segment GT 起点重建的 predicted behavior state，而不是错误使用当前帧 GT state。
- checkpoint/resume 保存并校验 k、query mode/token IDs、projector dims、policy、两个 horizon 与 loss decay。
- 新增 k8 WM fast-path 配置与 README。

## 文件修改

- `src/nimloth/training/rl/{rollout,trainer,loss,cli,checkpoint}.py`
- `src/nimloth/training/rl/{README.md,__init__.py}`
- `experiments/training/rl/{rollout_env.py,README.md}`
- `configs/training/rl/{defaults.yaml,k8_wm_fastpath.yaml}`
- `tests/training/rl/` 下 protocol、fast path、多步 loss、PPO ownership、checkpoint 和 transition-window 测试
- `ai_tasks/rl_kgt1_wm_multiaction_plan.md`
- `AI_branch_progress.md`

## 验证

- RED 已确认：首批新测试因缺少 k>1/fast-path/multi-step/PPO ownership API，collection 4 errors。
- 本地 Nix Python 3.13 环境提供 torch/pytest/einops/transformers 等依赖。
- `PYTHONPATH=src:. python -m pytest -q tests/training/rl tests/test_wm_predictor_rollout.py tests/test_wm_planning.py tests/test_latent_extraction.py`：`70 passed, 1 expected warning`。
- `ruff check src/nimloth/training/rl experiments/training/rl/rollout_env.py tests/training/rl`：通过。
- `python -m py_compile ...` 与 `git diff --check`：通过。
- 按人类要求未运行 smoke，未提交服务器任务。

## Smoke preflight

- 人类已解除 smoke 与最小服务器任务限制，允许真实 k=8/FSDP smoke。
- 已按实验开始规则核对实验约束并尝试连接 `superpod-csejzhang`，SSH forwarding connection timed out；按服务器规则停止重试。
- 人类恢复连接后 preflight 已完成：
  - 真实 source=`.../sft2/2_ddpsyncfix_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/train/epoch_002`；HF/processor/state projector/WM/value/training state 均存在，metadata 为 k=8/inject、hidden2048、projector input16384、epoch2/step2912 complete。
  - ENV worktree root=`b21ae10`、VAGEN=`bb26c0d`；`base_train.json` 含 `tasks` 1200条，loader读取该列表并以 `seed % 1200` 选任务，因此 seeds1..4 是明确训练数据。
  - W&B `nimloth-rl` 已有 numeric IDs 到62，下一 ID=63。
  - normal 当前有42张空闲GPU，多个单节点可提供2GPU。
- 已准备独立 k8 WM fast-path smoke config、runner 与 Slurm wrapper；尚未 reserve W&B run、创建远程输出或提交 job。
- 拟用2×H800、48CPU、180G、2h上限；预计实际8–15分钟，输出约60–100GiB。step1完整保存后由新 torchrun 以相同world-size恢复到step2。

## 待确认/风险

- 尚未使用真实 k=8 checkpoint、真实 processor 或环境验证；当前只能声明本地代码和单元测试通过。
- 未验证 GPU/FSDP checkpoint 的实际 tensor 与 optimizer resume；保留为后续 smoke gate。
- `generate` query mode 和 beam search 明确不在本次范围。
