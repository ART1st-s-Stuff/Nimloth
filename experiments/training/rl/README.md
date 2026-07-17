# RL 实验入口

详细协议见 `src/nimloth/training/rl/README.md`。

## 当前状态

ID11 只保留 FSDP/checkpoint mechanics 证据；它的 rollout 丢失真实任务文本，因此所有质量结论无效，禁止 resume/reuse。

新的动态 RL 必须先通过：

1. corrected k=8 protocol integration smoke；
2. 固定 heldout `base` seeds 1–20、greedy、evaluation-only baseline；
3. 人工审阅 baseline artifact 后，才能创建新的 quality pilot 身份。

当前资源入口`dynamic_fsdp_k8_fragmented_5plus3_env48.slurm`只接受`RUN_MODE=smoke`；它固定使用normal 5@dgx09+3@dgx27 trainers与独立preempt dgx48 env GPU，在同一allocation内重跑bounded create/prompt/reset/schema/close preflight，并明确拒绝baseline/pilot。dgx48已由job478976独立preflight通过。

## 主要文件

| 文件 | 用途 |
|---|---|
| `dynamic_env_server.slurm` | 当前 clean worktree/pinned VAGEN 的 AI2-THOR service |
| `dynamic_env_preflight.slurm` | 单GPU bounded create+prompt+reset+schema+close gate；`/health`不能替代 |
| `dynamic_fsdp_k8_fragmented_5plus3_env48.slurm` | 当前normal 5@dgx09+3@dgx27 world8 trainers + preflight-proven preempt dgx48 env |
| `dynamic_fsdp_k8_fragmented_6plus2_env1.slurm` | 历史ID19 attempt2入口；dgx13 env create超时，不得复用 |
| `dynamic_fsdp_k8_fragmented_3plus2plus2plus1_env1.slurm` | scheduler-selected 3+2+2+1 normal trainer fragments入口；当前因需要五个碎片同时满足而未采用 |
| `dynamic_fsdp_k8_fragmented_4plus2plus2_env1.slurm` | 历史4+2+2 fixed-node smoke入口 |
| `prepare_k8_sft2_init.py/.slurm` | 从稳定 SFT2 snapshot 构建 immutable k=8 RL init |
| `rollout_env.py` | 单进程 schema-v3 rollout producer |
| `run_e2e_smoke.sh` | JSONL/FSDP mechanics test；不能替代真实 env integration |
| `dynamic_fsdp_k8_smoke.yaml` | 两任务、最多两步的 corrected protocol smoke |
| `dynamic_fsdp_k8_baseline20.yaml` | evaluation-only fixed-20 heldout baseline |
| `dynamic_fsdp_k8_pilot.yaml` | 仅保留 corrected future config；baseline 审批前不得提交 |

旧 `diagnose_eval.py` / `debug_action.py` 已删除，因为它们会重建 taskless generic prompt，不能用于新的 baseline。

## 必须满足的 runtime 协议

- env server 与 trainer 使用同一个 clean server worktree；VAGEN 必须为 `e7cc2d01584abcab1e49ba4a6b18ba2067fb6762`。
- 禁止使用旧 `exp-vagen-1action` env worktree。
- `prompt_format=source_eval_mode`，环境值与 SFT collection 完全一致。
- system prompt、每轮 `obs_str`、policy-generated assistant response、thought/action behavior token log-probs、step reward/final reward全部进入 schema v3。
- `task_instruction` 必须匹配 initial observation 和每步 `info.instruction`。
- full history `history_window=112`；thought/action sampling使用配置 temperature/top-p。
- rollout、PPO 和 latent encoding 逐字重放同一 transcript。
- dynamic value ranking weight 为 0；`rl.batch_size` 是 microbatch，每轮全部 transitions 消费一次。
- checkpoint/resume 严格比较 `rollout_protocol`；旧协议 checkpoint 自动拒绝。

## Baseline

`dynamic_fsdp_k8_baseline20.yaml`：

- `training.evaluation_only=true`
- `iterations=0`
- heldout `base`, seeds 1–20
- `temperature=0`, `top_p=1`
- 不执行 optimizer step，不写 `final/`

launcher gate 会逐条对照 `base.json[seed % len(tasks)]`，并检查 observation/assistant/reward 长度、task text、reward 总和、W&B run 和零 optimizer artifact。

## 提交前

按项目实验协议：

1. commit code/config/docs；
2. 同步 server worktree 并确认 clean、HEAD/submodule commit；
3. 新建 output identity 和 W&B ID；
4. 运行 `on-experiment-start`；
5. 只提交获批的 smoke/baseline；
6. 结束后运行 `on-experiment-end` 并更新 progress。
