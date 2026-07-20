# RL k>1 inject + WM/ValueHead 连续动作实现进度

日期：2026-07-20

## 目标

在本地 feature worktree 中实现：

1. RL 全链路 metadata-driven `k>1 + inject`，保留 k=1；
2. `qwen | wm_value | qwen_wm` rollout policy；正式RL使用`qwen_wm`；
3. hybrid segment首步由Qwen从GT state采样并记录behavior log-prob，后续step连续使用WM predicted state + greedy ValueHead，结束后从真实observation重同步；
4. Qwen step以ValueHead为critic、使用标准`A=G-Q(s,a)` clipped PPO；WM step不进入Qwen PPO；
5. 连续trajectory window的多步dynamics loss，全部step训练WM predictor与ValueHead。

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
- 人类确认inject-only、greedy及hybrid ownership：segment首步Qwen sampled action做标准advantage PPO，后续WM/value action不做PPO；Qwen language full、WM predictor、ValueHead训练，vision与StateProjector冻结。
- RL protocol改为从checkpoint metadata读取任意正整数k，并全链路使用完整inject query block、显式token IDs和k-aware StateProjector。
- 新增`qwen | wm_value | qwen_wm` rollout policy。正式`qwen_wm`在Qwen首步后递归使用predictor state，达到horizon后从真实observation经Qwen重同步。
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
- 最终hybrid改动后，`PYTHONPATH=src:. python -m pytest -q tests/training/rl tests/test_wm_predictor_rollout.py tests/test_wm_planning.py tests/test_latent_extraction.py`：`74 passed, 1 expected warning`。
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
- 初版纯`wm_value` smoke方案被人类否决，因为没有训练Qwen；未reserve W&B、未创建输出、未提交job。
- 已按最终设计实现并本地验证`qwen_wm`：每条2-step segment为`Qwen sampled step → WM/value step`，同一次Qwen forward同时给出行为log-prob和GT k-query state；Qwen language full与WM/value联合更新，vision/StateProjector冻结。
- ID63 hybrid smoke已执行并失败：job`482447`在`dgx-21`运行`00:02:47`，W&B`wh351jfg`为failed；rollout成功生成4条真实`base_train`轨迹/8 transitions，全部行为归属为`qwen→wm_value`、state为`qwen_gt→wm_predicted`，但训练optimizer step为0、无checkpoint，不能resume。
- 失败根因：`encode_trajectory_hiddens()`调用`build_qwen_batch()`时漏传k=8，helper默认k=1并把query block归一化为单token，严格k=8提取器报malformed block。已增加真实RED回归测试并修复显式透传`latent_token_count`；本地全套`75 passed, 1 expected warning`。错误登记于`ai_rules/known_errors/E0031_forward_latent_k_to_qwen_batch.md`。
- ID63输出及`outputs/experiments/training/rl/progress.md`已记录失败与不可恢复原因。
- ID64 retry1随后执行：job`482474`在`dgx-10`运行`00:06:41`。k8 trainer编码修复生效，iteration1 forward全部有限（WM=0.00240672、Value=0.149349、actor=1.49e-08、entropy=1.54232），但optimizer step后trainable checkpoint全线NaN，fresh-process iteration2全部loss NaN；外部finite gate拒绝，W&B`ixvtbpqr`已显式标为failed。
- 根因经RED backward测试确认：top-p使用`-inf`mask，旧entropy的`where(p>0,p*logp,0)`只保证forward有限，autograd仍从`0*-inf`产生NaN；global grad clipping随后污染Qwen language、WM和ValueHead。已改为finite sentinel entropy并检查backward gradient，optimizer前新增non-finite loss/grad-norm fail-fast；登记`E0032_top_p_entropy_backward_nan.md`。ID64全部checkpoint不安全，不得resume；后续retry必须新ID/新输出。

## 待确认/风险

- 尚未使用真实 k=8 checkpoint、真实 processor 或环境验证；当前只能声明本地代码和单元测试通过。
- 未验证 GPU/FSDP checkpoint 的实际 tensor 与 optimizer resume；保留为后续 smoke gate。
- `generate` query mode 和 beam search 明确不在本次范围。
