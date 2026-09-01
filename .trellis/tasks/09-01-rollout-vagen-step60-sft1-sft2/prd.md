# 基于 VAGEN step 60 rollout 生成 SFT1/SFT2 数据集

## Goal

以远程 VAGEN `global_step_60` actor checkpoint 为冻结推理策略，在不可读取原始 VAGEN source commit 的条件下使用经审查、证据绑定的 runtime reconstruction，采集有明确 split、prompt/environment/generation provenance 的 navigation rollout；据此生成新的 SFT1 监督数据和可供当前 SFT2 trajectory 管线继续处理的数据，同时不伪造 CoT、reward、transition、checkpoint 或 source-code parity。

## User value

为后续 Nimloth SFT1 与 SFT2 训练提供相同策略来源、可审计且互相对应的新数据，替代旧 checkpoint/旧 runtime 的 rollout 数据。

## Confirmed facts

- 用户指定 checkpoint：`/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60`。
- 只读远程核验显示 checkpoint 存在、约 19 GB，包含 world-size 8 的 8 个 actor model shards、8 个 extra-state shards、`data.pt` 及 tokenizer/config 文件。
- `actor/huggingface/` **没有模型权重文件**，不能直接作为 HF/vLLM policy 目录；启动前必须使用源 VAGEN/VERL 兼容机制恢复/导出 actor，且验证 architecture/tokenizer/shard 完整性。
- 源 run 的 resolved log 记录 VAGEN commit `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49`，源 checkpoint world size 8，基础模型 `Qwen/Qwen2.5-VL-3B-Instruct`。
- 源数据资产由该 run 生成：train 20,000（`base` 10,000 + `common_sense` 10,000）、test 128（各 64）；dataset seed 为 42。源 env config 使用 `prompt_format=grounding_worldmodeling`、`max_actions_per_step=1`、`format_reward=0.02`、`invalid_action_penalty=-0.2`、`success_threshold=1.5`、无 state reward。
- 源 W&B generation table 的逐字 system prompt 明示 success reward 为 `10.0`，逐轮 transcript 保存实际 step reward；用户确认该 source run 未覆盖 VAGEN 默认移动距离，精确 `step_length=0.5` 米。正式 smoke 仍须把这些值写入 resolved runtime contract 并验证实际行为。
- 用户决定把 20,000 train rows 固定划分为 10 个互斥、类别平衡的 batch：每批 1,000 `base` + 1,000 `common_sense`，并保持各类别内部原始 parquet 顺序；本次只采集第 1 批，后续 9 批不自动启动。
- 用户决定在 batch1 内按共享 seed ordinal 确定性留出 10%：每个类别 ordinal 以十个为组时留出同一位置，使同一个 seed 的 `base`/`common_sense` 两行同时进入 held-out。预期 1,800 train rows + 200 internal held-out rows，seed 级无重叠；该 held-out 只证明 batch1 内部未见 seed，不代表未见环境分布。
- 用户选择以 `normal` partition、单节点总计 4 GPU 作为 launch contract 的资源方向：2 GPU 用于 policy TP2，2 GPU 用于 environment。正式提交前仍须重查资源可用性并对精确命令/时限取得 launch approval。
- 远程实际 parquet 核验：train SHA256=`3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`，前 10,000 rows 全为 `base`、后 10,000 rows 全为 `common_sense`；两个类别各有 10,000 个唯一 seed，并共享同一 seed 序列。test SHA256=`aa9b3903b35a83c7ce0f279c6f56b0469e14c3dcf4211b17cb1b1a208961f573`。
- `(eval_set, seed)` 检查显示 128 个 test keys 全部也存在于 train，因此当前 source `test.parquet` 不能作为无重叠 held-out 数据。本次首批只处理 train；不得把 source test 报告为 generalization split。
- 源 resolved log 的唯一生成块是 `actor_rollout_ref.rollout`：`do_sample=true`、`temperature=0.7`、`top_p=0.95`、`top_k=-1`、`n=1`、`ignore_eos=false`；该 run 没有独立的 `actor_rollout_ref.rollout.val_kwargs`。`max_turns=20`、每轮 response 256、trajectory 上限 6144（data 上限 16000）、window size 5。
- 对 12 个 source W&B generation tables 的 7,351 个可解析 turn 只读统计确认 step reward classes：strict valid non-success `0.02`（6,361）、strict valid success `10.02`（359）、invalid/forbidden/typo 或 malformed strict envelope `-0.2`（596）、too-many otherwise-valid actions `0.0`（35）；无效/多动作均未提取或执行 action。reconstruction 必须逐类 golden-test，不得把所有 format failure 合并猜测为同一 reward。
- 当前 `experiments/training/sft1/convert_rollouts.py` 可把 VAGEN dump 转成 SFT records，并输出 `train_all`、`train_success`、`val_all`、`test_all`；但当前版本只解析 `<action>`，而同类 hligb `grounding_worldmodeling` lineage 历史上需要显式 `<answer>` 解析。必须以 step60 的真实 raw transcript 做 schema/prompt/action gate，不能仅沿用历史判断。
- 当前 SFT2 reader 只接受 `record_format=nimloth_trajectory_v1`。普通 state 必须保留 rollout 中真实 assistant response；terminal observation 也必须有同一 observation 对应的真实模型生成 CoT，不能使用固定或占位 CoT。
- 用户决定 terminal observation 直接由同一个 VAGEN step60 policy 再生成一次完整 CoT 与 draft action。该 draft action只用于让真实生成到达动作边界并保留审计证据；不得执行、不得新增 transition/reward/success，也不得进入已执行 `action_indices`。SFT2 state 持久化生成响应中止于 state/action boundary 的 `terminal_assistant_prefix`。
- 用户明确：SFT1 与 SFT2 均不使用 terminal CoT 训练 LLM backbone；terminal CoT 的唯一用途是为最后一个 observation 获取 CoT-conditioned state。terminal 完整 response 不进入 SFT1 supervised assistant turns，SFT2 也只在 state encoder 边界消费其 prefix。
- 用户明确要求双轨持久化：SFT1/SFT2 的训练视图必须使用当前兼容的 K16 Nimloth prompt/response/action 格式；同时逐字保存 step60 实际使用的原始 prompt、完整真实聊天记录（包括普通 response 与 terminal full response）及其 hash/provenance。原始审计视图不得被改写后冒充 source transcript，训练视图也不得保留与 K16 supervision 冲突的 `<answer>` 格式指令。
- exact source VAGEN commit `fee3ffac...` 当前不可访问，且不在可用 object store/origin refs 中；用户明确批准返回规划阶段，采用 checkpoint、source log/config、W&B prompt/reward/image evidence 与可访问 legacy lineage 绑定的 runtime reconstruction。该路线不得声称执行了 exact source code。
- reconstruction 以可访问的 `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a` 为审计基线；它继承 `44be18c` 的 legacy batch API、compact action、0.5 m dynamics、1.5 m threshold 与 10.0 success reward，并已有隔离 compatibility mode。新的 step60 mode 必须以独立 patch commit 精确绑定本任务的 prompt hashes、`format_reward=0.02` 与 `invalid_action_penalty=-0.2`，不得修改 live checkout 或冒充 `fee3ffac...`。
- 用户已同意创建任务并批准上述 reconstruction 设计方向；这不替代新的 implementation approval、双仓库 commit/push approval或最终精确 launch approval。

## Requirements

### R1 — Checkpoint restoration and provenance

- 只读核验 source run、checkpoint shards、tokenizer/config 与恢复路径。
- 产出可由 rollout runtime 实际加载的冻结 actor artifact；记录 source checkpoint、component mapping、VAGEN commit、导出命令和 artifact hash/manifest。
- 不恢复 PPO optimizer/critic 训练，不修改源 checkpoint。

### R2 — Evidence-backed source-behavior rollout contract

- 从 step60 的真实 source config/log/runtime transcript/W&B evidence 核验 prompt、action parser、environment dynamics/reward、sampling、turn/token limits 和 split identity。
- reconstruction runtime 必须使用独立 clean VAGEN worktree/branch和可审计 patch commit；manifest 同时记录 unavailable source commit、reconstruction base/patch commits、精确 diff/hash、证据映射和已知限制。任何未被证据覆盖的语义必须在 smoke 前停止，禁止继承未核实的默认值。
- rollout 只做冻结推理，不训练任何模块。
- 正式采集前运行有界 smoke/concurrency gate；smoke 只能证明被测路径，不替代正式数据。
- 每个 shard 使用唯一输出，保留 JSONL、图像、resolved config、record counts、split/seed identity 和错误状态；不得把 partial/non-empty JSONL 当成完成 shard。

### R3 — SFT1 dataset

- 转换前验证 raw dump schema 与一条完整 transcript。
- SFT1 records 保存真实 rollout CoT 和已执行 action，并把 prompt/response 的 action-format 部分确定性转换为当前 K16 兼容格式；不得根据未执行的额外 action或terminal draft action猜测 supervision/transition。
- 每条 converted record 必须可追溯到逐字保存的 source prompt/chat transcript；conversion manifest 记录 source/converted prompt hash、转换版本和只限格式层的改写规则。
- terminal response 不作为 SFT1 的最后一个监督 assistant turn；SFT1 只监督实际执行过且有对应 environment transition 的动作。
- 输出至少区分 1,800-row train partition 的全量轨迹、其中成功轨迹和 200-row internal held-out 轨迹，并保存 conversion manifest、hash、过滤原因、前后计数及 image/action/message 对齐检查。
- internal held-out 由 batch1 的共享 seed ordinal 确定性产生；同一 seed 的两个 `eval_set` rows 必须同时 train 或同时 held-out，且 train/held-out seed overlap 必须为零。不得把它报告为未见环境分布上的 generalization split。

### R4 — SFT2 trajectory dataset

- 从同一批被验证 rollout 构造 SFT2-compatible `nimloth_trajectory_v1`：训练字段使用当前 K16 Nimloth prompt/response/action 格式，同时在独立 source-audit 字段逐字保留真实 system prompt、每个 observation、真实 assistant response 和完整聊天记录；两种视图均保留已执行 action、图像、success/reward provenance 与 split 的可核对 lineage。
- 每条 trajectory 的 terminal observation 使用同一 VAGEN step60 policy、同一获批 generation contract 额外生成完整 CoT + draft action；持久化 observation-aligned `terminal_assistant_prefix` 及可审计的 draft-action evidence，但绝不执行该 action，也不把它计作 transition。
- terminal CoT 只用于构造最后一个 CoT-conditioned state；SFT2 LLM-backbone objective 不得对它建立训练标签或梯度。
- 普通 state 与 terminal state 因此均来自同一个 VAGEN step60 policy。后续 SFT1 checkpoint 不参与本次 terminal state 生成。
- 不发明逐步 reward；本 reconstruction 固定使用 batch API 实际返回且与 archived table classes 对齐的有限 `step_rewards`。改用 `trajectory_terminal_reward` 必须返回规划阶段。
- terminal generation 的 checkpoint、sampling 参数、token/stop boundary、format failure policy 与 draft-action audit schema 必须在 launch contract 中精确列出并验证。当前设计只接受 tokenizer EOS 导致的 `finish_reason=stop`；length/other finish 或 parse failure 保留审计但使链接的 SFT1/SFT2 trajectory 整体排除，且不得执行 draft action。

### R5 — Data/split evidence and staged collection

- 记录 exact source parquet/path/version、选择规则、source row index、seed/row identity、转换 lineage、统计单位和 hash。
- batch1 内部 split 按共享 seed ordinal 固定为 1,800 train + 200 held-out，两个类别同 seed 同侧；manifest 必须证明 train/held-out 的 source row、`(eval_set, seed)` 和裸 seed overlap 均符合合同。
- 20,000 train rows 必须通过确定性 manifest 划分为 10 个互斥、并集恰为全部 source train rows 的 2,000-row batch；manifest 记录每批 source indices/keys/hash，禁止在重试时重新随机抽样。
- 本次实验只批准准备并启动 batch 1；batch 2–10 各自必须使用新的唯一 run/output identity，并在启动前重新走精确 launch approval。
- 明确训练与 held-out 数据的 overlap key，并测量 overlap；不得仅凭 `train`/`test` 名称断言无重叠。已确认 source test 与 train 在全部 128 个 `(eval_set, seed)` keys 上重叠，本次不得将其作为 held-out generalization evidence。
- 报告 conversion 前后各 split 的 record/trajectory/transition/image counts、success label prevalence 与所有排除记录。

### R6 — Outputs and safety

- 使用新的 `outputs/experiments/...` experiment group/run directory；启动前验证不存在且不覆盖任何已有 output/dataset/checkpoint。
- 远程 Nimloth 与 reconstructed VAGEN runtime 分别只使用绑定到各自已批准 commit 的独立 clean worktree；不得在服务器直接编辑生产代码或静默改变 Nimloth gitlink。
- 保留现有本地 dirty changes、受保护数据、checkpoint 和 runtime outputs。

## Acceptance Criteria

- [ ] step60 actor 已用来源兼容且可审计的方式恢复/导出，并通过 tokenizer/config/weight/load smoke。
- [ ] evidence-backed reconstruction 有独立 VAGEN base/patch commits、golden prompt/parser/reward/API tests、diff/hash manifest和明确的非-exact-source限制；Nimloth collector 验证真实 clean runtime identity而不接受 metadata relabel。
- [ ] 精确 rollout contract 与 source/reconstruction evidence、split/seed范围、资源、时间、输出、恢复和监控方式均写入任务并获得单独 launch approval。
- [ ] 20,000-row partition manifest 证明 10 个 batch 各 2,000、互不重叠且并集覆盖全部 source train rows；本次 batch 1 的全部计划 shard 达到完整性 gate，partial/failed shard 被隔离且未进入数据集。
- [ ] SFT1 输出通过 transcript/action/image/split/schema validation，并有原子 manifest、hash、counts 和 rejection sidecar。
- [ ] SFT2 trajectory 输出通过 `nimloth_trajectory_v1` validation，真实 CoT/action/reward provenance 未被合成或猜测。
- [ ] 每条有效 trajectory 的 terminal observation 已由 VAGEN step60 生成真实 CoT + draft action；只持久化 state prefix/审计证据，draft action 未执行且未形成 transition，terminal response 未进入 SFT1/SFT2 LLM-backbone supervision；manifest 满足 input = valid + excluded。
- [ ] train/held-out overlap 按约定 key 测得并记录，所有输出路径唯一且源 checkpoint/旧数据未修改。
- [ ] 终止时按实验 end contract 记录 scheduler/runtime、实际命令/commit、产物、限制与精确恢复方式。

## Out of Scope

- SFT1、SFT2、WM、Value、reconstruction 或 RL 训练。
- 修改或删除 hligb 源 checkpoint、原始 parquet、旧 rollout 或旧数据集。
- 用 smoke、静态 success prevalence 或 partial run 作为模型质量结论。
- 声称 reconstructed runtime 等同于不可读的 exact `fee3ffac...` source tree；允许的结论仅是列明证据覆盖范围的 source-behavior reconstruction。
- 在未取得 commit/push/merge 审批时执行这些 Git 操作。

## Deferred downstream decision and launch details

1. SFT1 训练集最终采用 `train_all` 还是 `train_success`；本任务将两者都产出并分别标注，后续训练选择可以延后决定。
2. 已选择 `normal` partition、单节点 4 GPU（policy TP2 + 2 environment GPUs）作为准备方向；最终 walltime、CPU、memory、实际 availability 和完整命令仍须在 launch approval 前核定。
