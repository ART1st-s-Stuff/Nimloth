# SFT2 H=1 / T=4 DINO-grid 重训

## 目标

- 使用单个当前 latent state（`H=1`）作为 WM 输入。
- 严格采用原始 rollout 中连续四个 action，自回归预测四个后续 latent state（`T=4`）。
- 对四个预测状态分别计算 WM latent、DINO-grid 和真实 Monte Carlo value loss。
- ValueHead 只监督各步实际执行的 action，不使用未执行 action 的 rank loss；`gamma=1`。
- 其余训练、冻结、数据和 DINO 监督配置保持当前 DINO-grid SFT2 合同。

## 当前计划

1. 核对 SFT2 sampler、batch/cache、WM rollout、DINO target 和 checkpoint 接口。
2. 增加独立的 `prediction_horizon=4`，实现固定连续四步训练窗口。
3. 补充 action/target 对齐、递归预测、四步梯度和配置回归测试。
4. 完成本地测试后执行远端单卡与正式拓扑 smoke。
5. smoke 通过后启动正式训练并监控首个 optimizer step/checkpoint。

## 已确认合同

- `history_size` 就是输入历史长度 `H`；本次固定为 1。
- `prediction_horizon` 是训练展开长度 `T`；本次固定为 4。
- 四步 action 必须来自同一条原始 rollout 的连续 transition。
- 四步 WM/DINO/value target 都与各自 transition 对齐；禁止 padding 或伪造未来 action/state。
- CE 与 SIGReg 保持当前 SFT2 配置语义。

## 当前状态

- 当前分支：`dev`，起始提交 `837d014e`。
- 工作区开始时仅有用户已有的 `external/le-wm` 和 `readme.md` 未跟踪内容。
- 已实现独立 `prediction_horizon`：`H=1,T=4` 使用同一 rollout 内的固定长度滑窗，
  四个 action、MC return、下一 observation 和 DINO target 均逐位置对齐；不跨 record、
  不跨缺失 step，也不补伪造 transition。
- WM 从真实 `s_t` 开始递归四步，后续输入只使用前一步预测；真实未来 state 只通过
  `no_grad` target encoder 进入 loss。ValueHead 在四个预测 state 上只 gather 原始
  rollout 实际 action 对应的 slot，rank loss 保持删除。
- 新增正式配置 `configs/training/sft2/dino_grid_k16_h1_t4.yaml`：`gamma=1`、两 epoch、
  DINO 权重和其余训练/冻结设置沿用当前 DINO-grid 配置。
- 本地回归：SFT2/WM/Agent 为 `127 passed, 1 skipped`；受共享 WM 接口影响的 RL/grid
  回归另为 `29 passed`。`compileall`、两个 shell 脚本 `bash -n`、`git diff --check`
  均通过。两条 Gloo 测试必须在允许 loopback socket 的环境执行；沙箱内的
  `Operation not permitted` 不属于代码失败。
- 重连后资源查询成功：normal 分区空闲 20/48 GPU，preempt 分区空闲 11/32 GPU；
  当前用户无运行中 Slurm 任务，尚未占用 GPU 或启动训练。
- 代码已提交并推送为 `c1e983ac` 与 `e664c49f`；clean server worktree 固定在
  `e664c49f9cb291de94697ef2c5d0a7ffd04b0280`。远端 exact-environment 定向回归为
  `55 passed in 68.55s`，并通过 compileall、shell syntax 和 diff-check。
- ID50 的 8-train/8-val fresh CPU cache smoke job `494514` 在读取第一条记录时正确失败：
  ID49 已审计 terminal-CoT JSONL 仍是没有 `record_format` 的 legacy trajectory，而当前
  reader 只接受 `nimloth_trajectory_v1`。任务最终状态 `FAILED 1:0`、耗时 `00:01:12`，
  没有 GPU/W&B/optimizer/checkpoint/cache manifest/done marker。
- 先前关于续建 ID49 partial cache（train image shards 32/489）的结论失效。迁移后的
  JSONL fingerprint 必然改变，因此旧 partial cache 不能作为本次训练的 resume source。
  strict reader 与 fingerprint gate 不会放宽。
- 修复提交 `03dd18fc` 已推送；新 clean server worktree 的迁移/H1-T4 定向回归为
  `64 passed in 13.12s`，compileall、shell syntax、diff-check 和 tracked clean gate 通过。
- ID52 CPU migration Slurm `494521` 已 `COMPLETED 0:0`（37秒，`intel-01`，peak RSS
  `512304K`）。train/val逐条验证分别覆盖3211/355 records与59269/6054 transitions；
  IDs唯一，原始rollout action和terminal CoT全部不变，manifest与source/output hash一致。
  migrated JSONL SHA256为train `d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97`、
  val `4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e`。
- ID50 migrated-data fresh cache Slurm `494524` 已 `COMPLETED 0:0`（32秒，`intel-02`）。
  train/val为138/99 transitions、146/107 unique images，fingerprint分别为
  `deacbccea6eec498`/`66186fbb54ef56bf`；格式、BF16、gamma1、terminal-CoT expansion、
  manifests与done flag完整。生产reader全量加载train 114/114、val 75/75个H=1/T=4
  窗口，确认每窗同rollout连续4步、recorded action对齐及1 current+4 next encoding。
- ID50单卡hold `494528`在`dgx-09`启动，W&B `9hcisto1`使用正确run name。17个真实窗口
  已完成step1--15且total/WM/DINO/value-MC/CE全部finite，`step_000015`为完整可训练
  checkpoint；但train step `494528.1`随后`FAILED 1:0`。最后一个T4窗口的terminal-CoT
  next encoding正确没有labels，与前三个含labels next row一起进入include-labels-false
  collate时触发`KeyError: 'labels'`。hold已取消、GPU已释放；没有epoch val/final/done，
  单卡smoke未通过。修复和terminal回归通过后可从step15恢复同一ID50/W&B run。
- 修复提交`4f66b8d3`在无label target路径collate前逐row删除labels，并令CE路径强制所有row
  有labels。远端相关回归`58 passed in 12.52s`；真实ID50最后窗口steps16--19、recorded
  actions`[1,3,1,3]`的label presence为`[true,true,true,false]`，修复后的生产assembler
  已成功输出4-row label-free next batch。代码gate通过，训练smoke仍待从step15恢复完成。
- 首次step15 resume使用hold`494533`/`dgx-21`；W&B `9hcisto1`正确resume，但在任何新
  microbatch前，SFT2 checkpoint loader因调用未导入的`ValueHead`触发`NameError`。
  step `494533.1`在1分02秒`FAILED 1:0`，hold已取消，step15未改变。需增加完整WM-owned
  modules save-load回归与真实step15 loader gate后再恢复。
- resume loader修复提交`63082ac3`已增加`ValueHead`导入和projector/predictor/ValueHead
  完整save-load回归；远端相关CPU回归`76 passed in 14.11s`。对实际ID50
  `step_000015`执行生产`load_world_model_checkpoint()`成功：H=1、K=16、D=1024，三个模块
  权重均finite，checkpoint仍为epoch1/micro-step15未完成。第二次resume的代码与真实
  checkpoint门禁均已通过。

## 待完成

- 修复mixed terminal-next label collation并通过回归，从step15恢复完成单卡smoke；随后
  运行2节点×4GPU smoke；通过后从
  同一 migrated JSONL fresh 构建 ID52 全量 preprocessing cache。smoke cache 与正式
  cache 不共用写路径。
- smoke 通过后确定未占用 W&B ID、正式输出目录和实测耗时，启动 world-size 8 正式
  SFT2，并监控至少首个 optimizer step 和首个可恢复 checkpoint。
