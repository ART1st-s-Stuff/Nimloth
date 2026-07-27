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
- ID49 preprocess cache 可精确续建：train image shards 为 32/489；没有 optimizer、
  checkpoint 或 W&B 训练状态。本次只复用其真实 terminal-CoT 数据、DINO sidecar 和
  processor-fingerprinted 预处理 cache，不 resume 旧 SFT2 optimizer。

## 待完成

- 提交并同步当前精确代码版本，在独立远端 worktree 做 preflight。
- 先用隔离的 8-record 真实数据 prefix cache 运行单卡和 2 节点 × 4 GPU smoke；通过后
  再原子续建 ID49 全量 preprocessing cache。smoke cache 与正式 cache 不共用写路径。
- smoke 通过后确定未占用 W&B ID、正式输出目录和实测耗时，启动 world-size 8 正式
  SFT2，并监控至少首个 optimizer step 和首个可恢复 checkpoint。
