# FSDP dynamic rollout

## 目标

为现有 Nimloth RL FSDP trainer 实现真正的在线动态 rollout：每个 RL iteration 使用当前更新后的 k=1/inject policy 访问 VAGEN train-split 环境，随后执行 PPO + WM/value update；参考 VAGEN actor-rollout/env-service 循环，但保留 Nimloth 的 latent、WM/value loss 和 checkpoint ownership。

## 约束

- 仅 rank0 访问外部 VAGEN/AI2-THOR HTTP env service；所有 FSDP rank 必须以相同顺序和形状参与 policy/encoding/PPO forward。
- rank0 采样并广播动作；各 rank 计算同一 action distribution，并校验一致性。
- 所有 rank 获得完全相同的完整 trajectory；只允许 rank0 写 JSONL/PNG，禁止文件写入竞争。
- env、policy 或 schema 错误必须同步失败或丢弃完整 episode，不得以默认动作/零 log-prob 冒充有效 rollout。
- train rollout 只允许实际 `*_train` split；当前 runtime 仅支持 k=1 inject。
- resume 后 rollout seed 不得从0重复；checkpoint world size gate保持不变。

## 当前计划

1. 将 action distribution 计算与 rank0 sampling 解耦。
2. 新增 distributed env collector，同步 rank0 env control、all-rank FSDP forward、rank0 action broadcast、trajectory broadcast。
3. trainer 允许 world>1 env collector，并在每轮 rollout 时保持 inference mode、设置可恢复 seed offset。
4. 增加单测及2-rank CPU/gloo同步集成测试；再做服务器2-GPU短 smoke。
5. 通过后再为 dgx-09 env + dgx-32 FSDP 正式在线 RL 请求昂贵实验确认。

## 已完成

- 创建分支/worktree：`feat/fsdp-dynamic-rollout` / `../nimloth-feat-fsdp-dynamic-rollout`，起点 dev `25f4237`。
- 核实 VAGEN：每个 global step动态 reset/step env，当前 actor产生trajectory，随后 update_actor；下一个step生成前同步最新FSDP权重。
- 新增`DistributedEnvRolloutCollector`：仅rank0持有HTTP env与写文件；所有rank同序policy forward；all-reduce检查8-action logits；rank0按确定性seed采样并广播action；rank0 step env并广播结果；最终完整trajectory广播。
- trainer已移除world>1 env guard，改为FSDP wrap后接线distributed collector；rollout与latent encoding使用临时eval mode，PPO继续带梯度。
- rollout与PPO改为同一canonical k1/inject prompt，使用真实历史图片和可配`history_window`；old/new log-prob均为temperature-scaled完整8-action distribution，top-p只约束采样。
- 删除policy错误时`moveahead + [0]*8` fallback；环境、policy、schema失败不能进入训练。新增finite/schema validator。
- checkpoint记录`rollout_protocol`；resume强制核对mode/split/eval sets/history window/temperature/top-p/seed offset，并根据已完成iteration恢复env seed cursor。
- 动态训练要求显式`*_train`且暂时`validation.enabled=false`，避免把train collector结果标成heldout validation。

## 修改

- `src/nimloth/training/rl/distributed_rollout.py`
- `src/nimloth/training/rl/{rollout,trainer,cli,checkpoint}.py`
- `configs/training/rl/defaults.yaml`
- `tests/training/rl/test_dynamic_rollout.py`
- RL README文档。
- 提交：`3f87a5c`、`a19ee8f`，已推送`origin/feat/fsdp-dynamic-rollout`。

## 验证

- 本地`compileall`与`git diff --check`通过。
- 服务器提交`a19ee8f`：RL/latent tests `29 passed, 1 expected warning`；后续定向回归`24 passed, 1 expected warning`。
- 2-rank gloo integration覆盖rank0-only fake env、all-rank action distribution collective、rank0 action broadcast、相同trajectory与rank0-only JSONL。

## 待确认/风险

- 尚未用真实Qwen FSDP + VAGEN env做2-GPU动态online smoke；CPU/gloo测试不能证明NCCL/FSDP模型forward不会遇到运行时问题。
- 外部环境服务超时期间其他rank会等待rank0广播；HTTP timeout为600秒，失败后同步丢弃完整episode或终止collective policy path。
- 当前逐episode、逐action forward保证语义但未做VAGEN式active-env batching，吞吐可能较低；需真实smoke后再优化。
- 服务器submodule Python cache已清理，launch worktree固定clean commit `1e93a74148eee9ca248c528de89c1686871097fc`。

## 真实NCCL动态smoke

- 人类允许先用dgx-51/dgx-52测试。新增config与两节点orchestrator：dgx-52 trainer2GPU NCCL/FSDP，allocation启动后自动向dgx-51提交1GPU VAGEN/AI2-THOR env child；HTTP timeout降到180秒并写入resume protocol，低于默认NCCL watchdog。
- W&B project=`nimloth-rl`，ID3，run=`3_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2`，internal=`66bsq5lp`已实际预留并持久化。
- 初始化k1/inject SFT2 epoch2；`base_train` seeds20001..20002；2 episodes×1 action；1 update/batch2；language full+WM/value train，vision/state projector freeze；只写final full checkpoint，不做效果claim。
- output=`outputs/experiments/training/rl/2026-07-16/3_smoke_fsdpdynamic_k1ep2_base2x1_ws2_iter1_b2`，README已记录commit/data/modules/checkpoint/resources/gates。
- trainer job `477191`提交dgx-52后因Priority pending。人类批准改用dgx-32；agent在同一critical section确认job仍为PENDING/elapsed0/no allocation后取消，未触发E0026竞态。
- dgx-32 retry trainer `477199`运行00:01:20后FAILED `1:0`，发生在worker/model初始化前：torchrun默认TCP port29500被共享节点其他进程占用，报`DistNetworkError/EADDRINUSE`。env477200 pending取消；8CPU replacement env477201在dgx-51通过health并于trainer失败后23s clean COMPLETED。
- 本次没有trajectory/CSV/model load/update/checkpoint，未验证NCCL/FSDP；W&B `66bsq5lp`仍只有queue step0。输出日志保留。已登记`E0027_torchrun_default_master_port_collision.md`并以`--standalone`修复。
- 人类批准retry2。dgx-32 trainer477204 acquired2GPU，但自动env477205被`MaxGRESPerAccount`阻塞：共享`csejzhang`身份的其他活跃任务已占剩余account GPU quota；未触碰这些任务。trainer仍只等待env URL，torchrun/model未启动。即时复核trainer=RUNNING/env=PENDING后取消，elapsed49s/0，无artifact，W&B仍step0。
- 待人类决策：等待account quota释放后保持dgx-32+dgx-51，或批准在dgx-32同一allocation内让env与FSDP rank0共享GPU。后者仍可验证NCCL/FSDP动态rollout，但不验证跨节点HTTP拓扑。
