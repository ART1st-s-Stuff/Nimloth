# 2026-07-25 direct-policy token-level credit

## 人类要求

- 停止 SFT2，立即转向 RL。
- 阅读既有 RL task 与 VAGEN 实现，在 `dev` 直接实现 token-level credit。
- 参数不明确时停止确认，禁止猜测后启动错误实验。

## 实现边界

- trajectory 新增逐步 `rewards`、`terminated`、`truncated`；真实 terminal 从 0
  bootstrap，truncation 必须显式选择策略。当前只实现 `zero`。
- 新增独立 `TokenValueHead`，输入为 Qwen replay 中每个 selected sampled token 之前
  的 hidden state；模板和 injected token 不参与 critic 或 PPO。
- 每个 environment step 的完整 rollout Monte Carlo return 放在该 turn 最后的 action
  token；更早 reasoning token reward 为 0，在 turn 内按显式 token gamma/lambda 计算
  GAE，turn 边界 reset。
- action ValueHead 继续用真实 rollout return 监督 `Q(s,a)`；token PPO 不再使用
  selected-action Q 作为 baseline。
- token head 已接入 optimizer、双卡副本的手工 gradient sync、单卡/FSDP路径的 DDP、
  checkpoint 保存/恢复与配置 metadata 校验。

精确算法名称是“真实 environment Monte Carlo return + turn 内 token GAE”。当前没有
high-level turn GAE、`gamma_turn/lambda_turn` 或 planner root policy，因此不能称完整
VAGEN Bi-Level GAE，也不能声称 planning PPO 已完成。

## 配置门禁

`actor.credit_assignment=token` 时必须显式提供：

- `token_credit.gamma`
- `token_credit.gae_lambda`
- `token_credit.value_lr`
- `token_credit.value_loss_weight`
- `token_credit.hidden_dim`
- `rl.truncated_bootstrap`

缺失字段会在配置解析阶段失败。未得到人类逐项确认前不启动 GPU RL。

## 验证

- 本地 `compileall` 与 `git diff --check` 通过；本机 Python 没有 pytest。
- 服务器定向测试：`56 passed`。
- 扩大回归首次得到 `134 passed, 1 failed`；唯一失败是测试 fake policy 选择
  reasoning+action token 却未显式声明 `turn` credit。生产校验正确拒绝该不一致；测试
  fixture 补齐显式契约后，完整 `tests/training/rl tests/agent
  tests/backbone/qwen25vl` 回归为 `135 passed, 1 expected warning`。
- 尚未启动 GPU experiment、rollout、W&B 或 optimizer step。

## 2026-07-25 planner-distillation RL 启动门禁

- 人类已明确“可以开始 RL”，但尚未给出新路径强制要求的数值和实验规模；按此前
  “参数未明确必须停下来确认”的规定，尚未提交 Slurm、GPU、rollout 或训练任务。
- 已确认可继承：corrected SFT2 `epoch_001` lineage、`planning.horizon=2`、64 条
  exhaustive candidate、planner distribution 采样环境动作、CoT token PPO、action
  distillation、DINO loss 关闭，以及 terminal observation 使用同 checkpoint/同采样
  参数生成到 `action_start` 并持久化真实 CoT、丢弃草稿 action。
- 仍需人类明确：`agent.planning.teacher_temperature`、
  `actor.planner_distillation_weight`、`agent.planning.device`、
  `predictor.train_wm`，全部 token-credit 数值、实验 episode/iteration/step 预算、
  rollout sampling 数值、partition 和物理 GPU/TP/world-size/gpus-per-rank 布局。
- 新 vLLM selected-hidden worker extension 当前只有 compile/static 边界；启动 GPU 前还要
  在远程 vLLM 0.11 环境完成 CPU/interface regression，再做一轮真实图片 GPU correctness
  smoke。未通过 smoke 前不得直接解释长期 success rate。

### CPU/interface 门禁结果

- 首次远程定向测试进入真实用例后为 `76 passed, 3 failed`，确认并修复：planner 的
  grid state/history shape、grid predictor 缺少 `rollout_from_history`、rollout-time
  grid ValueHead 缺少 slot mean-pooling，以及 planner replay 把额外 action logit row
  混入 CoT selected rows。提交 `927cf01`。
- 新增真实 grid checkpoint loader -> state -> H=2 exhaustive planner 回归；定向测试
  `86 passed`，扩大 `tests/training/rl tests/agent tests/backbone/qwen25vl
  tests/wm/test_grid.py` 回归 `148 passed, 1 expected warning`。
- 安装版 vLLM 0.11 静态核对确认 Qwen `forward` 返回 hidden/IntermediateTensors、
  `compute_logits(hidden_states)` 接口存在，`LLM.collective_rpc` 接口匹配；同时发现
  `worker_extension_cls` 错用冒号 FQCN。修复提交 `5534da0` 后，安装版
  `resolve_obj_by_qualname` 直接解析成功，扩大回归再次 `148 passed, 1 warning`。
- CPU/interface 门禁现已通过；仍未验证真实图片、TP workers 同步 hidden/action logits
  或 GPU optimizer step，因此仍须先做 GPU correctness smoke。

## 2026-07-25 人类修订启动方向

- 本次 RL 明确使用 Slurm；此前“登录节点直接运行”只针对远程文件处理，不适用于 GPU
  RL。资源按提交前实时空闲情况凑卡，当前目标约 8 张物理 GPU，不固定 normal 分区或
  固定节点。
- 人类撤销 H=2 exhaustive 64 条候选方案，要求先使用 greedy。当前代码和 trajectory
  校验只支持 exhaustive；greedy 是全局单路径，还是为每个 root action 保留一条 greedy
  continuation，仍会改变 Qwen 蒸馏目标与真实 behavior distribution，必须确认后实现，
  禁止把旧 exhaustive 配置直接改名启动。
- action ValueHead 的 environment return 使用 `rl.gamma=1.0`。
- Qwen 训练参数应对齐真实 VAGEN 源 run，而不是按 Nimloth 默认值猜测。已从服务器源
  run 的 resolved config 核实 actor lr/clip/entropy、KL、采样、optimizer 和 batch 参数；
  当前 Nimloth 尚无固定 reference KL，且 TokenValueHead/actor 的参数与梯度职责不等同于
  VAGEN 独立 critic，因此仍需明确“对齐”的范围并补齐所需实现后才能启动。

## 2026-07-25 H=2 greedy/reference-KL 实现结果

- 已确认且写入`configs/training/rl/planner_greedy_h2_smoke.yaml`：H=2逐深度greedy只保留
  1条候选；distillation weight=1；WM训练；Qwen actor lr=1e-6、AdamW decay=0.01、
  clip=0.2、entropy=0.01、response cap=512、temperature=0.7、top-p=0.95；reference
  low-var KL=0.001；environment/token gamma=1；DINO/SIGReg/ranking关闭。
- planner trace保存H行完整action-value，验证每一步确为argmax；behavior与teacher分字段
  保存且均为首动作确定性分布。planner action从token PPO mask和reference KL中排除；
  Qwen action head只通过交叉熵拟合greedy teacher。
- rollout后新增冻结reference独立重放阶段，在训练模型加载前为每个selected CoT token
  写入full-vocabulary log-prob；fresh manifest升级为v3并绑定reference artifact指纹。
  actor loss使用VAGEN实际执行的`low_var_kl`，没有实现reward KL。
- TokenValueHead接收detached Qwen hidden，critic MSE只训练token head；PPO、action
  distillation和reference KL仍可训练Qwen。CSV/W&B新增token value、distillation和KL指标。
- commits `5e141bb`、`49bbf0a`、`8c771db`已推送`dev`。服务器扩大CPU/interface回归
  `157 passed, 1 expected warning`；尚无GPU实验或optimizer-step证据。
