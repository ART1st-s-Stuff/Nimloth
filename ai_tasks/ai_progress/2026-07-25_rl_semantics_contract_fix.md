# 2026-07-25 RL semantics contract fixes

## 目标

修复近期 RL 审查确认的概率重放、fresh artifact、一次消费、验证器、KL 数值和
planner 多候选边界；保持当前算法的准确名称为“environment Monte Carlo return +
turn 内 token GAE”，不冒充 VAGEN Bi-Level GAE。

## 已确认语义

- vLLM behavior 与 HF PPO replay 必须使用相同 reasoning token support、temperature
  和 top-p 变换；同 checkpoint 回放 ratio 应为 1。
- planner 最终必须能模拟不同 root action；单候选 greedy 只能是一个显式 search mode，
  不能成为公共数据结构和配置的硬限制。
- 当前 WM 没有 reward/done head，多候选搜索使用明确命名的 leaf action-value heuristic，
  不逐层累加 Q-value。
- planner action 不进入 Qwen PPO/reference KL；Qwen action head由 planner root teacher
  蒸馏。

## 计划

1. 先补失败型单元/契约测试。
2. 统一 behavior probability contract。
3. 给 behavior/reference trajectory 增加内容指纹和事务化消费状态。
4. 统一 launcher resolved config，补全 validator 与稳定 KL。
5. 实现 exhaustive/beam 可扩展多候选 planner 和 root-action 反转测试。
6. 清理过度 cache 兜底、空 validation wrapper和过期文档。
7. 运行可用的静态、CPU、集成测试并记录验证边界。

## 当前状态

- 工作目录：`/workspace/remote2/nimloth-dev`（人类明确要求直接修改）。
- 起点：`104639407d72545a06899d66c1fe9d4e13cf484b`
- 已统一vLLM/HF reasoning forbidden-token support，并增加same-policy ratio=1回归断言。
- fresh manifest升级为v4，绑定behavior/enriched trajectory字节；消费状态改为
  `in_progress -> committed`，只有post-update `latest` checkpoint完成后才提交。
- planner支持`greedy/exhaustive/beam`；当前H=2 smoke配置改为`exhaustive`，保存候选、
  leaf score和root聚合分数，并补了未来分支反转root greedy动作的测试。
- launcher从RL YAML解析episode数、最大环境步数、temperature/top-p及planner参数；
  validator覆盖全部actor/token/planner/reference指标、组件checkpoint和消费提交状态。
- reference `low_var_kl`在最终clamp已饱和区间内预先约束exp输入，避免极端log-ratio溢出。
- vLLM缓存开关改为显式参数；默认仍关闭，等待同版本真实多图A/B parity验证后再启用。
- 删除无操作的online-policy validation wrapper，并同步代码/配置/README/方案文档。
- 未启动训练或GPU实验。

## 验证记录

- `git diff --check`：通过。
- 全部`src/**/*.py`、`tests/**/*.py`及`rollout_env.py`使用`ast.parse`：通过。
- `bash -n experiments/training/rl/run_vllm_online_ppo_smoke.sh`：通过。
- 使用本机Nix store中的PyTorch 2.12/pytest 9/PyYAML 6运行9个直接受影响测试文件：
  `78 passed, 1 expected warning`。
- 增加3条训练循环fault-injection测试：step前失败回滚、step开始后失败保留claim、
  成功路径先写post-update checkpoint再commit，定向`3 passed`。
- 扩大到`tests/agent`、`tests/backbone/qwen25vl`、`tests/rollout`和
  `tests/training/rl`：排除本机缺少vLLM而无法收集的`test_vllm_logits.py`后，
  `169 passed, 1 expected warning`。vLLM policy stub测试包含在通过范围内。
- `planner_exhaustive_h2_smoke.yaml`经真实`load_rl_config`解析并核对horizon、search、
  episode/max-step和temperature/top-p：`CONFIG_OK`。
- 尚未验证真实vLLM、真实图片、同checkpoint跨vLLM/HF ratio或GPU optimizer step；
  这些必须作为独立运行门槛，不能由CPU测试代替。
