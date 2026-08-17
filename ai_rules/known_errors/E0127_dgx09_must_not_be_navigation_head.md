# E0127 — dgx-09 must not be a Navigation head

## 现象

ID184 Job `521851`在`dgx-[09,18,30,37]`通过exact4×2 H800、10.23 fabric、四Ray节点、remote import、source checkpoint hash和SIGReg CUDA门禁后，`dgx-09`上的FloorPlan1 direct-render在固定150秒内零输出并exit124。

## 影响

失败发生在environment server、TaskRunner、validation、rollout、optimizer和W&B启动前；没有ID184 checkpoint或训练结果。延长render timeout会掩盖节点故障。

## 规则

- `dgx-09`不得承担Navigation/Ray head角色，直到经过单独重新认证。
- ID184 retry优先从allocation排除`dgx-09`；至少必须加入Navigation head exclusions。
- 继续保持direct-render 150秒和prewarm 300秒硬门禁。
- 失败output必须保留，retry使用新output和W&B identity。

## 证据

- Slurm Job `521851`: `FAILED 124:0`, elapsed `00:04:41`, nodes `dgx-[09,18,30,37]`。
- Server output README: `outputs/experiments/training/rl/2026-08-17/184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8/README.md`。
- Empty direct-render artifact: sibling `continue_step10_to20/render_probe.json`。
