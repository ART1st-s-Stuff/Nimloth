# E0128 — dgx-10 must not be a Navigation head

## 现象

ID185 Job `522498`在`dgx-[10,18,21,27]`通过exact4×2 H800、四节点Ray/fabric/import、source step20 hash、完整300-row dataset manifest、actor/planning checkpoint hash及SIGReg CUDA门禁后，`dgx-10`上的FloorPlan1 direct-render在固定150秒内零输出并exit124。

## 影响

失败发生在environment server、TaskRunner、test300 validation、trainer checkpoint restore、optimizer和W&B启动前；没有评估结果、W&B run或新checkpoint。延长render timeout会掩盖节点/head eligibility故障。

## 规则

- `dgx-10`不得承担Navigation/Ray head角色，直到经过单独重新认证；仍可作为allocation中的非head worker。
- 选择Navigation head时必须与`dgx-23`、`dgx-37`一并排除`dgx-10`。
- 继续保持direct-render 150秒和prewarm 300秒硬门禁。
- 失败output必须保留，retry使用新output suffix和W&B identity。

## 证据

- Slurm Job `522498`: `FAILED 124:0`, elapsed `00:06:25`, nodes `dgx-[10,18,21,27]`。
- Server output README: `outputs/experiments/training/rl/2026-08-18/185_eval_k4schemeb_dp8_tp8_source20_test5x60_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095/README.md`。
- Empty direct-render artifact: sibling `full_eval_test300/render_probe.json`。
