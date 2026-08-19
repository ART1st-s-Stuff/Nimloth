# E0129 — Navigation head must be runtime-qualified

## 现象

ID185 retry1 Job `522968`的静态head selector选择`dgx-18`。exact4×2、Ray/fabric/import、source/hash、dataset/checkpoint及SIGReg门禁均通过，但FloorPlan1 direct-render在固定150秒内零输出并exit124。`dgx-18`曾在ID184 retry1用同一门禁约3.8秒通过，因此历史allowlist或单次成功不能证明当前allocation中的Navigation-head健康。

## 影响

静态选择“第一个未列入坏节点的节点”会在Ray建立后才发现当前head无法渲染，浪费allocation并阻止评估开始。不能把一次瞬态失败直接升级为永久坏节点，也不能延长timeout掩盖故障。

## 规则

- Navigation/Ray head必须在当前allocation内用真实FloorPlan1 direct-render和固定150秒上限动态认证后再分配角色。
- candidate失败后可在同一allocation尝试下一个eligible节点；全部失败时fail closed。
- 历史坏节点仍不参与candidate；历史通过只能作为候选资格，不能替代本次认证。
- 正式runner仍保留150秒direct-render门禁；prewarm仍最多300秒。
- 失败attempt不可覆盖，retry必须使用新output和W&B identity。

## 证据

- Slurm Job `522968`: `FAILED 124:0`, elapsed `00:04:06`, nodes `dgx-[18,27,30,39]`。
- Server output README: `outputs/experiments/training/rl/2026-08-18/185_eval_k4schemeb_dp8_tp8_source20_test5x60_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_retry1/README.md`。
- Empty direct-render artifact: sibling `full_eval_test300/render_probe.json`。
- Prior contrary evidence: ID184 retry1 Job `521910` used head `dgx-18` and passed direct-render in `3.794s`。
