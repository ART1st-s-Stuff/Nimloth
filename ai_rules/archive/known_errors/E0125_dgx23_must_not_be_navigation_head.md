# E0125 — dgx-23 must not own the Navigation render service

## 已发生的错误

在人类临时允许使用 `dgx-23` GPU 后，ID183 Job `521033`按Slurm节点顺序把
`dgx-23`选为Ray/environment head。固定150秒FloorPlan1 direct-render门禁再次无输出
并超时；这复现了ID172 Job `519634`在同节点的独立失败。

## 原因

“允许节点参与模型计算”不等于“允许该节点承担AI2-THOR渲染”。launcher把排序后的
首节点同时用作Ray head和Navigation服务节点，没有单独约束head角色。

## 正确做法

- `dgx-23`可以按人类当前指令临时作为Ray模型worker，但不得作为Navigation head。
- 多节点launcher必须从实际allocation中显式选择已允许的render head，再将其放在
  cluster node/IP identity首位；若没有合格head则fail closed。
- 保持150秒direct-render和300秒prewarm上限，禁止用放宽timeout掩盖节点问题。

## 证据

- `AI_branch_progress.md`中ID172 Job `519634`和ID183 Job `521033`记录。
- Job `521033`输出目录`README.md`与`phase1_train_to_5/render_probe.json`（0 bytes）。
