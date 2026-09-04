# 实施计划 — 恢复Trellis基线并重建Nimloth政策层

## 隔离与RED

- [x] 从parent integration branch创建`task/rebuild-trellis-baseline-policy`及对应Nimloth worktree，核验root/branch/common git dir/`.local`和clean base。
- [x] 固定`0.6.16` package来源、parent baseline manifests和project-owned allowlist fixture。
- [x] 添加上游文件一致性RED，证明当前workflow、`task.py`或Trellis Pi extension存在分叉。
- [x] 添加薄`AGENTS.md`和规则owner RED，检测操作命令重复、缺失高风险skill路由及无owner合同。
- [x] 添加per-task branch、canonical主槽位、remote-only experiment和十五分钟总deadline静态合同RED。
- [x] 添加main/implement/check context bytes与重复读取测量fixture。

## 恢复发布版Trellis

- [x] 运行受控`trellis update`升级project metadata和自动更新文件；逐项处理modified templates，禁止repository-wide force。
- [x] 将`.trellis/workflow.md`、template-managed scripts、`.pi/extensions/trellis/index.ts`、generated Pi prompts/agents及bundled skills恢复为`0.6.16`。
- [x] 审计并移除上游入口中的typed approval、dashboard、execution和work-item引用；仅在零引用后删除额外custom modules/tests。
- [x] 重新运行一致性tests和`trellis update --dry-run`，记录上游一致项与project-owned例外。

## 建立Nimloth覆盖层

- [x] 重写薄`AGENTS.md`，只保留六类安全内核与不可遗漏的高风险skill路由。
- [x] 重组governance/experiments/python specs，使每条已批准合同有唯一owner且无相互冲突。
- [x] 精简或补充project-local skills/references，承载experiment、Slurm、worktree、progress和memory操作步骤。
- [x] 更新相关index/link和静态tests；不修改known-error内容或legacy progress payload。

## Context测量与检查

- [x] 在恢复后的上游context injection上运行固定main/implement/check fixture，保存原始bytes和重复读取结果，不实施压缩。
- [x] 运行全部受影响focused static/CPU tests、`task.py validate`和`git diff --check`。
- [x] 展示完整spec diff、理由和影响；人类拒绝时只回退spec修改。
- [x] 展示完整child diff、验证证据、删除清单和commit范围；未经批准不commit或merge。

## Guardrails

- 不修改pi-app、全局Trellis package、node_modules、template hash文件或runtime session pointer。
- 不启动实验/远程job，不迁移legacy archives，不删除live approval runtime。
- 不覆盖canonical dirty changes，不使用force/reset/clean，不提交其他task文件。
