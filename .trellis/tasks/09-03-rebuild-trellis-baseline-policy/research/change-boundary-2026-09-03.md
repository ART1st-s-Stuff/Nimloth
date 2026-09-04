# 修改边界（2026-09-03）

## 最小行为差距

当前Nimloth把项目审批、work-item和prompt压缩逻辑写入上游Trellis拥有的workflow、scripts和Pi extension；项目规则又重复分散在`AGENTS.md`、spec和skills。目标是恢复发布版`0.6.16`通用行为，并把已经批准的Nimloth合同放回项目拥有的层。

## 行为实际所有者

- 上游生成/升级语义：已安装`@mindfoldhq/trellis@0.6.16` templates和`trellis update`。
- 不可覆盖的项目安全入口：`AGENTS.md`。
- 稳定项目合同：`.trellis/spec/`。
- 操作步骤：project-local `.agents/skills/`。
- Context injection：发布版`.pi/extensions/trellis/index.ts`；本child只恢复和测量。

## 预计修改

- 恢复`.trellis/workflow.md`、template-managed `.trellis/scripts/`、`.pi/extensions/trellis/index.ts`、generated Pi prompts/agents和bundled skills，因为这些路径由发布版Trellis拥有。
- 删除确认零引用的custom approval/dashboard/execution/work-item modules及tests，因为这些能力不属于上游入口。
- 重写`AGENTS.md`，使其只保存安全内核和高风险skill路由。
- 修改governance/experiments/python specs及project-local skills/references，使每条批准合同有唯一owner。
- 添加只读contract/context measurement tests和research artifact，证明恢复与prompt基线。

## 明确不做

- 不修改pi-app或实现独立work-item extension。
- 不归档known errors、旧progress、`ai_tasks`或live approval runtime。
- 不启动实验、远程job或GPU工作。
- 不继续设计/实现context压缩。

## 不改变行为的证明

对于恢复路径，以发布版package内容、`trellis update --dry-run`和上游一致性hash检查证明；对于规则迁移，以静态owner tests、链接检查和完整diff审查证明合同未丢失或放宽。任何无法唯一归属的规则保留原文并返回设计审查。
