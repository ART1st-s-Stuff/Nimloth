# 需求 — 恢复Trellis基线并重建Nimloth政策层

## 目标

把Nimloth中的通用Trellis文件恢复为发布版`0.6.16`，并用薄`AGENTS.md`、scoped specs和project-local skills承载已经确认的项目规则，不维护Trellis fork。

## 需求

1. 发布版Trellis拥有workflow、template-managed scripts、generated Pi prompts/agents、Trellis Pi extension和bundled skills；这些文件不得包含Nimloth政策。
2. `AGENTS.md`只保留权威顺序、诚实/不确定性、替代机制、安全边界、Git不变量和不可遗漏的高风险skill路由。
3. 实现合同进入`.trellis/spec/`；操作步骤进入project-local skills/references；普通skill由Pi原生description匹配。
4. 所有真实实验必须是`meta.kind=experiment`且remote-only。短实验为预计十分钟内的复现/部分参数重跑，从成功提交起设置十五分钟总deadline，覆盖pending与running。
5. 每个task使用独立branch。单一主task在canonical directory实施；额外并行task使用worktree。开发branch只作为task branch base。
6. 恢复上游context injection并测量干净baseline；本child不实施新的context压缩。
7. 从上游所有的Trellis scripts/extension中移除typed approval、dashboard、execution和work-item能力；独立work-item extension由后续child实现。

## 验收标准

- [ ] `trellis update --dry-run`证明通用文件与`0.6.16`一致，project-owned例外有明确allowlist。
- [ ] 上游一致性tests能发现任一通用文件的Nimloth分叉。
- [ ] `AGENTS.md`不包含操作命令或大段实现规范，且高风险skill路由完整。
- [ ] Spec/skill合同覆盖实验、Git/worktree、protected/destructive、validation、progress和memory边界。
- [ ] Main/subagent context baseline有可复查的原始测量结果，不包含本child新增压缩机制。
- [ ] 受影响focused tests和`git diff --check`通过；spec diff在commit前单独展示审查。

## 排除项

- 不修改pi-app。
- 不实现独立work-item extension。
- 不迁移156个known errors、`AI_branch_progress.md`或`ai_tasks/`。
- 不删除旧live approval runtime。
- 不启动实验、远程job、push、force或protected branch mutation。
