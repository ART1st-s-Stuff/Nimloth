# 实施计划 — 独立Nimloth work-item可见性

## 执行规则

- 最终规划批准后可在批准范围内持续执行；设计歧义、越界、破坏性操作及commit/merge门禁时停止。
- Nimloth producer先完成并固定projection contract，pi-app consumer随后实施；每个worktree只有一个writer。
- 不修改上游Trellis/Pi core，不读取或迁移旧`.trellis/.runtime/execution`。

## W1 — 双仓库隔离与baseline

- [x] 从Nimloth parent `3caab1f`和pi-app parent `15618da`创建同名task branches/worktrees。
- [x] 核验root/branch/common-dir/clean status和Nimloth `.local` symlink。
- [x] 运行upstream fidelity、Pi extension和pi-app stable panel focused baseline。

## W2 — RED：透明parser与identity

- [x] 添加普通Markdown heading/checkbox parser RED。
- [x] 覆盖normalized identity、顺序、checked状态、duplicate ambiguity、文本/heading变化失效和malformed输入。
- [x] 证明不要求或写回`[W-xxx]`。

## W3 — GREEN：core runtime与transition

- [x] 实现`core.mjs` parser、SHA-256 identity和plan fingerprint。
- [x] 先补runtime RED：root/context/task/plan/item校验、transition、blocked要求、evidence bounds、corrupt文件、atomic write。
- [x] 实现`.local/nimloth-work-items/v1/`schema和select/update/evidence/release；不持久化plan文本、checkbox、tool payload或CoT。

## W4 — RED/GREEN：Pi动态tool

- [x] 用Pi extension harness固定复杂in_progress activation matrix及其他状态absence。
- [x] 实现`index.ts`注册工具并通过`getActiveTools/setActiveTools`只增删自身；不添加promptSnippet/guidelines或auto-discovered skill，简短操作指导只在动态tool description/result中。
- [x] 覆盖session/task切换、plan issue、completed状态、skill metadata absence及其他active tools保留。

## W5 — RED/GREEN：只读projection

- [x] 固定versioned projection schema和missing/stale/orphan/checked conflict、planning/completed inactive suppression tests。
- [x] 实现`projection.mjs`即时join task/plan/runtime，按consumer同值限制item/heading/count/issues/task/status/runtime字段，stdout bounded且错误fail-visible。
- [x] 运行upstream fidelity和Nimloth extension tests，确认无template-managed差异。

## W6 — RED/GREEN：pi-app projection reader与IPC

- [x] 从pi-app parent创建consumer RED：registered source、timeout、overflow、stderr、schema/root/context mismatch和WSL path。
- [x] 实现只读Main reader/strict IPC；Renderer只能提交registered worktree identity，不能提交CLI/root路径。
- [x] Projection unavailable只返回可见issue，不影响task browser。

## W7 — RED/GREEN：稳定Activity consumer

- [x] 固定source selector、current item/executor/state/elapsed/blocker/next/evidence及降级UI tests。
- [x] 在稳定`TrellisSidePanel`下新增独立Activity consumer；不import旧dashboard/approval/TaskTree types。
- [x] 覆盖worktree/session快速切换旧响应隔离和Documents视图保持可用。

## W8 — 最终检查与复审

- [x] Nimloth运行parser/runtime/activation/projection、upstream fidelity、task validation和diff check。
- [x] pi-app运行affected tests、full unit、Node/Web typecheck、lint、build和CRLF-aware diff check；clean-base例外独立归因。
- [x] 独立review检查权威、动态absence、runtime隐私/bounds、projection join、cross-worktree source和旧dashboard删除兼容性。
- [ ] 分仓库展示完整diff与精确commit范围；未经批准不stage/commit/merge。

## Guardrails

- 不修改`.trellis/workflow.md`、`.trellis/scripts/`、上游`.pi/extensions/trellis/`、generated files、Pi core或node_modules。
- 不创建approval receipt、TaskTree状态或task lifecycle mutation。
- 不写回implement checkbox/ID，不持久化完整plan、CoT、assistant response或tool输出。
- 不启动实验，不操作remote/Slurm，不删除legacy/live runtime。
- 不push、不合入dev/default branch、不cleanup worktree，除非分别到达明确门禁。
