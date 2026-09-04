# 需求 — 提取独立Nimloth work-item可见性

## Goal

在不修改上游Trellis 0.6.16、不要求显式`[W-xxx]`、不创建第二task authority的前提下，让复杂实施task的AI显式声明当前work item，并让pi-app只读显示executor、state、elapsed、blocker、next action和短证据。

## R1 — 上游与权威边界

- `task.json`和`implement.md`仍是task lifecycle、计划文本、顺序与checkbox完成状态的唯一权威。
- 不修改`.trellis/workflow.md`、`.trellis/scripts/`、上游`.pi/extensions/trellis/`、generated prompts/agents或Trellis npm package。
- Work-item extension/runtime缺失、损坏、过期或停用时，只隐藏实时可见性；Trellis task和普通开发必须继续工作。
- Runtime不得修改task status、checkbox、approval、commit或merge授权。

## R2 — 透明item identity

- 人类继续书写普通Markdown checkbox，不要求机器ID。
- 独立parser按task ref、heading path和规范化checkbox文本派生SHA-256 identity；UI显示原始checkbox文本，不突出内部hash。
- 同一heading下重复的规范化checkbox文本视为visibility ambiguity并显示issue，不阻断Trellis。
- Heading或checkbox文本变化使旧assignment失效；不得把旧assignment猜测迁移到新item。
- `[x]`/`[X]`只来自`implement.md`，runtime不能声明done。

## R3 — 动态激活

- 只有task status=`in_progress`、存在`design.md`和`implement.md`且plan至少含一个checkbox时，`nimloth_work_item`工具才进入Pi active tools。
- Planning、completed、无active task、轻量task、plan无checkbox或parser issue时不注入工具schema或prompt guideline。
- 每次session start、before-agent-start及task artifact变化后重新评估；状态变化最迟在下一次model request生效。
- 使用Pi官方`registerTool`/`setActiveTools`；不修改Pi core，不通过常驻prompt模拟动态行为。
- 不创建auto-discovered project skill；`select/update/evidence/release`简短用法只存在于动态active tool的description/result，ineligible session不加载该能力的skill metadata。

## R4 — Tool行为

`nimloth_work_item`支持：

- `select`：选择当前task的一个pending item并声明executor开始；
- `update`：更新state、blocker或next action；
- `evidence`：追加bounded短证据引用；
- `release`：结束当前assignment。

状态限定为`working`、`verifying`、`delegated`、`waiting_human`、`waiting_external`、`blocked`、`failed`。Tool必须绑定当前root、session/context、active task和有效item；跨root/task、checked/stale/ambiguous item及非法transition均fail closed。

## R5 — Runtime

- 路径为extension-owned、project-local、gitignored的`.local/nimloth-work-items/v1/`，不使用`.trellis/.runtime`。
- `.local`共享时按canonical worktree realpath/root fingerprint和context key隔离。
- 每个文件只保存schema version、root/task/item identity、executor/session、state、timestamps、blocker、next action及bounded evidence。
- 不复制task tree、完整plan、checkbox状态、tool args/output、prompt、response或CoT。
- 使用atomic write；损坏文件隔离为issue，不自动修复或阻断Trellis。

## R6 — 只读projection与pi-app

- Extension提供唯一只读projection，将当前task artifacts、透明parser和runtime join成versioned JSON；projection不执行mutation。
- pi-app不自行派生item identity、不写runtime，只验证projection schema/root/source freshness。
- 稳定Trellis panel的Activity视图显示当前item文本、executor、state、elapsed、blocker、next action和短证据。
- 无projection、stale、orphan、checked conflict或schema不支持时显示非阻塞降级；不得回退旧dashboard/typed approval数据。
- task status不是`in_progress`时projection必须输出`current=null`和可见`inactive-task` issue，不得展示旧runtime assignment。
- 同一逻辑workspace可按registered worktree source查看projection，但不切换workspace/session/worker cwd。

## R7 — Bounded与隐私

- Projection v1 producer与consumer使用同一硬上限：item text 2048、heading segment 512、heading depth 32、items 2000、issues 100、duplicate lines 100、task ref 512、status 64、blocker/next/evidence ref与summary 512、executor 128、session 256。越界plan内容必须变成bounded可见issue，producer不得输出consumer-invalid JSON。
- Evidence只允许`artifact/test/command/commit/job/url`等短引用和摘要，不保存命令输出正文。
- Runtime/projection不得包含CoT或assistant response正文。
- Context key、task ref、item ref和路径输入必须拒绝遍历与跨root。

## Acceptance Criteria

- [ ] Upstream 0.6.16 fidelity test保持GREEN，独立extension不进入template-managed路径。
- [ ] 普通checkbox获得稳定透明identity；文本变化失效，重复文本产生可见issue。
- [ ] 工具只在复杂`in_progress` task动态active，其他状态无schema/guideline/skill metadata开销。
- [ ] Tool transitions、atomic runtime、bounded evidence、root/session/task/item校验有RED/GREEN tests。
- [ ] Runtime位于`.local/nimloth-work-items/v1/`且不复制plan/CoT/task lifecycle。
- [ ] 唯一projection在missing/stale/corrupt/orphan/checked conflict时fail-visible但不阻断Trellis；非`in_progress`强制无current，全部projection-v1字段不超过consumer边界。
- [ ] pi-app稳定Activity视图只读显示当前item/executor/state/elapsed/blocker/next/evidence，并支持registered worktree source。
- [ ] 删除旧dashboard后，新projection consumer仍通过tests。
- [ ] 两仓库focused/final checks和独立review通过；未启动实验。

## Out of Scope

- 不实现typed approval、receipt、TaskTree或task lifecycle mutation。
- 不把work-item ID写回`implement.md`。
- 不修改上游Trellis/Pi core。
- 不记录CoT、完整tool payload/output或assistant正文。
- 不在本child处理generic question transport、legacy archive或live approval-runtime删除。
