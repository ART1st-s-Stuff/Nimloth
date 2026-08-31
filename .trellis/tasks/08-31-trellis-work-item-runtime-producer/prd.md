# 实现 Trellis work-item runtime producer

## Goal

在Nimloth本地Trellis/Pi integration中建立work-item identity、plan/task-tree dashboard和session-scoped runtime assignment producer，使每个主Agent/子代理run可显式关联到唯一Trellis plan item。

## Parent contract

Parent：`08-31-implement-trellis-work-item-visibility`。

本child拥有schema和producer；pi-app consumer不得自行发明不同plan/runtime语义。

## Requirements

- 解析`task.json.parent/children/status`和`implement.md`heading/checkbox/item。
- 支持显式`[W-xxx]`及标记为unstable的legacy derived ID。
- duplicate/malformed ID、tree cycle、missing child、orphan cursor和checkbox/runtime冲突可见且fail closed。
- 提供versioned、只读dashboard JSON，供pi-app异步调用；projection不得持久化第二任务清单。
- Dashboard提供raw `prd.md`/`design.md`/`implement.md`、SHA-256、相对上次review request的变化和typed approval request projection，使人类能审查Agent新建task的exact artifacts。
- Approval request必须绑定taskRef、kind、artifact hashes、scope/exclusions、session/request identity；artifact变化后receipt失效。
- 在`.trellis/.runtime/execution/<context-key>.json`原子写入assignments。
- assignment支持main/subagent、多executor、declared state、timestamps、heartbeat、blocker、next action和typed evidence。
- done只由`implement.md`checkbox决定；runtime工具不得直接标记plan done。
- Pi extension提供显式select/update/block/evidence/release能力，并把tool lifecycle与subagent run附着到assignment。
- `ctx.cwd`权威、root+session隔离、reload恢复和stale处理满足platform integration合同。
- 同步workflow、governance spec和相关skills，使Agent在实质item开始/切换/委派/阻塞/完成时维护cursor。
- 不使用Pi TaskTree，不批量迁移旧计划，不修改实验/训练语义。

## Authorization

- 当前只批准规划，不批准修改`.trellis/scripts/`、`.pi/`、spec、workflow或skills。
- 不commit、push、merge；不启动实验或远程job。
- implementation approval必须展示exact文件范围、schema、阈值、测试和template divergence。

## Acceptance Criteria

- [ ] Plan parser对现有376项只读兼容，显式/legacy identity行为有测试。
- [ ] Dashboard-v1 fixture稳定覆盖task tree、sections/items、assignments、review package、approval request和issues。
- [ ] Artifact hash/diff与typed approval request/receipt validation有测试，旧receipt不能授权修改后的plan。
- [ ] Runtime writer atomic、root/session隔离且不修改task/plan。
- [ ] 状态转换、heartbeat、waiting persistence、shutdown/stale/conflict通过测试。
- [ ] Pi main/subagent assignment和tool heartbeat通过safe probe。
- [ ] workflow/spec/skills语义一致，并通过template-hash/divergence审查。
- [ ] `ctx.cwd` foreign process cwd、同session跨root、reload验证通过。
- [ ] Pi TaskTree未被读取或写入。
- [ ] Consumer获得versioned schema、fixtures和调用说明。
