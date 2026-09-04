# 实施计划 — pi-app Trellis任务浏览器

## 执行规则

- 获得最终规划批准后，可在已批准范围内连续执行；遇到产品语义、越界修改、破坏性操作或commit/merge门禁时停止。
- pi-app修改只在`task/pi-app-trellis-task-browser`独立branch/worktree进行，base为parent integration commit `f84406a`。
- 每次跨repository mutation绑定并核验精确cwd/root/branch/status；不触碰canonical dirty内容。

## W1 — 建立隔离与clean baseline

- [x] 创建并核验pi-app child branch/worktree。
  - 从`task/rebuild-nimloth-trellis-prompts@f84406a`创建`task/pi-app-trellis-task-browser`。
  - 核验root、branch、common Git dir、clean status和依赖解析方式。
  - 运行现有Trellis panel/static reader focused baseline并记录既有失败。

## W2 — RED：固定聚合与artifact安全合同

- [x] 添加Main reader RED tests。
  - 临时repository创建两个registered worktrees、同名task多版本、active completed、archive及malformed task。
  - 固定source排序、instance identity、生命周期分类和archive lazy读取。
  - 固定artifact allowlist、missing/oversize、absolute/`..`/symlink escape。
  - 固定worktree消失、realpath替换、foreign common-dir fail-closed行为。

## W3 — GREEN：实现Main task browser reader

- [x] 实现独立只读reader并使W2 GREEN。
  - 复用`listGitReviewTargets`发现同repository registered worktrees。
  - 实现source/task inventory、同名多版本隔离、archive按需扫描和bounded issues。
  - 实现instance/artifact重新解析、allowlist、realpath containment和byte limit。
  - 不import或调用旧dashboard、approval、execution或Trellis mutation逻辑。

## W4 — RED/GREEN：类型化只读IPC

- [x] 添加并实现三个task-browser IPC。
  - 先测试无trusted workspace、extra/root/path注入、instance/source消失和错误透传。
  - 增加shared contract/channels与strict Zod schemas。
  - Handler只使用`getTrustedWorkspaceRoot()`及W3 reader；不得接受absolute repository/task path。
  - 复跑IPC contract和Main reader tests。

## W5 — RED：固定Renderer行为

- [x] 添加Trellis task-browser UI RED tests。
  - 默认未完成tab；首次进入已完成tab才请求archive。
  - 显示source branch/path/current、source filter和同名多版本。
  - 选择task后列出允许artifact，Markdown rendered/source切换，JSON/JSONL只读source。
  - 覆盖missing/malformed/oversize/source消失及快速切换旧响应隔离。
  - 断言不调用workspace/session/worker、Trellis mutation或Git mutation IPC。

## W6 — GREEN：实现任务与文档浏览UI

- [x] 实现独立task browser入口并使W5 GREEN。
  - 在Trellis panel中增加与运行/审批区域分离的任务入口。
  - 实现生命周期tabs、source filter、task rows、artifact tabs和只读内容视图。
  - 复用既有Markdown renderer；不复制或依赖旧approval/dashboard schema。
  - 添加中英文文案、loading/empty/error状态和generation identity。

## W7 — Refactor、最终验证与复审

- [x] 完成affected-scope检查和独立review。
  - 消除重复source/instance/allowlist逻辑，不新增dependency。
  - 运行Main/IPC/Renderer focused tests及相关既有panel tests。
  - 运行完整unit suite、Web/Node typecheck、lint、build和CRLF-aware diff check；既有失败必须由clean-base证据支持。
  - 独立review检查PRD/design、路径安全、source隔离、lazy archive、旧响应和后续dashboard删除兼容性。
  - 展示完整diff、验证、残余风险和精确commit范围；未经批准不stage/commit。

## Guardrails

- 不修改Nimloth生产代码、全局Pi/Trellis、node_modules或canonical dirty文件。
- 不实现task编辑、status修改、approval、Git操作、workspace/session切换。
- 不聚合未checkout branch，不跨Git common directory，不自动合并同名task版本。
- 不在本task删除旧dashboard/typed approval/TaskTree；该范围留给后续cleanup child。
- 不处理generic question transport、work-item runtime或legacy archive。
- 不push、不merge、不cleanup worktree，除非到达对应独立批准门禁。
