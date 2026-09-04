# 实施计划 — pi-app Commit Review与Merge Review

## 执行纪律

- 每个work-item只实施其列出的范围，完成focused RED/GREEN后停止并展示diff与证据。
- 人类已授权在批准范围内持续执行；只有实质性设计选择、越界修改、破坏性操作或commit/merge门禁才暂停。
- pi-app代码只在独立`task/pi-app-cross-worktree-review` branch及对应worktree中修改。
- 每次跨repository mutation前在同一shell调用核验cwd、Git root、branch与status。

## W1 — 建立隔离分支并固定现状

- [x] 建立隔离pi-app branch/worktree并固定clean-base证据。
  - 从pi-app `task/rebuild-nimloth-trellis-prompts`精确base创建`task/pi-app-cross-worktree-review`及独立worktree。
  - 核验worktree root、branch、common Git dir、base SHA和clean status；不得触碰pi-app canonical dirty changes。
  - 运行现有Review focused tests并记录clean-base结果；只分类既有失败，不在本item修复。
  - 只读确认Review IPC、Git helper、Renderer和i18n的实际修改边界。

**验证**：`git status --short --branch`；现有Review与Git workspace focused tests。

## W2 — RED：固定Git审查语义

- [x] 添加Main Git审查语义RED测试。
  - 在临时repository fixture创建分叉branch、remote-tracking ref及两个registered worktree。
  - 添加Commit Review RED：当前/其他worktree staged严格隔离，unstaged和untracked不进入结果。
  - 添加目标校验RED：不同common Git dir、消失worktree、realpath不匹配必须fail closed。
  - 添加Merge Review RED：Base/Compare通过merge-base比较，结果只包含Compare committed changes。
  - 添加branch边界RED：消失ref、Base=Compare、无merge-base、remote-tracking branch及ref切换。

**验证**：只运行新增`git-workspace` focused tests，并证明失败原因是缺少目标发现和两类snapshot实现。

## W3 — GREEN：实现Main只读Git服务

- [x] 实现Main只读Git审查服务并使W2 tests通过。
  - 实现registered worktree和local/remote branch发现，返回规范化path/ref/SHA。
  - 实现Commit Review snapshot，只执行目标worktree的`git diff --cached`。
  - 实现Merge Review snapshot，重新解析ref/SHA、计算merge-base并读取committed diff。
  - 每次调用重新核验realpath、registry和common Git dir；所有Git命令使用`execFile`参数数组。
  - 保持现有mutation helper和trusted-workspace授权语义不变。

**验证**：W2新增tests全部GREEN；复跑现有`git-workspace` tests。

## W4 — RED/GREEN：增加类型化只读IPC

- [x] 增加类型化只读Review IPC并完成RED/GREEN。
  - 先添加handler RED，覆盖trusted workspace绑定、Renderer伪造path/ref、目标在list后消失及错误透传。
  - 在shared contract、IPC channels和schema中增加`review.listGitTargets`、`review.getCommitDiff`、`review.getMergeDiff`。
  - 实现三个handler；不得接受任意repository root，不得调用mutation helper。
  - 保留现有mutation IPC；确认非当前worktree不能通过新response获得`mutable=true`。

**验证**：新增Review handler tests、IPC contract/typecheck focused检查全部GREEN。

## W5 — RED：固定Renderer交互合同

- [x] 添加Renderer双模式与权限边界RED测试。
  - 添加默认Commit Review RED，只显示当前worktree staged diff。
  - 添加mode/selector RED：Commit显示worktree selector，Merge显示Base/Compare selector。
  - 添加权限RED：当前worktree显示unstage/commit；其他worktree和Merge隐藏mutation controls。
  - 添加空staged、目标错误和快速切换RED，证明不回退unstaged且旧响应不能覆盖新目标。
  - 添加状态隔离RED，证明切换Review target不调用workspace/session/worker API。

**验证**：只运行新增Review Renderer tests，并确认均因尚未实现两个mode而失败。

## W6 — GREEN：实现两个Review面板

- [x] 实现Commit Review与Merge Review两个Renderer面板并使W5 tests通过。
  - 将`ReviewPanel`拆为顶层mode、Commit Review panel和Merge Review panel。
  - 实现targets、commit snapshot及merge snapshot hooks，request identity包含mode与目标SHA。
  - Commit Review仅解析staged diff；Merge Review仅解析committed diff。
  - 给diff组件增加显式mutation capability；当前worktree保留unstage/commit，其他目标严格只读。
  - 其他worktree文件打开使用已校验root；Merge Review不提供打开当前文件入口。
  - 添加中英文标签、空状态和可见错误文案。

**验证**：W5新增tests及现有Review focused tests全部GREEN。

## W7 — Refactor与最终affected-scope检查

- [x] 完成Refactor、最终affected-scope检查和提交前审查。
  - 消除Main/Renderer重复类型与重复target identity逻辑，不扩大文件范围或引入新依赖。
  - 检查评论identity包含mode、repository、target和SHA；不同审查对象不得串评论。
  - 运行一次最终affected-scope unit tests、typecheck、lint和build；既有clean-base失败必须与W1证据比对。
  - 运行`git diff --check`，展示完整pi-app diff、验证证据、残余风险和精确commit范围。
  - 未经人类单独批准，不创建local commit；不push、不merge、不清理worktree。

**最终验证命令**：

```bash
npm run test:unit -- --run <affected-review-and-git-tests>
npm run typecheck
npm run lint
npm run build
git diff --check
```

## W8 — Main正确性修复

- [x] 修复WSL路径边界与Commit snapshot一致性并完成RED/GREEN。
  - 先添加WSL contract RED，覆盖Git返回Linux common-dir/worktree path而host使用Windows/UNC trusted workspace的情况。
  - 复用现有WSL delegate/path conversion边界；禁止把WSL-native path直接交给host `realpathSync`或`path.resolve`。
  - 先添加并发snapshot RED，模拟读取期间HEAD或index变化，禁止返回混合状态。
  - Commit snapshot改为单一patch读取并在前后核验HEAD；以staged patch内容生成稳定snapshot identity，避免依赖并发的独立status进程。
  - 保持Merge Review、mutation helper、IPC信任边界和Renderer行为不变。

**验证**：新增WSL/race tests、现有Git review/Git workspace tests及Node typecheck focused检查。

## W9 — Renderer安全与评论身份修复

- [x] 禁用不安全的跨worktree文件打开并绑定稳定评论identity。
  - 第一版只允许当前trusted workspace打开文件；非当前worktree和Merge Review隐藏打开/定位入口。
  - 评论identity加入Commit snapshot identity；hunk key绑定规范化patch内容而非数组位置。
  - 添加staged hunk替换/重排回归测试，证明HEAD不变时旧评论不会串到新hunk。
  - 补充Merge Review快速切换及trusted workspace变化的旧响应隔离测试。
  - 不新增capability IPC，不扩大跨worktree权限。

**验证**：Renderer identity/mode tests、Review focused tests及Web typecheck focused检查。

## W10 — Typecheck归因、最终验证与复审

- [x] 重新归因TS2742并完成提交前质量门禁与独立复审。
  - 在clean HEAD与feature diff上复现并定位TS2742的实际import/type传播路径；若两者相同则记录为worktree依赖布局限制，不修改无关生产文件。
  - 只有证明由新增Review代码触发时才修复新增推断或icon引用；若必须修改既有`fluent.tsx`公共类型边界，先停止并请求范围批准。
  - 运行全部affected tests、完整unit suite、Web/Node typecheck、lint、build和CRLF-aware diff check。
  - 将clean-base rewind失败与feature回归分开报告，不把真实feature失败标为既有问题。
  - 再运行一次独立只读code review；存在P1/P2 finding时不得stage或请求commit批准。

**验证**：所有feature-attributable检查GREEN；完整diff、残余clean-base失败和精确commit范围可供人类审查。

## Guardrails

- 不修改Nimloth代码、全局Pi/Trellis、`node_modules`或任何canonical dirty文件。
- 不调用`workspace.open`或`activateWorkspace`实现Review target切换。
- 不对非当前worktree执行stage、unstage、discard或commit。
- 不执行checkout、switch、merge、rebase、push、force、clean或reset。
- 不把worktree/branch选择写成第二份task、session或workspace状态。
- 不顺手处理TaskTree、Trellis transport、既有rewind/CSS/typecheck问题或其他pi-app债务。
