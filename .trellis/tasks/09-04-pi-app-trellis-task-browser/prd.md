# 需求草案 — 改进pi-app Trellis任务浏览

## Goal

让人类在pi-app的Trellis面板中清楚区分未完成与已完成任务，并通过独立入口选择task、只读查看PRD等任务artifact，不再依赖审批视图才能阅读任务文件。

## 已确认事实

- 当前`WorkspaceTasksSidePanel`在static fallback中把所有`.trellis/tasks/*`任务放入同一`data.tasks`列表，仅在每行显示`status`。
- 当前dashboard模型已有task `status`、`location`及`review.artifacts`，但artifact正文只嵌在“审查与审批”区域。
- 当前static reader只读取`task.json`和`prd.md`摘要，并跳过`tasks/archive`；没有通用任务artifact读取接口。
- Parent架构将删除typed approval/dashboard私有治理，因此新浏览器不能把审批receipt或work-item runtime作为读取任务文档的前提。
- 本task只负责只读任务浏览，不修改Trellis lifecycle、task artifact或session状态。

## 初始需求

### R1 — 分离任务生命周期视图

- 使用两个独立tab：`未完成`与`已完成`，默认打开`未完成`。
- 分类依据来自Trellis task status，不由pi-app维护第二份状态。
- planning、in_progress及其他非终态归入未完成；completed归入已完成。
- `已完成`同时包含`.trellis/tasks/`中status为completed的任务，以及`.trellis/tasks/archive/<YYYY-MM>/`中的归档任务。
- 归档目录按需读取，不能拖慢默认`未完成`tab；归档task若metadata损坏需显示失败状态。
- 状态未知或task.json损坏时可见失败或单独标记，禁止静默归入completed。

### R2 — 独立任务文档入口

- Trellis面板增加独立的任务文档入口，与运行状态、审批和问题transport分离。
- 人类可以选择一个task并查看其任务artifact。
- 第一版支持核心文件`task.json`、`prd.md`、`design.md`、`implement.md`、`progress.md`、`implement.jsonl`、`check.jsonl`，以及`research/`下的Markdown文件。
- Markdown默认使用pi-app既有Markdown renderer渲染，并提供只读源码切换；JSON/JSONL使用只读源码视图。
- 缺失文件显示为缺失，不伪造内容；`research/`不存在时显示空态。
- 文档严格只读，不提供编辑、状态修改、approval、commit或Git操作。
- 文件读取必须限制在当前repository的`.trellis/tasks`允许目录及明确artifact allowlist，拒绝路径穿越和任意文件读取。

### R3 — 数据源边界

- Trellis仍是唯一task authority；pi-app只投影磁盘上的task metadata/artifact。
- 新入口不能依赖即将删除的typed approval receipts或自建work-item runtime。
- 当前workspace/session保持不变；选择task只改变Trellis面板的查看对象。

### R4 — 聚合同repository registered worktrees

- 从当前trusted workspace出发，只聚合同一Git common directory中`git worktree list --porcelain`登记的worktrees。
- 当前worktree默认排在首位；每个source清楚显示branch、path、HEAD及current标记。
- 同一task出现在多个worktree时保留为多个独立版本，分别显示source identity；禁止按task name自动合并、覆盖或选择“最新”。
- 选择其他worktree的task只改变只读浏览目标，不切换workspace/session/worker cwd，不执行checkout，也不修改该worktree。
- Worktree消失、realpath变化、common-dir不一致或artifact损坏时，对该source显示可见失败，不回退到其他worktree的同名task。
- 未checkout的branch不参与task浏览；branch committed diff属于独立Merge Review能力。

## 初始验收条件

- [ ] 未完成与已完成任务在UI中明显分离。
- [ ] 独立任务文档入口可选择task并查看允许的Markdown artifacts。
- [ ] 缺失、损坏和读取失败状态可见且不会显示其他task内容。
- [ ] 任务选择不修改Trellis、Git、workspace、session或worker状态。
- [ ] 同repository registered worktrees被聚合并标注source；同名task多版本保持独立。
- [ ] Tests覆盖状态分类、artifact allowlist、路径穿越、worktree消失/common-dir不一致、同名多版本、task快速切换和旧响应隔离。

## Out of Scope

- 不处理typed approval删除、TaskTree删除或generic question transport。
- 不修改Trellis task状态、artifact、runtime或memory。
- 不跨不同Git repository读取task。
- 不修改当前`pi-app-cross-worktree-review`实现。

## 已确认决定

- 聚合同repository registered worktrees。
- 同名task按worktree source独立显示，不自动合并。
