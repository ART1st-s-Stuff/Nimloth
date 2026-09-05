# 执行计划 — 重建 Nimloth Trellis prompts

> 本文件只定义获批后的执行顺序。当前task仍为`planning`；在最终明确批准前，不创建实施branch/worktree、不运行`task.py start`、不修改产品代码。

## 0. 启动前冻结范围与建立隔离

- [x] 重新核验Nimloth和pi-app的实际root、branch、HEAD、remote及完整dirty状态，区分本task、其他task和未知改动。
- [x] 将当前task的base branch从错误默认值`main`改为`dev`，声明parent integration branch为`task/rebuild-nimloth-trellis-prompts`；在pi-app记录同名integration branch及其精确base commit。
- [x] 根据执行时并发状态选择目录：canonical空闲时在canonical checkout task branch；canonical已被其他writer占用时创建符合repository规则的worktree。本次不得覆盖Nimloth和pi-app现有并发dirty changes。
- [x] 当前task目录尚未tracked；迁入新worktree时，只复制该task目录及JSONL明确引用但尚未tracked的单个research source，逐文件核对hash，不携带其他uncommitted diff。
- [x] 创建设计中四个实施child及一个integration/cleanup child；每个child记录独立branch、base commit、repository、worktree和排除项。涉及pi-app的两个child串行执行，不共享实施branch。
- [x] 保存发布版Trellis `0.6.16`来源、`trellis update --dry-run`结果、template hash状态和旧runtime inventory，形成可回滚baseline；只读动作不得改变runtime。

## 1. Child：Nimloth baseline与政策层

### RED — 固定上游边界和prompt基线

- [x] 增加focused contract tests：列出必须与`0.6.16`一致的workflow/scripts/Pi prompts/agents/extension，并让当前本地分叉首先失败。
- [x] 增加prompt/context测量fixture，记录恢复前后主session与subagent的bytes、重复artifact读取和自动注入来源；测试只测量，不规定新的压缩方案。
- [x] 为薄`AGENTS.md`、skill路由、per-task branch、canonical主槽位、remote-only实验及15分钟总deadline补充静态合同测试；先证明当前规则分散或冲突。

### GREEN — 恢复上游并建立项目覆盖层

- [x] 使用发布包逐文件恢复template-managed Trellis `0.6.16`；禁止repository-wide force。对`.trellis/config.yaml`、`AGENTS.md`和project-local skills/specs只做明确的项目层改动。
- [x] 从上游所有的`.trellis/scripts/task.py`、`.pi/extensions/trellis/index.ts`及其测试中移除本地typed approval、dashboard、execution和work-item能力；额外custom modules只在确认不再被引用后进入删除清单。
- [x] 将`AGENTS.md`重写为已确认的薄安全内核；把实现合同迁入scoped specs，把操作步骤迁入project-local skills/references，并保留不可遗漏的高风险skill路由。
- [x] 将Git合同改为“每task独立branch；单一主task使用canonical directory；额外并行task使用worktree”；开发branch只作为task branch的base。
- [x] 将实验合同改为remote-only、`meta.kind=experiment`、短实验十分钟估算与从成功提交起十五分钟总deadline，并明确deadline取消后的defer/blocker路由。
- [x] 逐项确认上游generated files与发布包一致；记录允许保持project-owned的精确例外。运行恢复后context测量并保存原始结果，不在本child继续优化。

### 本child验证和审查

- [x] 运行上游一致性、task lifecycle、prompt/context及project contract focused tests；运行`git diff --check`。
- [x] 单独展示本child的spec diff、理由和影响；拒绝时只回退spec改动。获准后再进入该child的commit review。

## 2. Child：独立work-item可见性

### RED — 定义人类可见行为

- [x] 为Markdown checkbox解析、内部指纹、文本变化导致旧assignment失效、atomic runtime和bounded evidence建立失败测试；禁止要求显式`[W-xxx]`。
- [x] 为动态激活建立失败测试：planning/lightweight task不得出现tool schema或skill提示；complex `in_progress` task才激活；退出该状态后停用。
- [x] 为降级行为建立失败测试：extension缺失、unknown schema、root不匹配、stale session或损坏runtime时，Trellis task仍正常工作，pi-app只隐藏实时步骤。

### GREEN — 独立producer、runtime和Pi tool

- [x] 在`.pi/extensions/nimloth-work-items/`实现唯一parser/runtime/projection producer；按最终设计不增加常驻静态skill，避免planning/lightweight prompt开销。
- [x] Runtime写入project-local、gitignored、extension-owned路径，只保存task/item引用、executor/session、状态、时间、blocker/next action和短证据引用；禁止复制task tree、plan、checkbox完成状态或CoT。
- [x] 使用Pi受支持的动态tool API实现启停；planning/non-`in_progress`状态保持停用。
- [x] 提供独立只读projection命令，由Pi与pi-app共同调用；所有mutation显式绑定目标root/worktree并重验containment。

### GREEN — pi-app只读消费

- [x] 在隔离pi-app task branch中，把Trellis panel reader从修改过的`task.py dashboard`切换到上游静态task读取加独立work-item projection；移除approval字段依赖。
- [x] UI继续显示当前executor、状态、时长、blocker、next action和短证据；projection不可用时保留静态Trellis task信息并显示非阻塞降级状态。

### 本child验证和审查

- [x] 运行producer/parser/runtime、Pi动态激活、pi-app reader/model/panel focused tests及两仓库`git diff --check`。
- [x] 在各repository分别展示diff、验证证据和commit范围；经精确批准后提交并合入两仓integration。

## 3. Child：pi-app通用问题传输与TaskTree删除

### RED — 固定通用问题语义

- [x] 为source-session路由建立或保留失败测试：切换session/background不能自动decline；稍后处理、明确拒绝、timeout/abort和worker cancellation必须可区分；回答必须回到原session。
- [x] 增加absence tests，证明pi-app不再拥有Trellis approval kind、receipt、artifact hash、supersede逻辑或task lifecycle mutation。
- [x] 增加registry/build tests，证明删除`pi-task-tree`后generic adapter loading、side-panel primitives和`workspace-json`消费者仍正常。

### GREEN — 保留收件箱，删除治理和TaskTree

- [x] 从generic extension UI queue中剥离Trellis presentation/reconciler/response逻辑，只保留完整来源地址和普通user response回传。
- [x] 删除shared Trellis approval types、approval fixtures/projection、UI文案和只服务typed approval的tests；不得删除其他extension共用的问答能力。
- [x] 全局删除`pi-task-tree` builtin adapter、专用panel/model/mutation mapping、registry、docs和tests；保留共享`workspace-json`基础设施。
- [x] 历史prototype worktree仅保留为独立cleanup候选；未覆盖或删除canonical pi-app中的其他session dirty changes。

### 本child验证和审查

- [x] 运行extension UI queue、worker routing、adapter registry、Trellis panel及TaskTree absence focused tests。
- [x] 在clean隔离worktree中运行一次受影响范围的web/node typecheck；记录可在clean HEAD复现的无关失败，不通过降低检查强度规避。
- [x] 展示完整pi-app diff和删除清单，确认没有误删generic side-panel/`workspace-json`能力后，经批准提交并合入integration。

## 4. Child：Legacy知识迁移

### 审计与提炼

- [x] 生成156行known-error manifest，逐条记录文件、主题、是否仍有效、证据状态、目标spec章节和archive destination；数量和文件集合已精确匹配。
- [x] 只把经源码、测试或可信artifact重新核验的通用failure patterns写入对应spec；未把原事故全文复制进现行合同。
- [x] `local:M0013`保持`pending-human-verification`且未作为权威证据或合同来源使用；归档不改变其非权威状态。

### Archive迁移

- [x] 建立traceability index，将known-error原文件移动到确认后的历史archive root并退出active routing；更新有效链接和计数测试。
- [x] 核验`AI_branch_progress.md`和`ai_tasks/`的tracked/untracked/ignored payload及引用后，移动到既有pre-Trellis archive布局并删除active prompt/write路径。
- [x] Archive mutation严格使用最终批准中精确列出的254条manifest和18文件rewrite；未新增目标。

### 本child验证和审查

- [x] 验证manifest恰好覆盖迁移时的完整known-error集合、archive前后hash一致、active routing无遗留链接，且历史索引可追溯。
- [x] 单独展示known-error提炼产生的完整spec diff；经审查、精确批准后提交并合入parent integration。

## 5. Parent integration、旧runtime处理与最终检查

### 集成

- [x] 将各child已批准commit按依赖顺序集成到parent branch；每次merge前核验目标repo/worktree/branch/status，未force、自动解冲突或夹带canonical dirty changes。
- [x] 检查跨层数据流：上游Trellis生成/升级、Nimloth skill路由、work-item动态激活、pi-app只读展示、跨session普通问题回传、TaskTree absence和legacy archive。
- [x] 对parent最终spec diff再次做完整一致性审查，确认没有用task artifact放宽人类prompt、`AGENTS.md`或上游合同。

### 旧typed approval runtime

- [x] 对旧live request/receipt runtime重新盘点并完整复制到timestamped `.local`审计目录；8个payload的byte counts/SHA-256一致，payload/manifest为0440、目录为0550。
- [ ] 在删除live runtime前，向人类展示精确路径、命令及影响并取得即时破坏性批准。未获批准时保留live文件并把cleanup标记blocked，不影响已完成的新架构验证。
- [ ] 删除后证明新Trellis、work-item extension和generic question queue都不读取archive receipt。

### 最终affected-scope验证（每个输入不变的最终批次只运行一次）

- [x] Nimloth：运行上游一致性检查、全部受影响Trellis/extension/static CPU tests、known-error/archive合同检查、context测量及`git diff --check`。
- [x] pi-app：运行21个受影响test files（118 tests）、Node typecheck、lint、build及`git diff --check`；Web typecheck仅保留clean integration已记录的`fluent.tsx:107 TS2742`。
- [x] 独立review审查完整两仓库diff、PRD/design符合性、跨层数据流、删除范围、回滚能力及残余风险；runtime snapshot remediation后APPROVED，无P0–P2。

## 6. Commit、交付和回滚

- [x] 分repository展示完整修改范围、验证证据、child commits、parent merge顺序和dirty files；所有local commits均按精确范围授权，未push。
- [x] `dev`从394-entry dirty状态先对必要内容做hash-bound `.local`保护并精确清空；重新核验`dev@cbd05e5d`与parent `0232b763`为fast-forward关系后执行用户批准的merge。
- [x] 以非force fast-forward把`task/rebuild-nimloth-trellis-prompts`合入Nimloth `dev`；HEAD=`0232b763`，post-merge 18 Python、28 Node、archive/launcher/context/diff checks通过。
- [x] pi-app成果仅在pi-app repository的既定integration目标中处理，未写入Nimloth `dev`。
- [x] 完成work-item边界的Trellis progress记录；未提出重复memory候选，未编辑memory JSONL。
- [ ] 执行finish-work前确认所有非实验acceptance criteria均有证据、未启动任何实验、blocked cleanup被明确标注，随后按上游流程archive task和记录session journal。
- [x] 回滚顺序已由task设计、legacy byte/hash manifest和只读runtime snapshot验证：停止读取新runtime → 恢复`0.6.16` → 回退overlay/pi-app commits → 从archive恢复legacy；未执行force cleanup或push。

## 7. 已选择的规划证据

### Spec与research context

- `implement.jsonl`：10项，覆盖authority、Git/worktree、platform integration、task/progress/memory、实验、验证、不确定性及两份专项审计。
- `check.jsonl`：9项，聚焦最终权威边界、跨repo安全、平台所有权、实验门禁、验证和legacy routing。

### 源码证据

- 发布版Trellis：`@mindfoldhq/trellis` 0.6.16的`workflow.md`、`task.py`、Pi extension、generated agents/prompts及bundled skills模板。
- Nimloth现状：`AGENTS.md`、`.trellis/config.yaml`、`.trellis/workflow.md`、`.trellis/scripts/task.py`、`.trellis/scripts/common/{approvals,dashboard,execution,work_items}.py`、`.pi/extensions/trellis/index.ts`及对应tests。
- pi-app现状：`src/extension-compat/builtin/{trellis,pi-task-tree}.adapter.json`、adapter loader、`packages/shared/trellis-dashboard.ts`、workspace task panel reader、Trellis panel/model、extension UI queue、worker bridge及相关tests/docs。

### 单条known error

- `ai_rules/known_errors/E0094_bind_repo_mutations_to_the_target_worktree.md`：所有跨repo mutation必须在同一命令中绑定并核验目标cwd/root/branch，禁止依赖前一工具调用的目录状态。

### 经重新核验证据的memory

- `local:M0013`（Slurm controller必须独立于login session）：已读取memory并核对`ai_tasks/ai_progress/2026-07-27_sft2_h1_t4_dino_retrain.md:145-154`，事故证据支持该结论。
- 该memory仍是`pending-human-verification`，不能声称已经人类批准，也不能upvote；本task只把它作为非权威设计线索。Legacy迁移前必须处理其即将失效的evidence路径。

## 8. 明确排除项

- 不修改全局Trellis npm package、Pi core或`node_modules`。
- 不启动training、evaluation、collection、rollout、GPU、Slurm或远程job。
- 不把Pi TaskTree、typed approval、receipt或artifact-hash治理以新名称重新引入。
- 不自动清理其他session的Nimloth/pi-app dirty changes或旧worktrees。
- 不手工修改`.memory/memories.jsonl`、`.local/memory/memories.jsonl`、`.trellis/.template-hashes.json`或Trellis runtime session pointer。
- 不在本task继续优化恢复后的上游context；只测量并记录后续候选。
