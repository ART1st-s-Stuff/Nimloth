# 技术设计 — 重建 Nimloth Trellis prompts

## 1. 总体架构

```text
发布版 Trellis 0.6.16（保持不改）
  workflow · scripts · Pi生成prompt/agent · bundled skills
                         │
                         ▼
Nimloth项目覆盖层
  薄AGENTS.md ──路由──► scoped specs ──► project-local skills
                         │
                         ├──► 实验/Slurm/worktree/memory操作流程
                         └──► 独立work-item可见性extension

Pi标准问题UI ── 通用source-session队列 ── pi-app
独立work-item producer ── 只读projection ── pi-app Trellis面板
```

Trellis继续作为task身份、生命周期和规划artifact的唯一权威。项目extension可以引用这些对象，但不能维护平行的task或approval状态。

## 2. 所有权划分

| 所有者 | 路径或职责 | 更新规则 |
|---|---|---|
| 发布版Trellis | `.trellis/workflow.md`、template-managed `.trellis/scripts/`、`.pi/extensions/trellis/index.ts`、生成的prompts/agents和bundled skills | 恢复为发布版`0.6.16`；不写入Nimloth政策 |
| 人类治理安全内核 | `AGENTS.md` | 只保留PRD R2/R11定义的薄合同；经审查task可以实施其中明确列出的diff |
| 项目合同 | `.trellis/spec/` | 保存实现、domain和operations合同；task结束时对精确spec diff单独审查 |
| 项目操作流程 | project-local `.agents/skills/` | description精确声明触发场景；`SKILL.md`保持简短，长说明和脚本按需加载 |
| Work-item可见性 | `.pi/extensions/nimloth-work-items/`及对应project-local skill | 可选观察层；只在复杂in-progress task中动态激活 |
| pi-app | 通用待处理问题传输和只读Trellis/work-item消费 | 不拥有Trellis审批权威；不保留TaskTree能力 |
| 历史证据 | archive路径和`.local`审计快照 | 不自动注入为现行政策 |

## 3. 恢复上游的边界

1. 修改前记录`trellis update --dry-run`、template hashes和精确本地diff。
2. 从发布版`0.6.16`恢复所有template-managed通用文件；禁止使用会覆盖用户数据的repository-wide `--force`。
3. 从上游所有的`task.py`和Trellis Pi extension中移除本地approval、dashboard及work-item扩展。
4. 将仍需保留的work-item逻辑迁入独立extension package，禁止继续从修改过的Trellis scripts导入。
5. 只有上游明确提供配置入口时，才在project-owned config中保留项目配置。
6. 恢复后，`trellis update --dry-run`必须证明通用workflow/scripts/Pi agents/prompts/extension不存在Nimloth政策差异。

## 4. 薄安全内核与按需合同

`AGENTS.md`只包含PRD R2确认的六类规则和一行式强制skill路由，不包含操作命令。

普通skill使用Pi原生name/description渐进式披露。强制路由只点名遗漏后可能跨越安全边界的project-local skills：

- 实验准备、启动和结束；
- Slurm及远程GPU操作；
- 并发需要隔离时的worktree创建与cleanup；
- curated memory mutation；
- 长job状态变化及跨session交接。

CoT/state、checkpoint、training、rollout和model语义保留在domain specs中，只在相关task选择context后加载。

## 5. 实验与远程操作控制流

### 5.1 Task合同

所有真实实验都设置`meta.kind=experiment`。复现或部分参数重跑可以使用PRD-only轻量规划，但仍必须记录：

- 精确source及repository/branch；
- 修改或复用的参数；
- 预计时长和运行次数；
- 输出位置；
- 该结果是否阻断当前阶段。

禁止在本地执行training、evaluation、collection或rollout。Local CPU unit/static checks不属于实验执行。

### 5.2 短时远程实验

短实验指预计十分钟内完成的复现或部分参数重跑。提交前，Slurm skill读取真实剩余容量，并选择任何不会抢占或干扰现有工作的可用GPU/CPU数量。

controller记录成功提交时间，并执行总deadline：

```text
deadline = submitted_at + 15 minutes
```

排队和运行共同计时。scheduler runtime limit本身不能限制pending时间，因此monitor/controller必须在总deadline到期时取消仍处于pending或running的job。

Deadline取消后：

- 记录job ID、观察到的状态和取消原因；
- 如果实验尚未成为当前代码批次的依赖，继续其他无关工作；
- 到下一个阶段边界，或下游工作开始依赖该结果时，停止并报告blocker；
- 禁止把取消、缺失输出或defer状态转成验证成功。

预计超过十分钟的实验需要通过普通对话取得明确launch决定。Experiment task自动获得其声明repository/remote/branches范围内的正常Git同步权限；force操作仍使用独立破坏性门禁。

## 6. pi-app如何显示“AI正在做哪一步”

复杂task的`implement.md`是一份带checkbox的执行计划。人类希望在pi-app中看到：当前哪个AI正在处理哪一条、已经持续多久、是否被阻塞、下一步是什么。

这项功能不再改造Trellis本身，而由一个独立的Nimloth extension完成：

1. Task还在规划阶段，或只是轻量task时，该extension保持安静，不向AI增加tool或额外说明。
2. 复杂task进入实施后，extension读取`implement.md`中的checkbox列表，并让AI明确选择自己正在处理的那一条。
3. Extension把“session A正在处理步骤B、当前状态和更新时间”写入自己的gitignored临时目录；pi-app只读取并展示。
4. AI完成或切换步骤时更新临时状态，但真正的完成状态仍只来自人类可见的`implement.md` checkbox。

Extension不会向`implement.md`写入`[W-xxx]`。它在内部根据task、标题层级和步骤文字生成一个不可见指纹。步骤文字被修改后，旧关联自动失效，AI需要重新选择；系统不能猜测新旧步骤是同一项。

建议文件边界为：

```text
.pi/extensions/nimloth-work-items/              # extension及唯一解析程序
.agents/skills/nimloth-work-item-visibility/    # AI如何选择/更新步骤
.pi/.runtime/nimloth-work-items/                # gitignored临时状态
```

临时状态只保存引用、AI/session身份、状态、时间、blocker、next action和短证据引用，不复制task内容或plan。Extension缺失、损坏或版本不兼容时，最坏结果只能是pi-app不显示实时步骤；Trellis task仍可正常规划、实施和完成。

## 7. pi-app如何传递“需要人回答的问题”

此前系统把Trellis审批做成了五种类型、hash和receipt，导致UI看起来像独立授权机关，也造成重复审批。新设计完全删除这套含义。

重建后，pi-app只充当一个通用“问题收件箱”：

1. AI在某个session中提出普通问题，例如“是否批准这份最终计划？”或“是否执行这条破坏性命令？”
2. 如果人类切换到其他session，问题仍留在待处理列表，不会被自动回答或伪装成拒绝。
3. 人类稍后回答时，pi-app根据原始workspace、session、request和worker信息，把回答送回最初提问的session。
4. 原session中的AI收到一条普通的人类回复，再按照上游Trellis workflow或项目安全规则决定下一步。

pi-app不判断“批准是否有效”，不生成receipt，不比较artifact hash，不运行`task.py start`，也不把按钮点击写成Trellis状态。`稍后处理`、人类明确拒绝、系统超时和worker终止必须保留不同含义。

技术上，每个问题都带有类似信封地址的完整来源标识：

```text
workspace/root + sessionFile + requestId + toolCallId + source worker
```

该标识只用于把回答送回正确会话，不构成授权证明。

## 8. 删除 pi-app TaskTree

删除内置`pi-task-tree` adapter及全部专用TaskTree panel/model/mutation/docs/tests。移除只服务该adapter的registry imports和component registrations。保留其他adapter共用的generic adapter loading、side-panel primitives和`workspace-json` state transport。

Trellis面板保持task权威只读。它消费上游静态task数据；独立work-item projection可用时，再叠加对应的runtime可见性。

## 9. 验证模型

保持上游agent/workflow文字不变。Nimloth Python quality spec定义可执行的项目语义：

- implementation loop：针对改动ownership boundary运行focused RED/GREEN；
- independent check：检查完整affected diff，并运行一次最终affected-scope static/CPU suite；
- 同一未变化final batch内可以复用已有成功证据，但不新增receipt/fingerprint runtime；
- 明确报告缺失依赖与skipped检查，不得转换为pass；
- 真实GPU/Slurm/model-quality证据只属于experiment合同。

在进一步优化前，先测量恢复后的上游context prompt。测量记录byte size和重复读取，但本task不修改该baseline。

## 10. 知识与遗留系统迁移

### 10.1 Known errors

创建包含156行的audit manifest，每条记录disposition和证据。按所属spec聚合已核验的重复failure patterns，对生成的spec diff单独审查，再将原始事故文件及索引移动到`ai_rules/archive/known_errors/`，或实施时确认的最近既有archive root。Archive保留traceability index，但退出active routing。

### 10.2 旧进度系统

核验payload和链接后，把`AI_branch_progress.md`及legacy `ai_tasks/`移动到既有pre-Trellis历史archive布局，并停止所有prompt引用和后续写入。

### 10.3 旧审批runtime

停用旧producer前，将完整live request/receipt runtime复制到带timestamp的只读`.local`审计目录，并核对byte counts和hashes。删除live runtime属于破坏性门禁：执行前展示精确路径并取得批准。禁止把archive receipt导入新问题队列。

## 11. 交付拆分

当前task作为parent integration contract。实施阶段拆为可独立验收的child tasks：

1. **Nimloth baseline与政策层**：恢复上游`0.6.16`，重写薄`AGENTS.md`，整理spec/skill并测量context。
2. **独立work-item可见性**：迁出producer/runtime/tool，再适配pi-app Trellis consumer。
3. **pi-app交互清理**：保留通用问题队列，删除typed approval语义和全局TaskTree能力。
4. **Legacy知识迁移**：审计156个known errors，单独审查spec diff，归档事故记录和旧progress。
5. **Integration与cleanup**：跨child验证，对live approval-runtime删除执行独立门禁；完成最终审查后，将Nimloth parent integration branch经单独批准合入`dev`。

依赖关系：child 1先于child 2和child 4；pi-app child 2与child 3在同一个隔离pi-app worktree中串行实施，确保one-writer ownership；child 5等待其余child完成。

## 12. Task branch、工作目录与回滚

- 每个task使用独立Git branch，并把branch记录在task metadata中；不同task不得共享实施branch。
- 单一主task在canonical directory checkout自己的task branch；repository声明的开发branch只是创建task branch的base，不直接承载task实现。
- 出现并行task时，当前主task继续留在canonical directory，新增并行task各自创建worktree并checkout自己的branch；不因为并发开始而搬迁已在canonical工作的主task。
- Canonical存在未提交修改时不得切换到其他task branch；并行task结束后按已确认的clean-check规则cleanup其worktree。
- pi-app当前存在并发dirty work，因此本task涉及pi-app的实施必须从记录的base创建隔离worktree，禁止覆盖canonical changes。
- 本重建task不启动任何实验。
- 每次迁移或删除前记录精确source hashes和destination manifest。
- 所有Nimloth child先汇入parent integration branch；待`dev`可安全作为clean merge target时，重新核验两端SHA、完整merge diff和最终验证证据，取得独立merge批准后再以非force方式合入`dev`。pi-app成果只在pi-app repository中集成。
- 回滚时，发布版Trellis文件从`0.6.16` package恢复，project文件从task diff恢复，legacy runtime/data从已核验archive恢复。删除独立extension后，上游Trellis必须仍可正常使用。
