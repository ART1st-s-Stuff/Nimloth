# SFT1/SFT2 debug 到新一轮训练的工作流延迟审计

日期：2026-08-31

审计窗口：2026-08-26 至 2026-08-31 05:54 UTC

性质：只读证据审计；未访问远程调度器，未修改源码、历史任务、实验或运行产物。

## 1. Executive summary

### 1.1 首先纠正当前状态

截至本地最新证据，**新一轮正式 SFT1 尚未开始训练**。Formal32 Slurm Job `539107` 于 `2026-08-31T13:28:16+08:00`（05:28:16 UTC）提交，但记录状态仍为 `PENDING (Priority)`，没有 allocation、CUDA、run/controller 或 W&B runtime。今天发生的是“正式训练提交”，不是“健康训练启动”。SFT2 也尚未开始；按已选阶段合同，它被有意阻塞在 formal SFT1 terminal evidence 和人工 state gate 之后。

证据：

- `.trellis/tasks/08-29-train-sft1-query-state/progress.md`，章节“Formal32 exact launch已提交”；
- 同文件“当前监控边界”；
- Job `539107` 的本地只读监控会话 `01a0564c-266...`。

### 1.2 为什么花了五天

直接原因不是单一的“审批太多”或“测试太慢”，而是以下关键路径叠加：

1. **目标在 8 月 27–28 日发生了实质纠正。** 初始方案膨胀为全量 14,256-row teacher cache + 三轮 broad canary；人类明确拒绝该形状，要求先诊断 Query token/state supervision。此后又把 instruction-first 改为 DeepSight-aligned Query-State，并选择 SFT1 → SFT2a → SFT2b。旧实现和旧 launch identity 因而不再位于关键路径。
2. **正式训练 owner 当时并不存在。** 8 月 29 日的 source audit 明确指出：已有代码只有 mechanics primitives，没有 pilot/formal schema、controller、validation/checkpoint/W&B transaction 和完整 diagnostics。8 月 30 日才提交 production owner `d0d5d62b`。
3. **生产路径缺陷被推迟到真实 GPU/FSDP job 才逐个发现。** 从 smoke 到 pilot，先后暴露 import provenance、BF16/FP32 FSDP composition、TensorDict iteration、runtime row audit、validation global normalization、generation boundary、checkpoint rank consensus、pilot tracking dispatch 等缺陷。它们会使训练无效或崩溃，不能跳过；但发现得太晚，导致反复生成 fresh commit/config/controller/output identity。
4. **审批有真实摩擦，但在 8 月 30 日取得 standing authorization 后已不是主要瓶颈。** 最大可确认等待包括一次本地 commit 门禁约 5 小时 23 分，以及 Formal32 candidate 再次请求确认后的夜间约 6 小时 29 分。后者尤其暴露了“candidate approval → final lock → launch approval”两次询问过细的问题。
5. **测试本身通常很快，慢的是完整迭代循环。** 本地 focused/adjacent pytest 多为 5–10 秒，远端一组 104 tests 为 57 秒。用户感受到的“测试时间”主要是“失败 → RED/GREEN → overlapping suite → 独立 review → commit/push → fresh preflight/contract → 新 job”的端到端串行成本。
6. **worktree/storage/治理旁支工作切断了关键路径可见性。** 8 月 28 日 Git 中可见约 10:03–14:59 UTC 的 prompt/worktree 迁移提交；用户随后也明确说“花了一天整理 worktree”。这些工作部分是必要基础设施修复，但在同一主会话中与 SFT 交织，使用户无法判断训练任务是否仍在推进。
7. **Pi/pi-app 有工具时间线和子代理卡片，但缺少统一的“关键路径/阻塞原因/下一门禁”状态。** 因此“正在读文件”“跑测试”“等待人类”“等待 VPN”“等待 Slurm”在体验上都可能表现为 Agent 没有动作。

### 1.3 责任判断

- **必要且正确：** real CoT/state 语义门禁、exact data/split、GPU launch 单独审批、fail-closed runtime checks、checkpoint/resume 和 distributed correctness 验证。
- **Agent 可改进：** 初始 canary scope 失焦；production preflight 与真实 runtime 不等价；每个小修复都重复 review/contract；在已有广泛授权后仍先问 candidate approval，再计划问 launch approval；状态汇报术语过多且缺少关键路径摘要。
- **项目策略可改进：** commit/push/retry 授权粒度过细；没有标准 authorization envelope；on-progress/独立 review 粒度偏重；没有 machine-readable `blocked_on`。
- **Pi/pi-app 可改进：** 已有时间线、tool lifecycle、Trellis panel、subagent tree cards和完成通知原语，但没有把它们组合为主 Agent 的长期 activity/blocked dashboard。

## 2. 关键时间线

| 时间（UTC） | 事件 | 对训练关键路径的影响 | 分类 |
|---|---|---|---|
| 8/26 01:03–11:58 | 初始 state-interface canary implementation 和一系列 real-data contract 修复；Git 从 `c4b2a357` 到 `21b6e669`。 | 建立数据、cache、FSDP、controller 基础，但还不是当前 Query-State formal 路线。 | 必要技术工作 + 早期设计偏差 |
| 8/26 13:56 | Job `532969` 提交，随后整夜 `PENDING`；约 8/27 05:49 才运行（由提交、8m15 runtime及05:57报告倒推，属估计）。 | 约 15h50 scheduler wait；不是 Agent 测试或审批时间。 | 基础设施等待（估计） |
| 8/27 05:49–05:57 | Job `532969` 运行8m15，因真实多轮prompt有4/5/6/7个结构block而在teacher forward前失败。 | 暴露production-shaped fixture/preflight缺口；无cache/checkpoint。 | 生产缺陷/返工 |
| 8/27 17:05–19:25 | 人类拒绝 broad canary，随后把重点从instruction改为Query token与DeepSight差异，并选择分阶段SFT1/SFT2路线。 | 旧全量cache+3epoch关键路径被废弃；这是实质scope reset。 | 必要需求纠正；早期规划失焦 |
| 8/28 06:41 | `f7ab6834`提交direct K16 Query-State训练路径。 | 当前SFT1路线的真实代码基线形成。 | 必要技术工作 |
| 8/28 10:03–14:59 | 中文prompt修复和canonical worktree迁移相关提交。 | 解决source isolation，但占用同一控制面并造成上下文切换。 | 旁支/基础设施 |
| 8/29 06:30–08:03 | `273b6806` production smoke准备，`6a0c59ef`修复CPU preflight。 | 为真实mechanics gate准备。 | 必要技术工作 |
| 8/29 09:25 | cwd/worktree大迁移后恢复；08-29 formal child task进入规划/实现。 | 需要重建source isolation和上下文。 | context switch/recovery |
| 8/29 17:58–18:07 | Fresh03 Job `537457`：错误VERL import provenance。 | 未进入FSDP/update。 | 生产环境缺陷 |
| 8/29 18:27–18:28 | Fresh04 Job `537492`：BF16 Qwen + FP32 direct head导致FSDP flatten失败。 | 需要source fix `b0f16070`。 | 生产composition缺陷 |
| 8/29 20:38–20:47 | Fresh05 Job `537574`：TensorDict被隐式当keys迭代。 | 未进入Qwen forward/update。 | 生产container缺陷 |
| 8/29 21:20–21:25 | Fresh06 Job `537588`完成两次真实update、checkpoint和fresh-process exact resume。 | mechanics gate首次通过。 | 有效里程碑 |
| 8/30 11:01 | `d0d5d62b`提交production Query-State training owner，9,131 insertions/117 deletions。 | 说明此前不存在可直接启动的formal owner。 | 必要大实现 |
| 8/30 11:24–14:31 | split、ID176 content identity、preflight/runtime row audit修复。 | 多项本可由更一致的production preflight提前发现。 | 必要修复 + preflight不足 |
| 8/30 14:31–17:27 | Pilot Jobs `538469/538512/538569/538580`依次暴露normalization、generation、checkpoint consensus、tracking dispatch；对应4个fresh source修复。 | 形成密集“短job发现一个缺陷”的串行返工。 | 主要延迟来源 |
| 8/30 17:38左右 | Retry11 Job `538625`完成14 updates、update7 fresh restore和terminal validation。 | 首次有效pilot；formal可开始锁定。 | 有效里程碑 |
| 8/30 20:09–22:30 | W&B query、统计合同、WS8早停、多节点topology gate提交。 | 部分为formal必要项；WS8/max10是human-selected replan。 | 必要formal准备 + scope change |
| 8/30 22:48 | Agent展示candidate并要求先确认，再称之后还要请求实际launch批准。 | 在已有standing authorization下形成重复审批点。 | 可避免审批摩擦 |
| 8/31 05:17 | 人类指出“已经重复授权很多次”。 | 终止夜间等待。 | 用户体验失败 |
| 8/31 05:28 | Job `539107`提交，2 nodes × 4 H800；提交时只有3 free GPU，仍PENDING。 | 当前阻塞变为Slurm queue；训练尚未开始。 | 基础设施等待 |

主要源码/实验事实来源：

- `.trellis/tasks/08-26-state-interface-v2-sft1-canary-exp/progress.md`；
- `.trellis/tasks/08-29-train-sft1-query-state/progress.md`；
- `.trellis/tasks/08-29-train-sft1-query-state/research/*-end.md`；
- Git commits `0b2fc5ca..b066dc3b`；
- raw Pi sessions `01a0398c-a369...`、`01a04471-7f1a...`、`01a04c54-3ff2...`、`01a05306-f434...`。

## 3. 延迟分解

### 3.1 必要技术工作：高占比

Git 在审计窗口内有34个SFT/Query-State相关提交，其中22个为`fix`、10个为`feat`。这说明任务不是“已有trainer只需点启动”，而是在连续完成：

- 新canonical state/objective；
- real-row/data/split/CoT合同；
- production mechanics；
- formal controller/checkpoint/W&B/validation；
- WS8 early-stop与2×4 topology。

不能把这些时间都叫“测试浪费”。但22个fix中相当一部分属于production integration遗漏，尤其是imports、dtype、TensorDict、runtime/preflight分叉、validation normalization和tracking mode；它们应更早在同一production-equivalence harness中成批暴露。

### 3.2 真实job与排队

Formal前共有13个已提交/取消的相关job（从`532969`到`538625`），已知runtime合计约85.5分钟；这不包括scheduler排队和preflight/修复时间。最显著的排队是首个Job `532969`整夜等待。当前Job `539107`也因8卡资源不足而等待。

结论：GPU实际运行时间不是五天的主体；**排队 + 每次短job只暴露一个缺陷 + fresh identity重建**才是成本。

### 3.3 审批等待

从raw Pi JSONL可复核的请求—回复样本：

| 请求 → 回复 | 等待 | 判断 |
|---|---:|---|
| 8/25 19:40本地commit请求 → 8/26 01:03“批准” | 5h22m57s | 当前Git策略要求批准；可通过implementation时一并授权commit减少 |
| 8/26 06:17 implementation请求 → 06:21“批准” | 4m23s | 正常 |
| 8/27 13:10 follow-up commit请求 → 13:11批准 | 10s | 正常 |
| 8/29 19:24 commit请求 → 19:35批准 | 10m22s | 正常 |
| 8/30 12:50 exact pilot launch请求 → 12:52批准 | 1m44s | 正常且必要 |
| 8/30 14:21 commit请求 → 14:31 standing authorization | 9m54s | 此后应减少逐次询问 |
| 8/30 22:48 Formal32 candidate确认请求 → 8/31 05:17用户回复 | 6h29m16s | 最大可避免摩擦；应先生成final lock，再只问一次exact launch |

必要保留的是**最终exact GPU launch approval**。可消除的是：

- 在最终lock之前询问candidate approval；
- commit和push拆成每个小修复逐次询问；
- 已有明确standing authorization后仍重复询问同一动作；
- 对不产生remote mutation的resolver/preflight要求人工停等。

### 3.4 测试与review

证据显示单次测试通常不慢：

- `67 passed in 5.50s`；
- `74 passed in 5.78s`；
- `104 passed in 9.68s`；
- `113 passed in 8.12s`；
- server production `104 passed, 1 warning in 57.20s`。

但2026-08-29至31审计到的Pi会话中，formal提交前约有：

- 20个`reviewer`子会话，累计open duration约0.62h；
- 7个`trellis-check`，约0.58h；
- 6个`trellis-implement`，约2.84h；
- 3个planner，约0.32h。

这些duration可能并行，不能相加为关键路径精确耗时；但20次narrow reviewer说明review粒度明显过细。部分review确实发现P1（如checkpoint partial mutation、统计合同、crash replay），不能取消；应合并为风险批次并去重suite。

### 3.5 上下文切换

worktree/storage工作解决了真实的canonical root、clean source和remote storage问题，因此不是纯浪费。但它与SFT控制面共用主会话，并在8月28日形成数小时明确Git活动窗口。用户在8月29日20:13还需要问“你是不是忘了我们在干什么、现在做到哪一步”，这是上下文可见性失效的直接证据。

## 4. 三个用户体验问题

### 4.1 Agent经常等待批准

判断：**体验真实；规则、Agent询问策略和GUI呈现共同造成。**

- 项目硬规则要求task implementation、commit、push、exact experiment launch分开审批。
- 人类离线时，任何一个未提前包络的门禁都会把关键路径挂起数小时。
- Agent没有在用户上线时一次性列出未来6–12小时可能出现的所有门禁，也没有把final contract准备到最后一步再问一次。
- pi-app支持结构化弹窗、挂起后“继续作答”、完成通知和queued follow-up，但当前Trellis流程没有把approval request提升为持续可见的`waiting-human`状态。

### 4.2 Agent花大量时间测试

判断：**“单次测试很慢”不成立；“测试/审查循环占用大量端到端时间”成立。**

测试真正必要的部分：real FSDP、global normalization、checkpoint/resume、generation safety、data split。可优化部分：每个小修复都重新运行重叠suite并启动独立review；preflight没有执行同一production path，导致测试没有在最便宜层捕获错误。

### 4.3 无法知道Agent在做什么

判断：**Pi/pi-app已有底层能力，但缺少面向目标的状态聚合。**

已确认能力：

- pi-app有streaming timeline、foldable read/edit/bash tool steps、Run/Context/Files等右栏和消息排队（`/workspace/pi-app/README.md`）。
- pi-app worker监听所有主会话`tool_execution_start/update/end`，不是完全看不到主Agent工具（`/workspace/pi-app/src/worker/worker-session-events.ts`）。
- pi-app内置Trellis只读side panel和`trellis_subagent` tree card；也有`pi-subagents` tree card。
- 项目`.pi/extensions/trellis/index.ts`为`trellis_subagent`提供更丰富的实时卡、elapsed、last tools和`Alt+O`。
- pi core extension API已提供`tool_execution_*`、`ctx.ui.setStatus/setWidget/setWorkingMessage`和streaming `onUpdate`，技术上可实现主Agent状态卡。

当前缺口：

- Trellis side panel只显示task status、description、children、acceptance criteria和最近journal，不显示`blocked_on`、当前phase step、正在跑的command/job、last progress、next approval。
- 项目特制实时卡主要覆盖`trellis_subagent`；主Agent虽有普通tool timeline，但没有“为什么做、还要多久、阻塞何处”的语义层。
- `in_progress`无法区分`working`、`waiting-human`、`waiting-VPN`、`waiting-Slurm`。
- `.pi/task-tree/`仍存在，违反项目“TaskTree保持空”的规则，并可能形成第二个误导状态面。

## 5. 根因树

### 直接原因

1. 初始需求/方案没有收敛到当前Query-State formal路线。
2. production training owner缺失，直到8月30日才实现。
3. production preflight与真实FSDP/runtime不等价，多个integration bug串行暴露。
4. formal预算最终又从WS2/2epoch改为WS8/max10 early-stop。
5. 当前8卡Job排队，训练仍未开始。

### 促成因素

- fresh immutable identity是正确安全设计，但每次小修复都放大成完整contract重建。
- review和测试按“每个bug一次”而不是按source batch组织。
- worktree/storage治理与SFT共用前台控制面。
- 人类离线前的授权未被形式化为机器可执行ask-before矩阵。

### 制度性根因

- 工作流只有task lifecycle状态，没有实时execution/blocker状态。
- 审批模型强调每一步单独确认，却缺少可审计的授权包络。
- 测试策略规定层级，但没有强制dedup/test-budget或production-equivalence preflight。
- Pi extension和pi-app的时间线/卡片原语没有被组合为“目标关键路径dashboard”。

## 6. 改进建议

### P0：立即采用，无需改代码

1. **每次长操作前发固定五行状态：**
   - `阶段`；
   - `当前动作/命令`；
   - `开始时间与预计耗时`；
   - `blocked_on`；
   - `下一次需要人类决定什么`。
2. **一次性离线授权包：** 用户离线前明确允许的branch、commit/push、CPU preflight、最多N次diagnostic retry、资源上限和必须询问的例外。正式formal launch仍保留一次exact approval。
3. **只在final immutable lock完成后问一次launch。** Candidate生成、hash、CPU preflight和non-mutating review都先自主完成；不要“先批准candidate，之后再批准launch”。
4. **测试预算回显：** 每轮先说明目的、预计时长、与上轮重叠部分、失败会阻断什么；完成后只报告新增覆盖。
5. **SFT关键路径单独前台lane。** worktree/storage/治理标为side work；除非它阻塞SFT，否则不得替代SFT状态汇报。
6. **使用pi-app现有能力：** 保持Trellis右栏打开；长运行时允许用户queue“查询进度”；确保完成/approval通知启用。但必须承认现有面板仍不展示blocker语义。

### P1：近期项目工作流/Trellis改进

1. 在experiment task中增加经人类审阅的`authorization_envelope`与`ask_before`矩阵；运行时严格只在越过包络时停。
2. 增加单一execution status：`working | waiting-human | waiting-vpn | waiting-slurm | blocked-evidence | completed`，附`since/current_action/next_gate`。它属于Trellis runtime，不复制到TaskTree。
3. 建立分层测试策略：
   - 每个修复：单一RED + focused；
   - source batch封版：affected/adjacent；
   - launch前：production-equivalence harness + 一次P0/P1 final review；
   - broad suite只在里程碑运行。
4. 合并review：普通窄修复一个owner+一个batch final review；data split、distributed checkpoint、generation safety、实验语义变化才单独review。
5. 把progress写入从“每个细小发现”合并到`fix validated / source published / job terminal / gate passed`四类checkpoint，减少隐形文档成本。
6. 自动生成launch contract、config hashes、command parity、output identity和approval summary，避免人工重复拼装。
7. 建立production-equivalence harness：使用相同server interpreter/import provenance、real TensorDict、实际dtype/model composition、同一validation normalization/checkpoint/tracking dispatch；需要CUDA的部分在一个获批hold allocation内用多个`srun`完成，避免每个缺陷重新排队。

### P1：近期Pi项目extension改进

1. 利用pi已有`agent_start/turn/tool_execution_*`事件，为主Agent增加activity widget/card。
2. 卡片字段至少包括`task/phase/current tool/elapsed/last progress/blocked_on/next gate/safe to interrupt`。
3. 将approval request作为显式等待状态并触发pi-app通知，而不是只发一条普通assistant消息。
4. 将Trellis panel数据扩展到current phase、checklist progress、latest evidence、remote job和blocker；保持只读。
5. 统一`trellis_subagent`与`subagent`的状态字段。pi-app已经有两类tree card，不需要从零实现，但应统一`run/blocked/elapsed/last tool`语义。

### P2：pi-app长期支持

1. 跨主会话/子代理的后台活动列表和可展开时间线。
2. 对长bash/SSH/Slurm显示last-output timestamp、完整elapsed、取消按钮和“可能在排队/无输出”的解释。
3. 常驻“等待你的操作”通知中心，支持从系统通知直接回到对应approval card。
4. Trellis panel原生显示关键路径和side-work lane，不读取或镜像Pi TaskTree。

## 7. 不应为了提速删除的门禁

- real CoT/state语义和不合成缺失CoT；
- exact data/split/overlap证据；
- source/config/output/checkpoint identity；
- formal GPU launch的最终exact人工批准；
- real FSDP optimizer/checkpoint/resume门禁；
- distributed normalization、rank consensus、generation safety；
- terminal evidence和`on-experiment-end`。

提速方向应是**把错误更早成批发现、把授权更早打包、把状态更清楚展示**，而不是降低验证真实性。

## 8. 证据缺口与限制

- 未访问远程Slurm；Job `539107`状态以本地最新task/monitor证据为准，可能在报告后变化。
- Pi JSONL能给出消息时间，但不能可靠分解Agent实际CPU时间、用户离开时间和后台subagent并行时间；session open duration不等于工作时长。
- 只量化了可明确配对的approval request/reply；其他“继续/可以”可能不是正式门禁回复，未纳入。
- 13个job的85.5分钟为已知runtime合计，不含未知queue时间。
- 未运行pi-app GUI或读取用户的Electron config-store；pi-app结论来自当前源码、README、adapter和Pi文档，不声称已验证用户实际启用了哪些通知/右栏选项。
- 没有把文件mtime当作事件因果证据；mtime只用于定位，结论由task、Git、job record和raw session交叉核验。

## 9. 最短结论

直到今天才提交formal SFT，不是因为pytest跑了五天，而是因为：**前两天重新定义了正确的state训练目标；随后才实现production trainer；真实FSDP路径连续暴露多个integration bug；每次修复都被immutable contract/review串行放大；期间夹杂worktree治理和两段可确认的离线审批等待。**

最大改进杠杆依次是：

1. production-equivalence preflight，减少“一个job发现一个bug”；
2. final-lock后只问一次launch + standing authorization envelope；
3. risk-based测试/review去重；
4. 主Agent/Trellis的`blocked_on`实时状态卡；
5. SFT关键路径与旁支任务分栏。
