# Implementation Plan — read-only visibility research

## 1. Establish source baselines

- [x] 记录Nimloth与pi-app实际root、branch、dirty status，保持所有既有修改。
- [x] 记录Pi/pi-app/Trellis版本与相关adapter配置。
- [x] 确认本任务不需要实验、远程访问、GUI mutation或受保护文件。

## 2. Trace the static Trellis path

- [x] 从`.trellis/tasks/*/task.json.status`追踪到pi-app workspace task reader。
- [x] 追踪reader输出到Trellis adapter和side panel的status label/render逻辑。
- [x] 列出当前侧栏实际读取的字段、刷新触发和stale行为。

## 3. Trace live activity surfaces

- [x] 完整阅读任务相关Pi extension/TUI文档及其交叉引用。
- [x] 追踪Pi主session的`tool_execution_start/update/end`到pi-app worker与timeline。
- [x] 追踪项目`trellis_subagent`和`pi-subagents`adapter的run/progress/elapsed数据。
- [x] 检查通知、queued follow-up、approval和长命令状态现有能力。
- [x] 确认哪些语义没有生产者，避免把UI缺口和数据缺口混为一谈。

## 4. Compare designs

- [x] 比较static enrichment、runtime projection、worker aggregation和hybrid。
- [x] 验证`ctx.cwd`、root/session isolation、restart/stale和backward compatibility约束。
- [x] 选择一个MVP并定义schema、owner、transport、UI和failure behavior。
- [x] 拆分Nimloth与pi-app实施任务、依赖和验证门禁。

## 4.5 Replan for task-tree work-item visibility

- [x] 追踪`task.json.parent/children`、`implement.md`checklist和active session pointer的真实所有权。
- [x] 审计现有任务计划是否有稳定work-item ID、结构化parser或runtime cursor。
- [x] 比较显式Markdown ID、derived ID和sidecar index的兼容/权威风险。
- [x] 定义task/work-item/executor/run绑定、状态机、委派转移、stale和冲突行为。
- [x] 设计能展开任务树与任务清单、突出当前item及证据的pi-app UI。
- [x] 修订报告，撤销“event aggregation即可作为MVP”的结论。
- [x] 重新请求人类审查修订后的端到端实施建议。

## 5. Report and verify

- [x] 写入`research/pi-app-trellis-visibility-design-2026-08-31.md`初版。
- [x] 为主要事实提供文件/symbol引用，并区分事实、推断和建议。
- [x] 运行task validation、路径存在性检查和task范围空白检查。
- [x] 确认没有本任务目录外的审计产生修改。
- [x] 向人类展示初版推荐MVP；人类已明确拒绝缺少work-item定位的方案。
- [x] 完成修订报告并重新运行全部验证。

## Approval gates

- 人类已批准创建只读调研任务。
- 完成本计划和context清单后，仍需一次明确的“开始调研”批准，再运行`task.py start`。
- 本任务不包含任何Nimloth/pi-app代码修改、commit、push、远程操作或实验。
- 后续实施必须另建或拆分Trellis任务并取得各仓库implementation approval。

## Validation commands

```bash
python3 ./.trellis/scripts/task.py validate .trellis/tasks/08-31-research-pi-app-trellis-visibility
python3 ./.trellis/scripts/task.py current --source
git diff --check -- .trellis/tasks/08-31-research-pi-app-trellis-visibility
find .trellis/tasks/08-31-research-pi-app-trellis-visibility -maxdepth 2 -type f -print | sort
```

## Validation evidence

- `task.py validate`通过：implement/check JSONL各8条。
- 最终报告390行、16,053 bytes；task文本无尾随空白。
- Nimloth仅本任务目录有本次调研新增内容；pi-app既有dirty status前后不变。
- 未运行GUI、tests、typecheck或build；这些属于后续实施验证，不作为本次只读调研完成证据。
