# Implementation Plan — read-only workflow audit

## 1. Establish baseline

- [x] 记录当前日期、canonical cwd、branch 和完整 dirty status。
- [x] 记录当前相关 Trellis task 状态与父子关系。
- [x] 建立证据清单，不读取明显无关或受保护的大型 payload。

## 2. Reconstruct the critical-path timeline

- [x] 阅读 8 月 26 日 SFT/state-interface 父任务及其研究、进度、实验合同。
- [x] 阅读 8 月 29 日正式 SFT1 query-state 训练子任务及其研究、进度、实验生命周期证据。
- [x] 检查 8 月 27 日 storage 审计与 8 月 28 日 worktree 重构中和 SFT 关键路径有关的部分。
- [x] 检查 8 月 26 日以来相关 Git commits 与 workspace journal。
- [x] 使用 `trellis mem` 定位会话中的审批请求、等待、测试循环、任务切换及训练启动决定，并用文件证据交叉核验。

## 3. Analyze workflow and platform behavior

- [x] 对关键路径事件进行延迟分类，区分直接原因、促成因素和制度性根因。
- [x] 汇总测试命令、重复执行和可识别的耗时；无法取得 duration 时避免伪精确量化。
- [x] 统计或列举关键审批门禁，判断哪些是项目硬规则、哪些可通过批量授权或 ask-before 改善。
- [x] 完整读取与任务相关的 Pi 文档及交叉引用，检查 Pi/pi-app 当前进度、异步运行和阻塞可见性能力。

## 4. Produce and verify the report

- [x] 写入 `research/workflow-audit-2026-08-31.md`，为主要结论给出文件、commit、session 或日志引用。
- [x] 检查结论是否把事实、估计、用户体验和建议分开。
- [x] 核对验收标准，运行任务验证与 `git diff --check`。
- [x] 确认除本任务目录外没有由本审计产生的改动。
- [x] 在对话中汇报结论、证据缺口和优先行动。

## Approval gates

- 当前只批准了任务创建和规划。
- 开始上述广泛只读研究前，需要人类批准本计划并运行 `task.py start`。
- 不包含实验 launch、远程操作、源码修改或 commit；若后续建议需要实施，必须另行规划和批准。

## Validation commands

```bash
python3 ./.trellis/scripts/task.py validate .trellis/tasks/08-31-audit-sft-debug-training-latency
python3 ./.trellis/scripts/task.py current --source
git diff --check -- .trellis/tasks/08-31-audit-sft-debug-training-latency
find .trellis/tasks/08-31-audit-sft-debug-training-latency -maxdepth 2 -type f -print | sort
```

## Validation evidence

- 2026-08-31：`task.py validate`通过，implement/check JSONL各6条。
- 报告273行；task文本无尾随空白；引用的主要task/Pi/pi-app路径均存在。
- `git diff --check -- <task>`通过；本任务为untracked目录，另以逐行空白检查覆盖新报告。
- 审计只新增/修改本任务目录；没有执行远程命令、实验、commit、push或cleanup。
