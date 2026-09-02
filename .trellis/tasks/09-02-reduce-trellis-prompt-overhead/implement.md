# Implementation Plan

## Context RED/GREEN

- [x] [W-001] 添加当前consumer fixture的context bytes、重复正文和delta行为RED测试。
- [x] [W-002] Main改为compact task locator与fingerprint delta，移除完整artifact/JSONL自动注入。
- [x] [W-003] Child context改为artifact单次内联＋canonical JSONL索引，修复absolute path与重复项。

## Prompt/workflow simplification

- [x] [W-010] 审计Pi/channel prompt重复加载来源并确认generated-file边界。
- [x] [W-011] 在`AGENTS.md`、workflow和governance spec定义fast/standard/high-risk、授权复用和验证复用。
- [x] [W-012] 收紧`on-progress`与项目workflow，合并同item记账并取消小修仪式。
- [x] [W-013] 恢复7个pristine generated agent/prompt文件，并由child dispatch contract禁止已提供artifact重读。

## Check

- [x] [W-020] 运行extension/context、task validation和prompt一致性测试。
- [x] [W-021] 输出修改前后bytes/token对比并确认高风险门禁未弱化。
- [x] [W-022] 独立完整diff审查一次；根据已批准commit policy提交，不重复运行已通过且输入未变的检查。

## Guardrails

- 不修改全局Trellis npm包或`node_modules`。
- 不触碰memory JSONL、Pi TaskTree、实验/远程状态或其他task dirty changes。
- 不用摘要冒充缺失合同；Agent可按需读取路径。
- 只在权限/风险边界变化时询问人类。
