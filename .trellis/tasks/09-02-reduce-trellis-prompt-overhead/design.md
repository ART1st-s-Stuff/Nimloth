# Design — bounded context and proportional workflow

## 1. Context ownership

### Main session

System prompt只保存稳定安全规则和compact task locator：taskRef/status、goal摘要、当前work-item、artifact hashes与路径。全文由Agent在真正需要时读取，不在SessionStart内联。

每轮只比较结构化fingerprint：`taskRef/status/workItemRef/artifactHashes/approvalState`。fingerprint未变不发消息；变化时只发送这些字段，不发送全文。Git status与heartbeat不得触发prompt delta。

### Trellis child

`trellis_subagent` prompt包含：agent definition、delegated instruction、task locator、完整PRD/design/implement各至多一次，并在agent definition之后明确声明其context-loading步骤已由dispatch满足。`implement/check.jsonl`只转成canonical path/reason/size/hash索引；不内联文件正文。Agent读取与实际修改/审查相关的条目，不机械遍历全部索引。

统一resolver先处理absolute/relative path，再以realpath/canonical path去重。task artifact与JSONL重复时artifact胜出。

## 2. Workflow risk classes

- **Fast path**：approved scope内的小修；无新task、subagent、额外approval、full suite、逐次progress。
- **Standard**：多文件或公共合同变化；一个task、一次实施审查、一个最终check批次。
- **High risk**：实验/远程/破坏/protected/push-merge；保留现有精确门禁。

直接人类prompt可授予task创建、implementation或local commit权限；授权按明确scope持续到目标、风险或排除项实质变化。内部记账、只读研究、local test和approved edit不是新的权限边界。

## 3. Verification economy

验证证据以`command + relevant-input fingerprint`识别。无相关输入变化时复用；失败后修复只重跑失败/受影响检查。Implement负责focused checks；Check负责语义review和缺失证据；完整suite只在最终批次运行。

## 4. Prompt synchronization

修改项目本地权威，不修改npm安装目录：

- `AGENTS.md`：最小充分验证、授权复用与fast-path硬规则；
- `.trellis/workflow.md`及governance spec；
- `.pi/extensions/trellis/index.ts`及其tests：compact/delta context；
- `.agents/skills/on-progress/SKILL.md`。

此前pristine且受template hash管理的`.pi/agents/trellis-{implement,check}.md`、`.trellis/agents/{implement,check}.md`和`.pi/prompts/trellis-{start,continue,finish-work}.md`保持当前`HEAD`原版；不为本项目规则制造新的prompt fork。

## 5. Rollout

先锁定context-size/duplication RED tests，再修改injection；随后修改prompt/workflow文本并做一致性扫描。旧task/JSONL继续兼容，不批量迁移。若compact context缺少必要信息，Agent按path显式读取，不恢复全量自动注入。
