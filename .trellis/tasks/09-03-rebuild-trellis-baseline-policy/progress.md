# Progress

## 2026-09-04 — A1上游0.6.16恢复已提交

- 人类批准精确34个staged文件；上游一致性unittest与cached diff check通过。
- 创建local commit `c1bd6414e98213b6d557c736a036f065effa7856`：`chore(trellis): restore upstream 0.6.16 baseline`。
- Commit只包含A1；A2删除、A3 tests/research、task记录及`trellis-update-spec`漂移保持未暂存。
- 未push、未merge。

## 2026-09-04 — A2删除候选审查完成

- 精确范围为11个旧typed approval/dashboard/execution/work-item模块、fixture及tests，共删除2804行。
- 在active `.agents/.claude/.codex/.pi/.trellis/scripts`中排除历史task/archive后，相关module/test/fixture引用为0。
- 恢复后的`task.py --help` smoke通过；删除patch diff check通过。
- 删除patch：`/tmp/nimloth-trellis-baseline-a2-deletions.patch`，SHA-256 `4e7c43affb5f459b8bac420263b4bfc790df9be4f19effde1a098156147fb1e0`。
- 尚未stage或commit；live `.trellis/.runtime` request/receipt数据不在本范围。

## 2026-09-04 — A2/A3提交与政策覆盖层完成

- A2经破坏性范围批准后提交为`432d9daef14cc6c60b0fce7b353233998e2c7994`，仅删除11个零active引用的legacy framework文件。
- A3经TDD commit批准后提交为`54ee00813a6cf147bbc6e2b9833785f590085a26`，仅含3个baseline/policy/context tests。
- 人类批准精确49行薄`AGENTS.md`；随后完成Git、实验、authority、task/progress、platform integration specs和五个操作skill的窄覆盖。
- Policy 4/4、upstream baseline 1/1通过；54个相对Markdown links有效；context为1750/3822/3766 B且marker各出现1次。
- `trellis-update-spec`畸形更新漂移已恢复到HEAD/template hash；dry-run仍因`.claude/skills`父目录symlink alias而误报regular `SKILL.md`为modified，记录为上游false-positive。
- 人类审查并批准13个spec/skill文件及薄`AGENTS.md`，政策覆盖层已提交为`18c3a51ad7627d7d4c63f0d2fb5dedffe3f79b1b`。

## 2026-09-04 — Final review P1 remediation

- 独立reviewer发现短实验审批、AGENTS领域边界、authority/task-consent和upstream test覆盖共4项P1；未发现protected/runtime/archive/pi-app越界。
- 按parent PRD修正：qualifying短复现无需额外launch询问；AGENTS移除CoT/state规则；spec高于task artifact；无active task先取得upstream task-creation consent。
- Upstream test新增固定0.6.16 manifest，锁定144个managed files的content hash和executable bit，并继续直接比较workflow/task.py/Pi extension与release package。
- 修复采用RED/GREEN：policy 5/5、upstream baseline 2/2通过。
- 第二次独立reviewer判定`APPROVED`，无P0–P2；完整Python contracts 7/7、context fixture、task validation、140个全局Markdown links和diff checks通过。
- 人类批准P1 correction精确11文件，提交为`f2c3add3a81156f37d10af6fd5dd31ae554880ba`；task记录继续保持独立未提交。
- Child实施、验证、spec审查和完整diff审查现均完成；下一步仅提交9个task/progress/research记录，然后进入parent fast-forward merge门禁。
