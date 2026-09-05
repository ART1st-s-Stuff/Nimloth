# Progress — pi-app Trellis交互清理

## 2026-09-05

- 在pi-app独立worktree从`475c06b`实施完整cleanup。
- 删除旧Trellis dashboard reader/types/model/panel、typed approval各层分支及全局pi-task-tree adapter/panel/model/docs/tests；保留Task Browser、Work-item Activity、generic workspace-json/dispatch和generic extension UI。
- Trellis adapter改用`renderer-owned`无状态provider与专用`trellis`component。
- 首次独立review发现unsupported custom伪装空questionnaire（P1）和input/editor suspend丢draft（P2）；修复为无provenance fail-closed，并按workspace/session/request/tool-call保存draft至terminal outcome。
- 最终独立review`APPROVED`，无P0–P2。
- Focused remediation：11 files/50 tests；full build、Node TS、lint通过。Full unit仅既有rewind失败；Web TS仅clean parent同一`fluent.tsx:107 TS2742`。
- 人类批准45文件patch（SHA-256 `a9d6a8f203e29734abefc7196e91c0eea1976880cdeffb15a754a242d26cf518`），提交`87629bae2dfca0bf4335c95c030efa4e5e9ce0cd`。
- 人类批准并完成FF到pi-app `task/rebuild-nimloth-trellis-prompts`；post-merge 6 files/21 tests通过，integration clean。未push、未合入默认branch、未cleanup worktree。
