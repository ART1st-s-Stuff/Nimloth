# pi-app cleanup最终检查（2026-09-05）

## Delivered

- `renderer-owned` Trellis panel registration；Documents/Activity不依赖legacy state。
- 删除dashboard、typed approval和pi-task-tree专用实现。
- 保留generic extension UI、workspace-json及adapter dispatch。
- Unsupported custom无generic provenance时返回`system-cancelled/unsupported-custom-ui`且不产生UI。
- Input/editor draft按完整attention identity持久到respond/decline/system cancel。

## Validation

- 实施focused：15 files/79 tests。
- Remediation focused：11 files/50 tests。
- Build、Node TypeScript、affected ESLint、JSON parse、residual absence和CRLF-aware check通过。
- Full unit：1095 passed、1 skipped，仅既有rewind test失败。
- Web TypeScript：clean parent与feature均为`fluent.tsx:107 TS2742`。
- 独立review：首次`NEEDS CHANGES`（1 P1、1 P2），修复后`APPROVED`。

Review artifacts：
- `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/pi-app-cleanup-review.md`
- `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/pi-app-cleanup-remediation-review.md`
