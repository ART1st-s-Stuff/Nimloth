# Legacy archive最终post-move验证（2026-09-05）

## Bound operation

- Move manifest：254 rows，SHA-256 `990a4bbb907e156c87b1e6301899a8de9049502505110fa0a5b2a11aa078e465`。
- Rewrite patch：18 files / 28 replacements，SHA-256 `ff0075f148dac52c5632c9c9a30d73d17e9a575ae70ee52112a4e940ce923716`。
- Audit：33 retain / 117 archive-only / 2 obsolete / 4 needs-human；不确定项E0066、E0068、E0072、E0147未进入mandatory spec。

## Result

- 157 known-error tree、95 ai_tasks和2 root history文件均为R100 rename。
- 所有source absent、destination SHA-256与Git blob匹配。
- Active context 19处、eval provenance 4处、SFT1 Slurm provenance 5处已按reviewed whole-file hash改写。
- Active specs/skills/source不再把legacy路径作为current authority；archive仅供traceability/provenance。

## Validation

- Archive-plan tests：11/11。
- Post validator（含active contexts、Markdown links和focused launchers）：PASS。
- Nimloth policy：5/5；upstream fidelity：2/2。
- Task context validation：9 implement、8 bounded check entries，无context-size warning。
- `git diff --check HEAD`及cached diff check：PASS。

## Reviews

- Pre-move reviews修复retain mapping、destination symlink、active refs/post validation、provenance、hash gate和rollback。
- Post-move首次review修复test mode。
- 最终post-move remediation review：`APPROVED`，无P0–P2。
- Artifacts：`legacy-pre-move-review.md`、`legacy-pre-move-remediation-review.md`、`legacy-final-hash-gate-review.md`、`legacy-post-move-review.md`、`legacy-post-move-remediation-review.md`位于当前Pi session outputs。
