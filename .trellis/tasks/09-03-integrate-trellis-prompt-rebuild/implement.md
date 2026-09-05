# 实施计划 — 最终集成

## W1 — 集成inventory
- [x] 核验两仓integration SHA、child ancestry、clean状态和完整commit顺序。
- [x] 补齐已完成child的任务状态、commit和parent checklist，不伪造未完成门禁。

## W2 — Nimloth最终检查
- [x] 运行Trellis policy/upstream、work-item、archive、extension/static及context/diff检查。
- [x] 扫描active typed approval、TaskTree、legacy routing和已迁移路径残留。

## W3 — pi-app最终检查
- [x] 运行Review、Task Browser、Work-item Activity、generic extension UI及absence focused tests。
- [x] 运行typecheck/build/lint并隔离任何clean-base既有失败。

## W4 — 最终review与merge
- [x] 独立review两仓完整committed diff和跨层数据流；snapshot remediation后最终APPROVED，无P0–P2。
- [ ] 展示最终SHA/diff/验证；仅在`dev`目标安全时非force合入并做post-merge smoke。

## Guardrails
- 不push、不force、不cleanup worktrees、不删除live runtime或修改memory JSONL。
- 不运行实验、GPU、Slurm或remote job。
- 不覆盖canonical `dev`的并发dirty changes。
