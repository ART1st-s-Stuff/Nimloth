# 实施计划 — Legacy知识归档

## W1 — Inventory RED
- [x] 固定156 E文件、157 known-error tree、95 ai_tasks files及两个root历史文件的完整manifest/count/hash。
- [x] 固定duplicate prefix、destination collision、active reference和hash-preserving tests。

## W2 — 四组并行审计
- [x] 按排序路径1–39、40–78、79–117、118–156逐文件复核正文及当前source/spec证据。
- [x] 严格reconcile为33 retain-contract、117 archive-only、2 obsolete、4 needs-human-decision；每个retain均有机器验证的owning spec/unique anchor/exact excerpt，四个deferred case无mapping。

## W3 — Spec与引用GREEN
- [x] 按owner去重核验合同；implementation/version/machine-local规则不扩写spec，单独生成完整spec diff。
- [x] 冻结hash-bound exact rewrite patch：19处active context、4处eval test和5处Slurm provenance；pre-move不应用post-move路径。
- [x] 扩展forbidden active-reference、current active context和link validation。

## W4 — 破坏性move门禁
- [x] 准备156+ai_tasks+root文件的精确source→destination manifest、SHA/blob、命令和影响；move count固定254。
- [x] harden preflight：拒绝symlink/non-directory/escape，要求全部destination absent，首个move前预创建并复核全部目录，每次mutation前再检查。
- [x] 准备post mode验证destination SHA/blob、source absence、active refs/context、Markdown links与focused tests。
- [x] 准备deterministic partial rollback工具并保持dry-run。
- [x] 经人类精确批准后执行hash-bound `--execute-approved-moves`；254个R100 move和18文件rewrite通过post验证。

## W5 — 最终检查
- [x] 运行manifest/archive contracts、Markdown links、launcher/static tests、Trellis policy/upstream checks及diff check。
- [x] 独立review完整spec/ref/move范围；最终post-move复审无P0–P2，commit/merge/push仍需对应门禁。

## Guardrails
- 不修改memory JSONL、template hashes、runtime、实验输出或`.trellis/tasks/archive/`。
- 不运行实验/remote/Slurm，不重写历史正文。
- 不覆盖并发dirty task artifacts；所有move必须等待精确破坏性批准。
