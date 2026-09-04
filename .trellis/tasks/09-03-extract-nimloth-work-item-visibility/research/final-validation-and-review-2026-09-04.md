# Work-item visibility最终验证与复审（2026-09-04）

## 实现边界

- Nimloth：`.pi/extensions/nimloth-work-items/`中的唯一parser/runtime/projection和动态tool。
- Runtime：`.local/nimloth-work-items/v1/<root-hash>/<context>.json`，atomic、bounded、非权威，不持久化plan/checkbox/CoT/tool output。
- 无auto-discovered feature skill；操作说明只存在于动态active tool description/result。
- pi-app：registered-worktree-only Main reader、strict IPC、稳定Activity view和5秒non-overlapping polling。
- 未修改上游Trellis managed files、Pi core、TaskTree/approval authority或legacy runtime。

## 关键修复

1. Renderer发送Pi `sessionId`，Main派生上游格式`pi_<sessionId>`；producer/CLI接受最大163字符context key。
2. task非`in_progress`时projection强制`current=null`并报告`inactive-task`。
3. Activity轮询保留上一snapshot、单generation最多一个in-flight request，并拒绝source/session旧响应。
4. Projection v1对item、heading、count、issues、duplicate lines、task/status、executor/session、runtime字段和timestamp实施同值边界；`linesTruncated`是显式schema字段。

## 验证

- `node --test .pi/extensions/nimloth-work-items/*.test.mjs`：28/28。
- `test_nimloth_policy_overlay.py`：5/5；`test_upstream_trellis_baseline.py`：2/2。
- context measurement：main 1750 B、implement 3822 B、check 3766 B。
- pi-app focused set：5 files、24 tests；独立review额外相邻set 29/29。
- pi-app Node TypeScript、affected ESLint、production build通过。
- Full unit：1131 passed、1 skipped，仅既有rewind test失败。
- Web TypeScript：feature和clean parent均为`fluent.tsx:107 TS2742`。
- CRLF-aware diff check：0个真实whitespace issue。

## 独立review

- 首次：`NEEDS CHANGES`，3 P1 + 1 P2。
- 修复后：`APPROVED`，无P0–P2。
- 报告：
  - `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/work-item-visibility-final-review.md`
  - `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/work-item-visibility-remediation-review.md`

## 残余限制

- 未运行live Electron/WSL UI smoke；自动测试覆盖host/WSL invocation、polling、generation和降级。
- 两个既有pi-app baseline failure不属于本diff，未降低检查强度。
