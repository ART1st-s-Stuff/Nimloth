# Progress — 独立Nimloth work-item可见性

## 2026-09-04

- 从Nimloth parent `3caab1f53a8f`和pi-app parent `15618da97448`创建独立同名task branches/worktrees；Nimloth child `.local`指向canonical `.local`。
- 两仓并行完成W2–W7：Nimloth producer实现普通checkbox identity、bounded atomic runtime、动态`nimloth_work_item`工具及只读projection；pi-app实现registered-worktree reader、strict IPC和稳定Activity consumer。
- 跨仓核验发现renderer原始Pi session ID与Trellis `pi_<sessionId>` context key不一致，已改为Main从bounded `sessionId`派生context key并补回归。
- 首次独立review结论`NEEDS CHANGES`：inactive task current、Activity单次snapshot、auto-discovered skill metadata和producer/consumer bounds共3项P1、1项P2。
- 并行修复：inactive current suppression、5秒non-overlapping polling、移除static skill、tool-only动态guidance、完整projection-v1 bounds；随后补齐`linesTruncated`、严格timestamp和160→163字符session/context边界。
- 最终独立review：`APPROVED`，无P0–P2。
- Nimloth最终focused：28/28 extension tests；policy 5/5、upstream 2/2、context measurement、task validation、diff check通过。
- pi-app最终focused：24/24；Node TypeScript、affected ESLint、build、CRLF-aware diff check通过。Full unit只剩既有rewind失败；Web TypeScript只剩clean parent可复现的`fluent.tsx:107 TS2742`。
- 两仓均无staged files；下一步展示完整diff并请求分别local commit批准。未push、未merge、未cleanup worktrees。
