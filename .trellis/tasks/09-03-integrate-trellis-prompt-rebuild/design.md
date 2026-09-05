# Design — 最终集成

1. 所有检查和修改绑定既定Nimloth/pi-app integration worktree；canonical只读检查状态。
2. 先核验child commit ancestry与integration clean状态，再运行受影响测试，避免重复full suite。
3. 用symbol/reference scans确认删除边界，用已有focused tests确认generic transport和只读UI仍工作。
4. 将child task completion与parent checklist按证据更新，任务状态不先于merge事实。
5. 最终review只读检查两仓committed diff与跨层数据流。
6. `dev` dirty时禁止merge、stash、reset或update-ref；保留parent branch并报告精确blocker。
