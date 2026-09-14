# Execution
1. Implement and test full-language mode, freeze scope, optimizer membership, checkpoint identity/export/resume.
2. Inspect dtype/FSDP viability and run focused CPU tests.
3. Commit scoped changes, sync remote dedicated worktree; check input lineage and output uniqueness.
4. Launch 8GPU comparison from original Stage1-derived input, confirm finite actual updates and save/reload mechanics.
5. Record real progress and checkpoint paths. Preserve old runs. Old run paused at committed step201; exit75 is intentional.
