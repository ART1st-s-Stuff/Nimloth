# Progress

## 2026-09-02 — bounded context and proportional workflow implemented

- Replaced Main's full task/artifact/JSONL system injection with a compact locator containing task/work-item, artifact hashes and approval counts; later changes emit a bounded hash delta and per-turn Git/session overview churn is no longer appended.
- Replaced child JSONL body injection with canonical path/reason/size/hash indexes. Task artifacts are supplied at most once, absolute/relative/symlink duplicates collapse, and missing/truncated entries remain visible.
- Same automated fixture: legacy body payload 54,870 bytes; Main locator 615 bytes; child context 21,658 bytes; artifact delta 165 bytes. The integrated Main prompt assertion is <=16 KiB and child assertion <=32 KiB.
- Added Fast/Standard/High-risk contracts, reusable authorization/evidence, one final validation batch and consolidated progress in AGENTS, project workflow, governance specs and `on-progress`.
- After human review of ownership, restored the seven previously pristine generated Pi/channel agent and start/continue/finish prompt files to their recorded template hashes. The child context now explicitly says the generated agent's artifact-loading steps are already satisfied, so no generated prompt fork is needed.
- Preserved separate experiment launch, remote/destructive, protected-data and push/merge gates. Did not modify npm/global Trellis, memory, Pi TaskTree, experiment state or unrelated dirty work.
- Validation: extension safe probe passes; 15 Python work-item tests pass; task validation, prompt consistency, high-risk gate scan and diff-check pass. Independent final review found no P0/P1 and changed no files. Live `/reload` probe remains for the user/current session boundary.
