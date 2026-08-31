# Trellis Work-item Runtime Consumer Contract

## Read path

Call from the project root and pass the selected Pi context explicitly:

```bash
python3 ./.trellis/scripts/task.py dashboard --json --context pi_<session-id>
```

The command is read-only. It returns dashboard schema major `1` even when plans or runtime files contain errors; consumers must inspect top-level `valid` and render every typed entry in `issues`. Do not infer a current item, parse Markdown independently, or read `.pi/task-tree/`.

Stable fixture: [tests/fixtures/dashboard-v1.json](tests/fixtures/dashboard-v1.json).

## Dashboard-v1 shape

- `schemaVersion`, `kind`, `generatedAt`: projection identity.
- `root.path/fingerprint`: resolved project identity; cache keys include this fingerprint plus context key.
- `selected.contextKey/taskRef`: selected runtime context and its session task pointer.
- `taskTree.roots/nodes`: `task.json` parent/children/status projection, including archived children.
- `tasksByRef.<task>.plan`: sections, ordered items, checkbox-only plan state, stable/legacy identity and parse issues.
- `tasksByRef.<task>.review`: raw UTF-8 `prd.md`/`design.md`/`implement.md` with original line endings, SHA-256 of exact file bytes, heading sections, deterministic review-set hash and changes from the latest bound request. Unreadable/non-UTF-8 artifacts produce a typed issue and cannot enter an approval request.
- `execution`: schema version, 10-second heartbeat interval, 30-second stale threshold, per-context projections and flattened active assignments.
- `approvalRequests`: typed request identity, scope/exclusions/validation commands, current hash validity, changes and optional receipt.
- `issues`: duplicate/malformed plan IDs, missing child/cycle, malformed runtime, orphan/stale/conflict and stale approval request issues.

Unknown schema majors must be rejected by the consumer rather than guessed.

## Producer CLI

Pi normally calls the `trellis_work_item` tool. Other local integration tests may invoke the underlying CLI:

```bash
python3 ./.trellis/scripts/task.py work-item select \
  --context pi_<session-id> --task <task-dir-name> --item W-010 \
  --role primary --executor-kind main --session-id <session-file> --agent main
python3 ./.trellis/scripts/task.py work-item update \
  --context pi_<session-id> --assignment <assignment-id> --state verifying
python3 ./.trellis/scripts/task.py work-item block \
  --context pi_<session-id> --assignment <assignment-id> \
  --state waiting_human --blocker approval
python3 ./.trellis/scripts/task.py work-item evidence \
  --context pi_<session-id> --assignment <assignment-id> \
  --evidence-kind test --ref unittest:<test> --summary "passed"
python3 ./.trellis/scripts/task.py work-item release \
  --context pi_<session-id> --assignment <assignment-id>
```

Runtime writes are atomic under `.trellis/.runtime/execution/<context-key>.json`. These commands never edit `task.json`, planning artifacts, checkboxes, memory or Pi TaskTree.

## Typed approval gate

Pi Agents normally use `trellis_approval` with an explicit task, gate kind, scope and exclusions. The tool first persists the hash-bound request, then uses Pi TUI select/input or an explicit Desktop `ui.custom(..., { trellisApproval: ... })` payload. It accepts only an exact root/context/session/toolCall/request/task/kind/artifact/review response, records the receipt, and reads it back through `validate-approval` before returning. Unsupported headless/custom transports, system cancellation, late/mismatched responses and artifact changes fail closed; the tool never calls `task.py start`.

The underlying CLI remains available for local integration tests: create a request with `work-item request-approval` and explicit repeated `--scope`, `--exclusion` and `--validation-command` values. Record `approve`, `decline` or `comment` with `record-approval`; verify an approve receipt with `validate-approval` immediately before the gated operation. Validation binds root, context/session/request, task, kind, scope/exclusions and current planning artifact hashes. `record-approval` rechecks current artifacts before writing, so a changed artifact cannot persist a stale receipt.
