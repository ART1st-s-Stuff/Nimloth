# Known-Error Routing

Known errors are confirmed failure patterns under [`ai_rules/known_errors/`](../../../ai_rules/known_errors/). They are live evidence, not a task system or a replacement for specs.

## Selection process

During planning and again during full-scope checking:

1. List touched paths, concepts, interfaces, data/checkpoint/runtime boundaries, and planned commands.
2. Open the [categorized index](../../../ai_rules/known_errors/README.md).
3. Search filenames/index by those terms and nearby synonyms.
4. Read individual candidate files. Add only genuinely relevant entries to `implement.jsonl`/`check.jsonl` or task research, with a reason tied to the planned change.
5. Recheck the selected failures against the final diff and validation evidence.

Do not inject the whole directory. Large undifferentiated context dilutes relevant constraints. Do not treat a filename as the full rule; read the entry and its evidence.

## New entries

Add an `E*.md` only after a failure or misjudgment actually occurred and was confirmed. One file records one pattern: incorrect conclusion/action, cause, correct practice, and concise evidence. Do not create speculative known errors, rewrite old entries for style, or use the library as progress logging.

When a confirmed error reveals a stable mandatory contract, update the owning spec as well. Keep the known error as historical evidence of the failure; avoid duplicating long prose.
