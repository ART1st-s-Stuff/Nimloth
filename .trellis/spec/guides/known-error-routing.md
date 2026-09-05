# Historical Failure Evidence

Current mandatory contracts live in the owning scoped specs, not in incident records. Pre-Trellis known-error files are immutable historical provenance being moved to the legacy archive; they are not automatically loaded or routed as current instructions.

## Investigation process

During planning and full-scope checking:

1. List touched paths, concepts, interfaces, data/checkpoint/runtime boundaries, and planned commands.
2. Read the owning scoped specs and inspect current source and focused tests.
3. Consult an archived incident only when the task needs that explicit historical provenance.
4. Record task-specific evidence and unresolved uncertainty in the current Trellis task.
5. Recheck the selected current contracts against the final diff and validation evidence.

Do not inject the incident archive or infer a current rule from an incident filename. If current evidence is insufficient, preserve that uncertainty rather than promoting the incident narrative into a spec.

## New failures

Do not add new active `E*.md` files. Record a confirmed task-local failure in Trellis research or checks. If it reveals a stable mandatory contract, add only the deduplicated, executable rule to the owning scoped spec; historical narratives remain provenance rather than live policy.
