# Final review remediation — 2026-09-04

## First review

Independent reviewer returned `NEEDS CHANGES` with four P1 findings: short runs incorrectly required a launch question, thin `AGENTS.md` retained CoT/state policy, authority/task-creation semantics contradicted the parent PRD, and upstream fidelity tests covered too few managed files. Task records also needed current commit/checklist state.

## Corrections

- Qualifying unchanged remote reproduction/partial rerun expected within 10 minutes launches from its reviewed lightweight experiment task without another question; longer/materially new runs require explicit launch approval.
- Removed CoT/state implementation policy from `AGENTS.md`; project specs now precede reviewed task artifacts.
- No-active-task behavior now follows upstream task-creation consent, including simple/trivial code work.
- Added fixed `@mindfoldhq/trellis@0.6.16` manifest for 144 managed paths, content hashes and executable flags. `.trellis/config.yaml` is the sole explicit project-owned registry exception; workflow/task.py/Pi extension remain directly compared to package sources.
- Corrected progress/research wording and parent/child checkboxes to match committed evidence.

## Validation and second review

- Python contracts: 7/7 passed.
- Context fixture: 1750/3822/3766 bytes, one occurrence per selected artifact marker.
- Task manifests: 8 implement and 6 check entries valid.
- Global spec/skill Markdown links: 140 checked, none broken.
- Stale typed-approval/runtime contract search: none outside historical/task evidence.
- Working and committed-range diff checks: passed.
- `trellis update --dry-run`: no mutation; 144 unchanged files, project-owned config/AGENTS plus the documented `.claude/skills` parent-alias false positive.
- Second independent reviewer: `APPROVED`, no P0–P2.

Review artifacts:

- First: `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/710b6c9b-a456-422c-a628-b006730e60b9/baseline-final-review.md`
- Second: `/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/4101c88a-22f9-4631-9102-b428226c53d4/baseline-remediation-review.md`
