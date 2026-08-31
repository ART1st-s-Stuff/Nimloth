# Progress

## 2026-08-31 — implementation boundary and worktree

- Parent implementation was approved with `执行`.
- Human explicitly selected nested pi-app worktree path `/workspace/pi-app/.worktree/feature-trellis-work-item-visibility`.
- Verified branch `feature/trellis-work-item-visibility`, exact base `1b2834dcb70bb593a4fcab8aa5357437c5632b0f`, common Git dir `/workspace/pi-app/.git`, and clean status.
- Original `/workspace/pi-app` `main` remains dirty only with pre-existing unrelated changes and must not be modified or cleaned.

### Consumer change boundary

Behavior gap:

- pi-app currently reads a static Trellis lifecycle summary and auto-cancels background/session-switched questionnaires. It cannot display task-tree work items/executors, review exact task artifacts, or preserve pending cross-session approvals.

Expected ownership:

- pi-app shared IPC/AppEvent contracts for selected session, dashboard data and pending request identity.
- Electron main async Trellis dashboard reader and source-session request registry.
- Renderer task-tree/review/approval UI and workspace pending-attention queue.
- Adjacent unit/component/integration tests for dashboard fallback, request correlation, session switching, concurrency and cancellation semantics.

Explicit exclusions:

- Do not modify `/workspace/pi-app` main worktree or its dirty files.
- Do not duplicate the Trellis Markdown parser/state machine in pi-app.
- Do not write Trellis task/plan or Pi TaskTree state.
- Do not commit/push/merge, run experiments, or alter Nimloth producer semantics without returning to parent planning.

The producer contract and dashboard-v1 fixture are the consumer authority.

## 2026-08-31 — consumer RED/GREEN implementation milestone

Implemented in the approved pi-app worktree only:

- Locked a full dashboard-v1 consumer fixture generated from the producer fixture and contract.
- Replaced synchronous `task.py current` execution with an async guarded `task.py dashboard --json --context pi_<selected-session>` reader; added timeout, output-size, schema-major, root-identity, malformed-output and static fallback handling.
- Added dashboard presentation for selected current work item, task parent/children including archived nodes, plan sections/items/progress, main/subagent assignments, declared vs observed source labels, elapsed, blocker, queue, typed evidence and next action.
- Added raw PRD/design/implement review with exact hashes/change state and typed approval boundaries. Approval controls fail closed unless an exact root/session/request/toolCall plus task/kind/hash-correlated Pi request exists; changed artifacts expire the controls.
- Replaced Renderer single-slot extension UI state with a workspace-scoped registry keyed by root/session/request/toolCall. Background and session switch now retain pending requests; later, explicit user decline and worker/system cancellation have distinct state.
- Main now forwards background dialogs with source workspace/session/toolCall identity and validates response identity before routing to the source Worker.

### RED/GREEN evidence

- RED: 4 focused suites failed before implementation (missing dashboard model/reader API and pending registry; background request was auto-cancelled).
- GREEN: 7 focused suites, 47 tests passed after implementation, including the real async child-process path, reader/fallback, model/status, component review, session switch, background, concurrency, exact routing and terminal semantics.
- `npm run build`: passed.
- `npm run lint`: passed with one pre-existing warning in `timeline.tsx` and no errors.
- Full unit suite: 224/225 files passed; one unrelated existing rewind test `tree-to-timeline-view.e2e.test.tsx` fails in isolation with `row not found: user: message-5`.
- `npm run typecheck`: blocked by existing untouched `src/renderer/src/components/icons/themes/fluent.tsx:107` TS2742; no touched-file type errors remained before that project-level failure.

### Remaining checks / risks

- W-030 remains open pending confirmation that status presentation should be refactored into a shared RunPanel/tree-card helper rather than the current use of authoritative `runState` plus local labels.
- W-041 remains open because project typecheck is not green.
- Development-mode manual probes for long tools, waiting states, subagents, session switch/reload and live approval transport remain open.
- Producer currently exposes no demonstrated live `trellis_approval` Extension UI correlation request; dashboard review therefore correctly remains read-only in that case. The UI supports exact correlated transport but this path still needs an end-to-end producer probe.
- The exact external producer of the string `User declined to answer questions` remains unlocated; the confirmed automatic-cancel paths and single-slot loss are covered and removed without claiming that missing string provenance.

## 2026-08-31 — independent-check blockers GREEN

- W-030 completed: extracted RunPanel's existing `idle | running | failed | tool | thinking` resolver into a shared presentation helper and reused it for Trellis Pi AppEvent observations. Focused tests now assert task lifecycle `in_progress`, declared assignment `working`, plan `pending`, and observed presentation `tool` remain separate facts.
- Added a Main-owned live pending-request snapshot keyed by workspace root, session file, request ID and toolCall ID. The new allowlisted `extension.pendingUI.list` IPC reconstructs Renderer attention rows as suspended after reload/workspace hydration; it stores no Trellis task/plan authority and does not auto-cancel Workers.
- Worker response, explicit cancel, dismiss-all and worker exit remove the Main snapshot atomically with the existing source identity. Duplicate/late response and cancel fail closed; recovery rejects foreign-root rows and does not resurrect a request that became system-cancelled while enumeration was in flight.
- RED evidence: the three focused suites initially had 6 failures (parallel observed `running` label, absent Main enumeration, absent Renderer recovery).
- GREEN evidence: focused extension/Trellis suites pass (6 files, 56 tests); Node typecheck passes; build passes; lint has 0 errors and one untouched `timeline.tsx` warning.
- Web typecheck still stops at untouched `src/renderer/src/components/icons/themes/fluent.tsx:107` TS2742. The file and both tsconfigs are byte-identical to HEAD; Node typecheck is GREEN, so W-041 remains open only on the clean-base web blocker.
- Full unit result: 225/226 files passed, 1042 tests passed and 1 skipped; the sole failure is the existing isolated `tree-to-timeline-view.e2e.test.tsx` `row not found: user: message-5`. That test file is byte-identical to HEAD and reproduces alone.
- `npm run test:scripts` has one environment/dependency failure: unchanged `electron-builder-yaml-doc.test.mjs` cannot find `node_modules/yaml/dist/doc/directives.js`; the script, `package.json` and lockfile are byte-identical to HEAD and the runtime file is absent.
- `git -c core.whitespace=cr-at-eol diff --check` passes. Plain `git diff --check` reports CRLF as trailing whitespace in pre-existing CRLF shared files, so the CR-at-EOL-aware result is the applicable content check.
- Trellis task validation passes (8 curated implement entries and 8 check entries). Pi TaskTree/first-unchecked diff scan is clean; pi-app dirty `main` remains unchanged.

## 2026-08-31 — W-039 real typed approval consumer GREEN

- Desktop `ui.custom()` accepts a tested explicit `trellisApproval` option and emits `kind=trellis_approval`; generic custom/questionnaire heuristics are not used. Worker/Main preserve full producer and transport identity, keep background requests recoverable, and system-cancel malformed root/session requests with a typed reason.
- Renderer validates producer root/session against Main-injected source identity, correlates the dashboard request exactly, and returns approve/decline/comment with request/task/kind/hash/review/session/toolCall/root identity. Unknown, duplicate, changed-artifact and late responses remain fail closed.
- Focused GREEN is 4 suites / 32 tests, including actual custom request emission, tool-abort/system cancellation, malformed typed-option rejection without questionnaire fallback, Main mismatch/late routing, raw payload validation and all three review decisions. Build passed; lint has zero errors and one untouched warning; Node and non-composite web typechecks passed.
- Canonical web typecheck still hits untouched `fluent.tsx:107` TS2742. Full unit remains 226/227 files because unchanged `tree-to-timeline-view.e2e.test.tsx` fails with `row not found` in isolation. These are recorded residual branch checks, not W-039 completion claims.

## 2026-08-31 — approved consumer work commit

- After complete-diff review and explicit human commit approval, the entire pi-app consumer implementation/tests were committed locally in the approved worktree as `4efdce1` (`feat(trellis): add live task execution and approval review`).
- W-045 is complete. The approved worktree is clean and dirty `main` remains unchanged. No push or merge occurred.
- Consumer W-041/W-042/W-043 remain open because canonical nested-worktree composite typecheck and Electron visual/manual acceptance are not fully green in this environment.

## 2026-08-31 — feature-worktree NixOS launcher GREEN

- Kept dirty `/workspace/pi-app` main untouched; the approved branch was already checked out at `/workspace/pi-app/.worktree/feature-trellis-work-item-visibility`. Added `npm run start:nixos` plus a worktree-relative `scripts/start-nixos-wayland.sh` so the launcher resolves and runs the feature worktree rather than main.
- The launcher supports conditional build, `--dry-run`, external `--isolated-profile`, proxy forwarding, cached Nix Electron 43 discovery and Waypipe/GTK portal recursion. It explicitly unsets inherited `ELECTRON_RUN_AS_NODE`, which otherwise made Chromium flags fail when launched from a Pi Desktop worker environment.
- Actual launch succeeded with isolated profile `/tmp/pi-desktop-trellis-visibility-20260831-green`: renderer PID `493638` reports exact `--app-path=/workspace/pi-app/.worktree/feature-trellis-work-item-visibility`; launch log is `/tmp/pi-desktop-trellis-visibility-launch-green.log`.
- Independent check found and RED-first fixed one P1 no-Electron-candidate diagnostic bug under `pipefail`; focused regression `scripts/tests/start-nixos-wayland.test.mjs`, shell parse, npm help/dry-run, production build and diff checks pass.
- This closes only the launcher/startup portion of W-042. The broader task-tree/review/approval manual interaction matrix remains pending user operation in the running feature app.
- Human directed that this session's remaining modifications be staged or committed. The reviewed launcher/package/test scope was committed locally in the feature worktree as `3a2ab1c` (`feat(dev): add NixOS Wayland launcher`). Nothing was pushed or merged.

## 2026-08-31 — pi-app canonical directory switched to feature branch

- Human explicitly requested preserving the dirty pi-app main worktree, deleting the nested feature worktree, and checking the canonical `/workspace/pi-app` directory out to the feature branch.
- Validated the seven pre-existing main-worktree files with three focused suites (11 tests), production build and diff check, then preserved them on dedicated branch `wip/main-local-changes-before-trellis-switch-20260831` as commit `28d59ff` rather than mixing them into the Trellis feature branch. Original `main` remains at `1b2834d`.
- Before cleanup, verified nested worktree tracked/untracked cleanliness, 147 ignored entries limited to the shared `node_modules` symlink, build outputs and TypeScript build info, no submodules, exact branch/HEAD and the isolated feature app process group. Stopped only that isolated process group, then ordinary `git worktree remove` succeeded without force; path and registration disappeared.
- `/workspace/pi-app` is now clean on `feature/trellis-work-item-visibility@3a2ab1c`, and it is the repository's sole registered worktree. `npm run start:nixos -- --build --isolated-profile /tmp/pi-desktop-trellis-root-feature-20260831` built and launched successfully; renderer PID `510592` reports exact `--app-path=/workspace/pi-app` and the isolated profile.
- No change was discarded, no force/prune/manual Git metadata edit occurred, and nothing was pushed or merged.

## 2026-08-31 — host-visible Waypipe launcher correction

- Human reported that the launched process was not visible on the host. Re-investigation showed the previous probe inherited `WAYLAND_DISPLAY=wayland-i631JlJe3w` from the running Pi Desktop worker and therefore bypassed `wh`; process existence and renderer `--app-path` did not prove host visibility. The prior success claim was corrected.
- Stopped only the invisible isolated instance. Added a fail-closed worker-environment guard and explicit `--host-display`: it clears inherited display/socket, invokes `wh --with-gtk-portal`, and uses `PI_APP_HOST_DISPLAY_READY=1` to prevent recursive Waypipe loops. Dry-run now distinguishes host-forward from display reuse.
- RED tests fail on `3a2ab1c` and GREEN 3/3 covers worker rejection, fresh host-display resolution and no-Electron diagnostics. Independent review plus a fake-`wh` quoting/recursion probe found no remaining P0/P1 launcher issue.
- Real host-channel probe succeeded, then the actual app was launched with a distinct `WAYLAND_DISPLAY=wayland-qVzd6k92DA` and isolated profile `/tmp/pi-desktop-trellis-host-visible-20260831`; renderer PID `515342` is live. This proves a separate host-forward channel and live renderer, while actual human visual confirmation remains pending.
- The launcher correction was committed locally in pi-app as `97ef029` (`fix(dev): require host-forwarded Wayland launch`). Nothing was pushed or merged.

## 2026-08-31 — native Wayland initial-window fallback

- Human still saw no window after a distinct host-forward channel, so channel/process evidence was again treated as insufficient. Source inspection found the feature branch only presented the BrowserWindow from `ready-to-show`; Electron native Wayland can complete renderer `did-finish-load` without emitting that event.
- RED imported the previously preserved window-presentation test while the module was absent. Implemented a shared exactly-once presenter using `ready-to-show` plus `did-finish-load` fallback, preserving development `showInactive`, production/e2e `show`, and destroyed-window safety. Only the three window-presentation paths were recovered from the WIP evidence; unrelated markdown and old launcher content were not merged.
- Independent review expanded focused coverage to 6 cases and found no remaining P0/P1 source issue. Focused test, Node typecheck, ESLint/build and diff checks pass; the behavior matches Electron native Wayland `ready-to-show` omission evidence.
- Rebuilt and launched a fresh host-forward instance on `WAYLAND_DISPLAY=wayland-DS5xFcoF5p` with renderer PID `518965` and profile `/tmp/pi-desktop-trellis-window-fallback-20260831`. This proves the fixed build is live but still awaits human visual confirmation.
- The fix was committed locally in pi-app as `f596a9c` (`fix(window): show app after renderer load on Wayland`). Nothing was pushed or merged.

## 2026-08-31 — bare launcher defaults to host-forward

- Human confirmed the fallback-fixed instance was visible, but a direct bare script invocation was not. The remaining difference was operational: successful probes passed `--host-display`, while bare invocation still reused any inherited shell `WAYLAND_DISPLAY`.
- Changed the safe default so bare `scripts/start-nixos-wayland.sh` clears inherited display/socket and creates a fresh `wh --with-gtk-portal` channel. Existing display reuse now requires explicit `--reuse-display`; Pi Desktop worker non-dry-run invocation remains rejected.
- RED/GREEN tests cover default host-forward under inherited display and explicit reuse. Independent check verified one `wh` call, recursion-marker termination, quoting, worker rejection and 5/5 focused tests with no P0/P1 issue.
- Actual direct invocation without `--host-display` created `WAYLAND_DISPLAY=wayland-Z56g5Gl6yT`; renderer PID `523217` is live with profile `/tmp/pi-desktop-trellis-default-script-20260831`. This follows the same host-forward and window-presentation path the human previously confirmed visible; current direct-instance visual confirmation remains pending.
- The default correction was committed locally in pi-app as `435706c` (`fix(dev): host-forward bare launcher by default`). Nothing was pushed or merged.

## 2026-09-01 — legacy-reference visual-density P1 check

- Reviewed only the latest dirty Trellis panel/component-test delta on canonical `/workspace/pi-app`, now verified on `feature/trellis-work-item-visibility@435706c`; the former nested worktree no longer exists.
- Found two P1 primary-view identity defects: matched legacy current execution had dropped its task ref, and an orphaned legacy assignment could still expose the full `legacy-<hash>` ref. Added RED coverage for both, then kept the internal correlation identity while rendering `task#item text` or `task#旧计划项` plus the compact accessible legacy badge.
- Stable `W-010` plan-row rendering remains visible. The presentation-model warning/error collection is unchanged; only the redundant `legacy_unstable_id` row text is represented by the accessible badge.
- Refresh semantics were rechecked rather than overstated: selected session/workspace changes fetch immediately; dashboard/runtime files are polled every 10 seconds and also refresh at terminal run/tool-end events or manually; Pi run/tool activity is merged locally from AppEvent. Dashboard-only changes can therefore lag by roughly 10 seconds plus reader/render latency.
- Final validation is GREEN: Trellis panel/model suites pass (2 files / 16 tests), canonical `npm run typecheck` passes, production build passes, and `git diff --check` passes. ESLint has zero errors and one unchanged pre-existing `timeline.tsx:794` unused-disable warning.
- The compact legacy-label fix was committed locally in pi-app as `5dd9adc` (`fix(trellis): compact legacy work-item labels`). No producer parser/state semantics, push, merge, process lifecycle or Pi TaskTree state was changed.
