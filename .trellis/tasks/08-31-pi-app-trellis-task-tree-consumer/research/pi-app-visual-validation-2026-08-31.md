# Visual Test Report

**Target:** `/workspace/pi-app/.worktree/feature-trellis-work-item-visibility` (`feature/trellis-work-item-visibility`)
**Workspace intended for UI:** `/workspace/remote2/nimloth`
**Mode:** read-only/runtime validation; no source edits or commits
**Viewports tested:** none — isolated Electron could not launch

## Summary

**Not ready for visual acceptance in this environment.** The project build and the targeted Trellis/attention unit tests pass, but the project-supported Playwright Electron launcher cannot start because the repository's bundled Electron lacks Nix runtime shared libraries. A fallback to the system Nix Electron is incompatible with the app bundle (`electron` resolves as a Node module and does not export `BrowserWindow`). Therefore no isolated window, screenshots, accessibility snapshot, or UI interaction can be honestly reported.

No message was sent to a real user session. No UI fixture was injected.

## Commands and observations

1. `cd /workspace/pi-app/.worktree/feature-trellis-work-item-visibility && git branch --show-current && npm run build`
   - **PASS.** Build completed: main, preload, and renderer produced.
   - Build emitted five existing Vite `INEFFECTIVE_DYNAMIC_IMPORT` warnings (clipboard/config/ASR/notification modules), not runtime errors.

2. Isolated Playwright Electron launch using the project E2E shape:
   - `PI_E2E=1 ... electron.launch({ executablePath: require('electron'), args: ['--user-data-dir=/tmp/pi-trellis-visual-user-data', 'out/main/index.js'] })`
   - **FAIL before first window.** Playwright reports `Error: Process failed to launch!`.
   - Direct executable evidence:
     ```text
     /workspace/pi-app/node_modules/electron/dist/electron: error while loading shared libraries: libglib-2.0.so.0: cannot open shared object file: No such file or directory
     ```
   - `ldd` confirms further unresolved bundled-Electron dependencies: `libglib-2.0.so.0`, `libgobject-2.0.so.0`, `libgio-2.0.so.0`, `libnss3.so`, `libgtk-3.so.0`, `libX11.so.6`, and others.

3. Fallback attempt with the running environment's Nix Electron and the same isolated user-data directory:
   - **FAIL.** Placing `--user-data-dir` before the app gives `bad option`; placing it after the app loads the JS as ordinary Node and fails:
     ```text
     SyntaxError: The requested module 'electron' does not provide an export named 'BrowserWindow'
     ```
   - This cannot safely replace the supported bundled-Electron E2E route.

4. `npx vitest run src/renderer/src/features/side-panels/trellis-dashboard-model.test.ts src/renderer/src/features/side-panels/workspace-tasks-side-panel.test.tsx src/renderer/src/stores/__tests__/extension-ui-store.test.ts src/renderer/src/lib/extension-ui-channel.test.ts src/main/workspace-task-panel-reader.test.ts`
   - **PASS:** 5 files / 32 tests passed in 1.98s.

## Requested UI checks

| Check | Result | Evidence / reason |
|---|---|---|
| Nimloth Trellis sidebar tree; plan sections/items; current task/item; lifecycle vs runtime | NOT VISUALLY VERIFIED | No isolated window could be created. Relevant model/panel tests passed. |
| Planning review raw PRD/design/implement, hashes/changes, scope/exclusions/validation; read-only absent reliable request | NOT VISUALLY VERIFIED | No isolated window. Relevant panel tests passed. |
| Legacy/stale/conflict/issues clarity | NOT VISUALLY VERIFIED | No isolated window. Relevant dashboard-model tests passed. |
| Session switch/reload recovery; pending question not auto-declined | NOT VISUALLY VERIFIED | No isolated window. Relevant extension UI/store tests passed. |
| Isolated background pending-attention and “later” fixture flow | NOT RUN | Not injected: doing so requires a running isolated renderer. No real session was contacted. |

## Screenshots / console

- **Screenshots:** none; launch failed before `firstWindow()` and before the first screenshot call.
- **Renderer console/page errors:** unavailable for the same reason.
- **Launch console error:** the missing `libglib-2.0.so.0` error above.

## Findings

### P1 — Visual acceptance blocked by unsupported local Electron runtime

- **Location:** project-supported Playwright/Electron E2E bootstrap in this Nix shell.
- **Description:** `require('electron')` resolves to the bundled Electron expected by `e2e/*.spec.ts`, but it cannot load basic system libraries. The available Nix Electron cannot run this bundle as Electron, so it is not a safe workaround.
- **Suggested fix:** execute the supported `npm run test:e2e` / Playwright Electron route in the project-provisioned shell/container with the bundled Electron runtime dependencies, or provide a documented project E2E launcher that uses the Nix Electron correctly. Re-run the requested isolated visual matrix there.

## Source-tree safety

- No source files were edited by this validation; no commit was made.
- The target worktree already had the implementation's unstaged changes on entry. Build output is ignored.
- `git diff --cached --name-only` was empty (no staged files).
- `git diff --check` reports pre-existing trailing-whitespace findings in target implementation files; this validation did not change them.
