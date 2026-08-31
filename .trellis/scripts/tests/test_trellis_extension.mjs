import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";

const projectRoot = resolve(dirname(new URL(import.meta.url).pathname), "../../..");
const extensionPath = join(projectRoot, ".pi/extensions/trellis/index.ts");
const require = createRequire(import.meta.url);
const globalNodeModules = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
const jitiModule = process.env.TRELLIS_JITI_MODULE ?? join(
  globalNodeModules, "@earendil-works/pi-coding-agent/node_modules/jiti"
);
const { createJiti } = require(jitiModule);
const jiti = createJiti(import.meta.url);
const { default: extension } = await jiti.import(extensionPath);
delete process.env.TRELLIS_SUBAGENT_CHILD;

function makeRepo(base, name) {
  const root = join(base, name);
  const task = join(root, ".trellis/tasks/task");
  mkdirSync(task, { recursive: true });
  symlinkSync(join(projectRoot, ".trellis/scripts"), join(root, ".trellis/scripts"), "dir");
  mkdirSync(join(root, ".pi"), { recursive: true });
  symlinkSync(join(projectRoot, ".pi/agents"), join(root, ".pi/agents"), "dir");
  writeFileSync(join(task, "task.json"), JSON.stringify({
    id: "task", title: "Task", status: "in_progress", parent: null, children: [],
  }));
  writeFileSync(join(task, "prd.md"), "# Task\n");
  writeFileSync(join(task, "implement.md"), "## Work\n- [ ] [W-001] Implement.\n");
  return root;
}

const shortHash = (value) => createHash("sha256").update(value).digest("hex").slice(0, 24);

const base = mkdtempSync(join(tmpdir(), "trellis-extension-test-"));
const rootA = makeRepo(base, "root-a");
const rootB = makeRepo(base, "root-b");
const foreign = join(base, "foreign");
mkdirSync(foreign);
process.chdir(foreign);

const tools = new Map();
const handlers = new Map();
const pi = {
  registerTool(tool) { tools.set(tool.name, tool); },
  registerShortcut() {},
  on(event, handler) {
    const list = handlers.get(event) ?? [];
    list.push(handler);
    handlers.set(event, list);
  },
  getThinkingLevel() { return "off"; },
};
extension(pi);

assert(tools.has("trellis_work_item"), "work-item cursor tool must be registered");
assert(tools.has("trellis_approval"), "typed approval tool must be registered");
const cursor = tools.get("trellis_work_item");
const approval = tools.get("trellis_approval");
assert.deepEqual(cursor.parameters.properties.action.enum,
  ["select", "update", "block", "evidence", "release"]);
const subagent = tools.get("trellis_subagent");
assert("workItemRef" in subagent.parameters.properties,
  "Trellis subagent dispatch must carry an explicit workItemRef");
assert(subagent.parameters.required.includes("workItemRef"),
  "workItemRef must be schema-required, not only runtime-checked");

const ctx = (root, sessionId = "same-session") => ({
  cwd: root,
  sessionManager: {
    getSessionId: () => sessionId,
    getSessionFile: () => `${sessionId}.jsonl`,
  },
  ui: { notify() {} },
  model: { provider: "test", id: "model" },
});
async function emit(name, event, context) {
  let result;
  for (const handler of handlers.get(name) ?? []) result = await handler(event, context);
  return result;
}

const approvalRequests = [];
const approvalCtx = (root, responseFactory, sessionId = "approval-session") => ({
  ...ctx(root, sessionId),
  hasUI: true,
  mode: "rpc",
  ui: {
    notify() {},
    async custom(factory, options) {
      approvalRequests.push({ factory, options });
      return responseFactory(options.trellisApproval);
    },
    async input() { return "TUI comment"; },
  },
});

const approved = await approval.execute("approval-tool-call", {
  taskRef: "task", kind: "implementation",
  scope: [".trellis/scripts"], exclusions: ["commit", "push"],
  validationCommands: ["python3 -m unittest"],
}, undefined, undefined, approvalCtx(rootA, (request) => ({
  decision: "approve",
  workspaceRoot: request.workspaceRoot,
  requestId: request.approvalRequestId,
  taskRef: request.taskRef,
  kind: request.approvalKind,
  artifactHashes: request.artifactHashes,
  rootFingerprint: request.rootFingerprint,
  contextKey: request.contextKey,
  sessionId: request.sessionId,
  toolCallId: request.toolCallId,
  reviewSetHash: request.reviewSetHash,
})));
assert.match(approved.content[0].text, /approved.*authorized/i);
const approvalPayload = approvalRequests[0].options.trellisApproval;
assert.equal(approvalPayload.workspaceRoot, rootA);
assert.equal(approvalPayload.contextKey, "pi_approval-session");
assert.equal(approvalPayload.sessionId, "approval-session.jsonl");
assert.equal(approvalPayload.toolCallId, "approval-tool-call");
assert.equal(approvalPayload.taskRef, "task");
assert.equal(approvalPayload.approvalKind, "implementation");
const approvalRuntime = JSON.parse(readFileSync(
  join(rootA, ".trellis/.runtime/execution/pi_approval-session.json"), "utf8"
));
assert.equal(approvalRuntime.approvalRequests.length, 1);
assert.equal(approvalRuntime.approvalReceipts.length, 1);
assert.equal(approvalRuntime.approvalReceipts[0].decision, "approve");

const tuiRoot = makeRepo(base, "tui-root");
const tuiResult = await approval.execute("approval-tui-call", {
  taskRef: "task", kind: "planning", scope: [".trellis/tasks/task"], exclusions: ["implementation"],
}, undefined, undefined, {
  ...ctx(tuiRoot, "tui-session"),
  hasUI: true,
  mode: "tui",
  ui: {
    notify() {},
    async select() { return "comment"; },
    async input() { return "Revise the plan"; },
  },
});
assert.match(tuiResult.content[0].text, /comment.*no authorization/i);
const tuiRuntime = JSON.parse(readFileSync(
  join(tuiRoot, ".trellis/.runtime/execution/pi_tui-session.json"), "utf8"
));
assert.equal(tuiRuntime.approvalReceipts[0].decision, "comment");
assert.equal(tuiRuntime.approvalReceipts[0].comment, "Revise the plan");

const piAppRoot = process.env.PI_APP_WORKTREE;
if (piAppRoot) {
  const { createDesktopUIBridge } = await jiti.import(
    join(piAppRoot, "src/worker/desktop-ui-bridge.ts")
  );
  const integratedRoot = makeRepo(base, "integrated-root");
  const emitted = [];
  let bridge;
  bridge = createDesktopUIBridge(
    { on() { return () => {}; } },
    (request) => {
      emitted.push(request);
      if (request.kind !== "trellis_approval") return;
      queueMicrotask(() => bridge.handleExtensionUIResponse({
        id: request.id,
        result: {
          decision: "approve",
          workspaceRoot: request.workspaceRoot,
          rootFingerprint: request.rootFingerprint,
          contextKey: request.contextKey,
          sessionId: request.sessionId,
          toolCallId: request.toolCallId,
          requestId: request.approvalRequestId,
          taskRef: request.taskRef,
          kind: request.approvalKind,
          artifactHashes: request.artifactHashes,
          reviewSetHash: request.reviewSetHash,
        },
      }));
    },
  );
  const integrated = await approval.execute("approval-integrated-call", {
    taskRef: "task", kind: "implementation", scope: ["src"], exclusions: ["commit"],
  }, undefined, undefined, {
    ...ctx(integratedRoot, "integrated-session"),
    hasUI: true,
    mode: "rpc",
    ui: bridge.uiContext,
  });
  assert.match(integrated.content[0].text, /approved.*authorized/i);
  assert.equal(emitted.length, 1);
  assert.equal(emitted[0].kind, "trellis_approval");
  assert.equal(emitted[0].workspaceRoot, integratedRoot);
  assert.equal(emitted[0].sessionId, "integrated-session.jsonl");
  assert.equal(emitted[0].toolCallId, "approval-integrated-call");
  const integratedRuntime = JSON.parse(readFileSync(
    join(integratedRoot, ".trellis/.runtime/execution/pi_integrated-session.json"), "utf8"
  ));
  assert.equal(integratedRuntime.approvalReceipts.length, 1);
  assert.equal(integratedRuntime.approvalReceipts[0].decision, "approve");
  bridge.dispose();
}

const staleRoot = makeRepo(base, "stale-root");
await assert.rejects(approval.execute("approval-stale-call", {
  taskRef: "task", kind: "implementation", scope: ["src"], exclusions: ["commit"],
}, undefined, undefined, approvalCtx(staleRoot, (request) => {
  writeFileSync(join(staleRoot, ".trellis/tasks/task/implement.md"), "## Work\n- [ ] [W-001] Changed.\n");
  return {
    decision: "approve", workspaceRoot: request.workspaceRoot,
    requestId: request.approvalRequestId, taskRef: request.taskRef,
    kind: request.approvalKind, artifactHashes: request.artifactHashes,
    rootFingerprint: request.rootFingerprint, contextKey: request.contextKey,
    sessionId: request.sessionId, toolCallId: request.toolCallId, reviewSetHash: request.reviewSetHash,
  };
})), /artifact_hash_mismatch/);
const staleRuntime = JSON.parse(readFileSync(
  join(staleRoot, ".trellis/.runtime/execution/pi_approval-session.json"), "utf8"
));
assert.equal(staleRuntime.approvalReceipts.length, 0);

const cancelRoot = makeRepo(base, "cancel-root");
await assert.rejects(approval.execute("approval-cancel-call", {
  taskRef: "task", kind: "implementation", scope: ["src"], exclusions: ["commit"],
}, undefined, undefined, approvalCtx(cancelRoot, () => ({
  status: "system_cancelled", reason: "worker-stopped",
}))), /system_cancelled.*worker-stopped/);
const cancelRuntime = JSON.parse(readFileSync(
  join(cancelRoot, ".trellis/.runtime/execution/pi_approval-session.json"), "utf8"
));
assert.equal(cancelRuntime.approvalReceipts.length, 0);

await emit("session_start", { reason: "startup" }, ctx(rootA));
const selected = await cursor.execute("cursor-call", {
  action: "select", taskRef: "task", workItemRef: "W-001",
  nextAction: "write code",
}, undefined, undefined, ctx(rootA));
assert.match(selected.content[0].text, /W-001/);
const runtimeA = join(rootA, ".trellis/.runtime/execution/pi_same-session.json");
const runtimeB = join(rootB, ".trellis/.runtime/execution/pi_same-session.json");
assert.equal(JSON.parse(readFileSync(runtimeA, "utf8")).assignments.length, 1);
assert.throws(() => readFileSync(join(foreign, ".trellis/.runtime/execution/pi_same-session.json")));

await emit("tool_execution_start", {
  toolCallId: "read-1", toolName: "read", args: { path: "secret-is-not-copied" },
}, ctx(rootA));
let state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert.deepEqual(state.assignments[0].observedActivity, {
  toolName: "read", toolCallId: "read-1", status: "running",
  at: state.assignments[0].observedActivity.at,
});
assert(!JSON.stringify(state).includes("secret-is-not-copied"));

const aborted = new AbortController();
aborted.abort();
const delegatedProbe = await subagent.execute("trellis-sub-1", {
  agent: "trellis-check", prompt: "safe aborted probe", mode: "single",
  workItemRef: "task#W-001",
}, aborted.signal, undefined, ctx(rootA));
assert.equal(delegatedProbe.details.runs[0].status, "cancelled");
state = JSON.parse(readFileSync(runtimeA, "utf8"));
const trellisDelegated = state.assignments.find(
  (item) => item.executor?.toolCallId === "trellis-sub-1"
);
assert.equal(trellisDelegated.role, "delegated");
assert.equal(trellisDelegated.declaredState, "failed");

await subagent.execute("trellis-chain", {
  agent: "trellis-check", mode: "chain", prompts: ["first", "never-started"],
  workItemRef: "task#W-001",
}, aborted.signal, undefined, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
const chainAssignments = state.assignments
  .filter((item) => item.executor?.toolCallId === "trellis-chain")
  .sort((a, b) => a.executor.runId.localeCompare(b.executor.runId));
assert.equal(chainAssignments[0].declaredState, "failed");
assert(chainAssignments[1].releasedAt,
  "a chain executor that never started must be released instead of reported failed");

await emit("tool_execution_start", {
  toolCallId: "generic-1", toolName: "subagent",
  args: { agent: "scout", prompt: "generic-prompt-is-not-copied" },
}, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
const delegated = state.assignments.find((item) => item.executor?.toolCallId === "generic-1");
assert.equal(delegated.role, "delegated");
assert(!JSON.stringify(state).includes("generic-prompt-is-not-copied"));
delegated.heartbeatAt = "2020-01-01T00:00:00Z";
delegated.updatedAt = "2020-01-01T00:00:00Z";
writeFileSync(runtimeA, JSON.stringify(state));
await emit("tool_execution_update", {
  toolCallId: "generic-1", toolName: "subagent",
}, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert.notEqual(
  state.assignments.find((item) => item.assignmentId === delegated.assignmentId).heartbeatAt,
  "2020-01-01T00:00:00Z",
  "live delegated assignments must receive heartbeat while the outer subagent tool runs",
);
await emit("tool_execution_end", {
  toolCallId: "generic-1", toolName: "subagent", isError: false,
}, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert(state.assignments.find((item) => item.assignmentId === delegated.assignmentId).releasedAt);

const alternateCtx = ctx(rootA, "alternate-session");
await cursor.execute("cursor-alt", {
  action: "select", taskRef: "task", workItemRef: "W-001",
}, undefined, undefined, alternateCtx);
const runtimeAlt = join(rootA, ".trellis/.runtime/execution/pi_alternate-session.json");
for (const context of [ctx(rootA), alternateCtx]) {
  await emit("tool_execution_start", {
    toolCallId: "shared-generic-id", toolName: "subagent", args: { agent: "scout" },
  }, context);
}
await emit("tool_execution_end", {
  toolCallId: "shared-generic-id", toolName: "subagent", isError: false,
}, ctx(rootA));
const mainShared = JSON.parse(readFileSync(runtimeA, "utf8")).assignments.find(
  (item) => item.executor?.toolCallId === "shared-generic-id"
);
const alternateShared = JSON.parse(readFileSync(runtimeAlt, "utf8")).assignments.find(
  (item) => item.executor?.toolCallId === "shared-generic-id"
);
assert(mainShared.releasedAt, "generic completion must release its own context assignment");
assert.equal(alternateShared.releasedAt, null,
  "same toolCallId in another context must remain isolated");
await emit("tool_execution_end", {
  toolCallId: "shared-generic-id", toolName: "subagent", isError: false,
}, alternateCtx);

const setupCallId = "partial-setup";
const setupCollisionId = `a-sub-${shortHash(`${setupCallId}:1`)}`;
execFileSync("python3", [join(rootA, ".trellis/scripts/task.py"),
  "work-item", "select", "--context", "pi_same-session",
  "--task", "task", "--item", "W-001", "--assignment", setupCollisionId,
  "--role", "delegated", "--executor-kind", "subagent",
  "--session-id", "same-session.jsonl", "--agent", "trellis-check",
], { cwd: rootA });
await assert.rejects(subagent.execute(setupCallId, {
  agent: "trellis-check", mode: "parallel", prompts: ["first", "second"],
  workItemRef: "task#W-001",
}, undefined, undefined, ctx(rootA)), /duplicate assignment id/);
const partialSetupId = `a-sub-${shortHash(`${setupCallId}:0`)}`;
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert(state.assignments.find((item) => item.assignmentId === partialSetupId).releasedAt,
  "partial delegated setup must clean assignments created before a later setup failure");

await cursor.execute("cursor-call-2", {
  action: "select", taskRef: "task", workItemRef: "W-001",
}, undefined, undefined, ctx(rootB));
assert.equal(JSON.parse(readFileSync(runtimeB, "utf8")).assignments.length, 1,
  "same session id must remain isolated by root");

await emit("session_shutdown", { reason: "reload" }, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert.equal(state.session.active, false);
await emit("session_start", { reason: "reload" }, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert.equal(state.session.active, true, "reload must resume persisted assignments");

await assert.rejects(
  cursor.execute("cursor-block", { action: "block" }, undefined, undefined, ctx(rootA)),
  /blocker is required/,
);
await cursor.execute("cursor-call-3", { action: "release" }, undefined, undefined, ctx(rootA));
state = JSON.parse(readFileSync(runtimeA, "utf8"));
assert(state.assignments[0].releasedAt);
await emit("session_shutdown", { reason: "quit" }, ctx(rootA));

console.log("trellis extension safe probe: ok");
