import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";

const projectRoot = resolve(dirname(new URL(import.meta.url).pathname), "../../..");
const extensionPath = join(projectRoot, ".pi/extensions/trellis/index.ts");
const require = createRequire(import.meta.url);
const globalNodeModules = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
const jitiModule = process.env.TRELLIS_JITI_MODULE ?? join(
  globalNodeModules, "@earendil-works/pi-coding-agent/node_modules/jiti",
);
const { createJiti } = require(jitiModule);
const jiti = createJiti(import.meta.url);
const { default: extension } = await jiti.import(extensionPath);

const base = mkdtempSync(join(tmpdir(), "trellis-context-measurement-"));
const root = join(base, "repo");
const task = join(root, ".trellis/tasks/task");
mkdirSync(task, { recursive: true });
mkdirSync(join(root, ".pi"), { recursive: true });
mkdirSync(join(root, ".trellis/.runtime/sessions"), { recursive: true });
symlinkSync(join(projectRoot, ".trellis/scripts"), join(root, ".trellis/scripts"), "dir");
symlinkSync(join(projectRoot, ".pi/agents"), join(root, ".pi/agents"), "dir");
writeFileSync(join(task, "task.json"), JSON.stringify({
  id: "task", name: "task", title: "Task", status: "in_progress", parent: null, children: [],
}));
const prdMarker = "MEASURE_PRD_MARKER";
const designMarker = "MEASURE_DESIGN_MARKER";
const planMarker = "MEASURE_PLAN_MARKER";
const specMarker = "MEASURE_SPEC_MARKER";
writeFileSync(join(task, "prd.md"), `# Task\n${prdMarker}\n`);
writeFileSync(join(task, "design.md"), `# Design\n${designMarker}\n`);
writeFileSync(join(task, "implement.md"), `# Plan\n- [ ] [W-001] ${planMarker}\n`);
mkdirSync(join(root, ".trellis/spec"), { recursive: true });
writeFileSync(join(root, ".trellis/spec/context.md"), `# Context\n${specMarker}\n`);
for (const name of ["implement", "check"]) {
  writeFileSync(join(task, `${name}.jsonl`), `${JSON.stringify({
    file: ".trellis/spec/context.md", reason: `${name} measurement`,
  })}\n`);
}
writeFileSync(join(root, ".trellis/.runtime/sessions/pi_measure-session.json"), JSON.stringify({
  current_task: ".trellis/tasks/task",
}));

const captureScript = join(base, "capture-cli.mjs");
writeFileSync(captureScript, `
import { writeFileSync } from "node:fs";
let data = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { data += chunk; });
process.stdin.on("end", () => {
  writeFileSync(process.env.TRELLIS_CAPTURE_PATH, data);
  console.log(JSON.stringify({message:{role:"assistant",content:"ok"}}));
});
`);
process.env.TRELLIS_PI_CLI_JS = captureScript;

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

const ctx = {
  cwd: root,
  sessionManager: {
    getSessionId: () => "measure-session",
    getSessionFile: () => "measure-session.jsonl",
  },
  ui: { notify() {} },
  model: { provider: "test", id: "model" },
};
async function emit(name, event) {
  let result;
  for (const handler of handlers.get(name) ?? []) result = await handler(event, ctx);
  return result;
}

await emit("session_start", { reason: "startup" });
const main = await emit("before_agent_start", { systemPrompt: "BASE_SYSTEM_PROMPT" });
const mainText = main?.systemPrompt ?? main?.message?.content ?? "";
assert(mainText, "main context measurement must capture injected content");

const subagent = tools.get("trellis_subagent");
assert(subagent, "Trellis extension must register trellis_subagent");
const childTexts = {};
for (const agent of ["trellis-implement", "trellis-check"]) {
  const capture = join(base, `${agent}.txt`);
  process.env.TRELLIS_CAPTURE_PATH = capture;
  const result = await subagent.execute(
    `measure-${agent}`,
    { agent, prompt: "measurement only", mode: "single", workItemRef: "task#W-001" },
    undefined,
    undefined,
    ctx,
  );
  assert.equal(result.details.runs[0].status, "succeeded");
  childTexts[agent] = readFileSync(capture, "utf8");
}

const occurrences = (text, marker) => text.split(marker).length - 1;
const measurement = {
  main: {
    bytes: Buffer.byteLength(mainText, "utf8"),
    prdOccurrences: occurrences(mainText, prdMarker),
    designOccurrences: occurrences(mainText, designMarker),
    planOccurrences: occurrences(mainText, planMarker),
    specOccurrences: occurrences(mainText, specMarker),
  },
};
for (const [agent, text] of Object.entries(childTexts)) {
  measurement[agent] = {
    bytes: Buffer.byteLength(text, "utf8"),
    prdOccurrences: occurrences(text, prdMarker),
    designOccurrences: occurrences(text, designMarker),
    planOccurrences: occurrences(text, planMarker),
    specOccurrences: occurrences(text, specMarker),
  };
}

if (process.env.TRELLIS_CONTEXT_MEASUREMENT_OUT) {
  writeFileSync(process.env.TRELLIS_CONTEXT_MEASUREMENT_OUT, `${JSON.stringify(measurement, null, 2)}\n`);
}
console.log(JSON.stringify(measurement));
