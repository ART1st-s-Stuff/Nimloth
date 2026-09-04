#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import { buildProjection, parsePlan, rootFingerprint } from './core.mjs';

const OUTPUT_LIMIT = 64 * 1024;
const contextIndex = process.argv.indexOf('--context-key');
const requestedContext = contextIndex >= 0 ? process.argv[contextIndex + 1] : process.env.TRELLIS_CONTEXT_ID;

async function findRoot(start) {
  let current = path.resolve(start);
  for (;;) {
    if (existsSync(path.join(current, '.trellis'))) return realpath(current);
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

const base = (root, code, message) => ({
  schemaVersion: 1,
  root: root ? { path: root, fingerprint: rootFingerprint(root) } : null,
  task: null, items: [], current: null,
  issues: [{ code, message }],
});

async function project() {
  const root = await findRoot(process.cwd());
  if (!root) return base(null, 'root-missing', 'No Trellis project root found');
  if (typeof requestedContext !== 'string' || !/^[A-Za-z0-9._-]{1,163}$/.test(requestedContext)) return base(root, 'invalid-context', 'A safe context key is required');
  let pointer;
  try { pointer = JSON.parse(await readFile(path.join(root, '.trellis/.runtime/sessions', `${requestedContext}.json`), 'utf8')); }
  catch { return base(root, 'active-task-missing', 'Active task pointer is missing or invalid'); }
  const taskRef = typeof pointer?.current_task === 'string' ? pointer.current_task : '';
  const tasksRoot = await realpath(path.join(root, '.trellis/tasks'));
  const candidate = path.resolve(root, taskRef);
  if (candidate !== tasksRoot && !candidate.startsWith(`${tasksRoot}${path.sep}`)) return base(root, 'task-outside-root', 'Active task escapes the task root');
  let taskDir;
  try { taskDir = await realpath(candidate); }
  catch { return base(root, 'task-missing', 'Active task directory is missing'); }
  if (taskDir !== tasksRoot && !taskDir.startsWith(`${tasksRoot}${path.sep}`)) return base(root, 'task-outside-root', 'Active task escapes the task root');
  let task, markdown;
  try {
    task = JSON.parse(await readFile(path.join(taskDir, 'task.json'), 'utf8'));
    markdown = await readFile(path.join(taskDir, 'implement.md'), 'utf8');
  } catch { return base(root, 'task-artifact-invalid', 'Task metadata or implementation plan is unreadable'); }
  const plan = parsePlan(taskRef, markdown);
  return buildProjection({ root, contextKey: requestedContext, sessionId: null, taskRef, taskStatus: task?.status, plan });
}

let output;
try { output = await project(); }
catch { output = base(null, 'projection-failed', 'Projection failed'); }
let encoded = `${JSON.stringify(output)}\n`;
if (Buffer.byteLength(encoded) > OUTPUT_LIMIT) {
  output = { schemaVersion: 1, root: output.root, task: output.task, items: [], current: null, issues: [...(output.issues ?? []).slice(0, 20), { code: 'projection-overflow', message: 'Projection exceeded output limit' }] };
  encoded = `${JSON.stringify(output)}\n`;
}
process.stdout.write(encoded);
