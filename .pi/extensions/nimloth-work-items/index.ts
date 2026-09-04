import { createHash } from 'node:crypto';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { applyAction, parsePlan } from './core.mjs';

const TOOL_NAME = 'nimloth_work_item';
const TOOL_GUIDANCE = 'select: omit itemRef once to list pending choices, then select one ref; update: change state/blocker/next action; evidence: add only a short bounded reference and summary; release: end the current assignment. Plan checkboxes remain authoritative.';
const hash24 = (value) => createHash('sha256').update(value).digest('hex').slice(0, 24);
const safeContext = (value) => typeof value === 'string' && /^[A-Za-z0-9._-]{1,163}$/.test(value) ? value : null;

function findRoot(start) {
  let current = path.resolve(start || process.cwd());
  for (;;) {
    if (existsSync(path.join(current, '.trellis'))) return realpathSync(current);
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

function contextKey(ctx, explicit) {
  if (safeContext(explicit)) return explicit;
  const id = ctx?.sessionManager?.getSessionId?.() ?? process.env.PI_SESSION_ID ?? process.env.PI_SESSIONID;
  if (id) {
    const raw = String(id).trim();
    const normalized = raw.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._-]+|[._-]+$/g, '').slice(0, 160);
    return `pi_${normalized || hash24(raw)}`;
  }
  const transcript = ctx?.sessionManager?.getSessionFile?.();
  return transcript ? `pi_transcript_${hash24(String(transcript))}` : null;
}

function loadCurrent(root, key) {
  if (!root || !key) return { eligible: false, reason: 'missing-context' };
  let pointer;
  try { pointer = JSON.parse(readFileSync(path.join(root, '.trellis/.runtime/sessions', `${key}.json`), 'utf8')); }
  catch { return { eligible: false, reason: 'missing-active-task' }; }
  const taskRef = typeof pointer?.current_task === 'string' ? pointer.current_task : '';
  let tasksRoot;
  try { tasksRoot = realpathSync(path.join(root, '.trellis/tasks')); }
  catch { return { eligible: false, reason: 'missing-tasks-root' }; }
  const candidate = path.resolve(root, taskRef);
  if (candidate !== tasksRoot && !candidate.startsWith(`${tasksRoot}${path.sep}`)) return { eligible: false, reason: 'task-outside-root' };
  let taskDir;
  try { taskDir = realpathSync(candidate); } catch { return { eligible: false, reason: 'missing-task' }; }
  if (taskDir !== tasksRoot && !taskDir.startsWith(`${tasksRoot}${path.sep}`)) return { eligible: false, reason: 'task-outside-root' };
  let task;
  try { task = JSON.parse(readFileSync(path.join(taskDir, 'task.json'), 'utf8')); } catch { return { eligible: false, reason: 'invalid-task' }; }
  if (task?.status !== 'in_progress' || !existsSync(path.join(taskDir, 'design.md')) || !existsSync(path.join(taskDir, 'implement.md'))) return { eligible: false, reason: 'not-complex-in-progress' };
  let plan;
  try { plan = parsePlan(taskRef, readFileSync(path.join(taskDir, 'implement.md'), 'utf8')); } catch { return { eligible: false, reason: 'invalid-plan' }; }
  const pending = plan.items.filter((item) => !item.checked && !item.ambiguous);
  if (!plan.valid || pending.length === 0) return { eligible: false, reason: 'invalid-or-empty-plan', taskRef, task, plan };
  return { eligible: true, root, contextKey: key, taskRef, task, plan };
}

function setToolActive(pi, enabled) {
  const active = pi.getActiveTools?.() ?? [];
  const next = enabled ? [...new Set([...active, TOOL_NAME])] : active.filter((name) => name !== TOOL_NAME);
  if (next.length !== active.length || next.some((name, i) => name !== active[i])) pi.setActiveTools?.(next);
}

export async function reconcileTool(pi, options = {}) {
  const root = findRoot(options.cwd ?? process.cwd());
  const current = loadCurrent(root, safeContext(options.contextKey) ?? contextKey(options.ctx));
  setToolActive(pi, current.eligible);
  return current;
}

function result(payload) {
  const details = { ...payload, guidance: TOOL_GUIDANCE };
  return { content: [{ type: 'text', text: JSON.stringify(details) }], details };
}

export default function nimlothWorkItemsExtension(pi) {
  pi.registerTool?.({
    name: TOOL_NAME,
    label: 'Nimloth Work Item',
    description: TOOL_GUIDANCE,
    parameters: {
      type: 'object', additionalProperties: false, required: ['action'],
      properties: {
        action: { type: 'string', enum: ['select', 'update', 'evidence', 'release'] },
        itemRef: { type: 'string', maxLength: 64 }, executor: { type: 'string', maxLength: 128 },
        state: { type: 'string', enum: ['working', 'verifying', 'delegated', 'waiting_human', 'waiting_external', 'blocked', 'failed'] },
        blocker: { type: 'string', maxLength: 512 }, nextAction: { type: 'string', maxLength: 512 },
        kind: { type: 'string', enum: ['artifact', 'test', 'command', 'commit', 'job', 'url'] },
        ref: { type: 'string', maxLength: 512 }, summary: { type: 'string', maxLength: 512 },
      },
    },
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const root = findRoot(ctx?.cwd ?? process.cwd());
      const key = contextKey(ctx);
      const current = loadCurrent(root, key);
      if (!current.eligible) return result({ ok: false, error: { code: 'inactive-task', message: current.reason } });
      const sessionId = ctx?.sessionManager?.getSessionId?.() ?? key;
      const action = { ...params };
      if (action.action === 'select' && !action.itemRef) {
        return result({ ok: false, error: { code: 'item-required', message: 'Select one current pending item ref' }, availableItems: current.plan.items.filter((item) => !item.checked && !item.ambiguous).map(({ ref, text, headingPath }) => ({ ref, text, headingPath })) });
      }
      if (action.action === 'select' && !action.executor) action.executor = `pi:${hash24(String(sessionId))}`;
      const output = await applyAction({ root, contextKey: key, sessionId: String(sessionId), taskRef: current.taskRef, taskStatus: current.task.status, plan: current.plan }, action);
      return result(output);
    },
  });
  setToolActive(pi, false);
  pi.on?.('session_start', async (_event, ctx) => { await reconcileTool(pi, { cwd: ctx?.cwd, ctx }); });
  pi.on?.('before_agent_start', async (_event, ctx) => { await reconcileTool(pi, { cwd: ctx?.cwd, ctx }); });
}
