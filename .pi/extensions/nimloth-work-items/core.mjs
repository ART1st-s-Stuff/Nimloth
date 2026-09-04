import { createHash, randomUUID } from 'node:crypto';
import { closeSync, fsyncSync, openSync, renameSync, unlinkSync, writeSync } from 'node:fs';
import { mkdir, readFile, realpath } from 'node:fs/promises';
import path from 'node:path';

export const SCHEMA_VERSION = 1;
export const MAX_RUNTIME_BYTES = 16 * 1024;
export const MAX_TEXT = 512;
export const MAX_ITEM_TEXT = 2048;
export const MAX_HEADING_SEGMENT = 512;
export const MAX_HEADING_SEGMENTS = 32;
export const MAX_ITEMS = 2000;
export const MAX_ISSUES = 100;
export const MAX_DUPLICATE_LINES = 100;
export const MAX_TASK_REF = 512;
export const MAX_STATUS = 64;
export const MAX_EXECUTOR = 128;
export const MAX_SESSION = 256;
export const MAX_EVIDENCE = 8;
export const EVIDENCE_KINDS = new Set(['artifact', 'test', 'command', 'commit', 'job', 'url']);
const STATES = new Set(['working', 'verifying', 'delegated', 'waiting_human', 'waiting_external', 'blocked', 'failed']);
const TRANSITIONS = {
  working: new Set(['verifying', 'delegated', 'waiting_human', 'waiting_external', 'blocked', 'failed']),
  verifying: new Set(['working', 'delegated', 'waiting_human', 'waiting_external', 'blocked', 'failed']),
  delegated: new Set(['working', 'verifying', 'waiting_human', 'waiting_external', 'blocked', 'failed']),
  waiting_human: new Set(['working', 'verifying', 'blocked', 'failed']),
  waiting_external: new Set(['working', 'verifying', 'blocked', 'failed']),
  blocked: new Set(['working', 'verifying', 'waiting_human', 'waiting_external', 'failed']),
  failed: new Set(['working']),
};

const sha = (value) => createHash('sha256').update(value).digest('hex');
const normalized = (value) => String(value ?? '').normalize('NFC').trim().replace(/\s+/gu, ' ');
const issue = (code, message, extra = {}) => ({ code, message: String(message).slice(0, MAX_TEXT), ...extra });
const appendIssue = (issues, entry) => {
  if (issues.some((existing) => existing.code === 'issues-truncated')) return;
  if (issues.length < MAX_ISSUES) issues.push(entry);
  else issues[MAX_ISSUES - 1] = issue('issues-truncated', `Projection issues exceed ${MAX_ISSUES}`);
};
const boundedIssue = (entry) => {
  const result = issue(String(entry?.code ?? 'plan-issue').slice(0, 64), entry?.message ?? 'Plan issue');
  if (Number.isSafeInteger(entry?.line) && entry.line > 0) result.line = entry.line;
  if (Array.isArray(entry?.lines)) {
    result.lines = entry.lines.filter((line) => Number.isSafeInteger(line) && line > 0).slice(0, MAX_DUPLICATE_LINES);
    if (entry.lines.length > MAX_DUPLICATE_LINES || entry.linesTruncated) result.linesTruncated = true;
  }
  return result;
};
const safePart = (value, label) => {
  if (typeof value !== 'string' || !/^[A-Za-z0-9._-]{1,163}$/.test(value) || value === '.' || value === '..') {
    throw new Error(`Invalid ${label}`);
  }
  return value;
};
const bounded = (value, max, label, { nullable = true } = {}) => {
  if (value == null && nullable) return null;
  if (typeof value !== 'string' || value.length < 1 || value.length > max || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/u.test(value)) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
};
const timestamp = (value, label) => {
  bounded(value, 64, label, { nullable: false });
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u.test(value) || Number.isNaN(Date.parse(value))) {
    throw new Error(`Invalid ${label}`);
  }
  return new Date(value).toISOString();
};

export function rootFingerprint(root) {
  return sha(path.resolve(root));
}

export function parsePlan(taskRef, markdown) {
  const canonicalTask = normalized(taskRef);
  if (!canonicalTask || typeof markdown !== 'string') return { valid: false, fingerprint: sha('invalid'), items: [], issues: [issue('invalid-plan', 'Task ref and Markdown are required')] };
  const headings = [];
  const headingIdentities = [];
  const items = [];
  const issues = [];
  if (canonicalTask.length > MAX_TASK_REF) appendIssue(issues, issue('task-ref-too-long', `Task ref exceeds ${MAX_TASK_REF}`));
  for (const [index, raw] of markdown.split(/\r?\n/u).entries()) {
    const heading = /^(#{1,6})\s+(.+?)\s*#*\s*$/u.exec(raw);
    if (heading) {
      const level = heading[1].length;
      const fullHeading = normalized(heading[2]);
      headings.length = level - 1;
      headingIdentities.length = level - 1;
      headings[level - 1] = fullHeading.slice(0, MAX_HEADING_SEGMENT);
      headingIdentities[level - 1] = fullHeading;
      if (fullHeading.length > MAX_HEADING_SEGMENT) appendIssue(issues, issue('heading-too-long', `Heading at line ${index + 1} exceeds ${MAX_HEADING_SEGMENT}`, { line: index + 1 }));
      continue;
    }
    const checkboxLike = /^\s*[-*+]\s+\[([^\]]*)\](?:\s*(.*))?$/u.exec(raw);
    if (!checkboxLike) {
      if (/^\s*[-*+]\s+\[/u.test(raw)) appendIssue(issues, issue('malformed-checkbox', `Malformed checkbox at line ${index + 1}`, { line: index + 1 }));
      continue;
    }
    const marker = checkboxLike[1];
    const fullText = String(checkboxLike[2] ?? '').trim();
    if (!/^[ xX]$/u.test(marker)) {
      appendIssue(issues, issue('malformed-checkbox', `Unsupported checkbox marker at line ${index + 1}`, { line: index + 1 }));
      continue;
    }
    if (!normalized(fullText)) {
      appendIssue(issues, issue('empty-item', `Empty checkbox at line ${index + 1}`, { line: index + 1 }));
      continue;
    }
    if (items.length >= MAX_ITEMS) {
      if (!issues.some((entry) => entry.code === 'items-truncated')) appendIssue(issues, issue('items-truncated', `Plan exceeds ${MAX_ITEMS} items`));
      continue;
    }
    if (fullText.length > MAX_ITEM_TEXT) appendIssue(issues, issue('item-text-too-long', `Item at line ${index + 1} exceeds ${MAX_ITEM_TEXT}`, { line: index + 1 }));
    const headingPath = headings.filter(Boolean).slice(0, MAX_HEADING_SEGMENTS);
    const identityText = normalized(fullText);
    const identityHeading = headingIdentities.filter(Boolean).slice(0, MAX_HEADING_SEGMENTS);
    const ref = sha(JSON.stringify([SCHEMA_VERSION, canonicalTask, identityHeading, identityText]));
    items.push({ ref, text: fullText.slice(0, MAX_ITEM_TEXT), checked: marker.toLowerCase() === 'x', headingPath, line: index + 1, ambiguous: false, identityText, identityHeading });
  }
  const duplicateGroups = new Map();
  for (const item of items) {
    const key = JSON.stringify([item.identityHeading, item.identityText]);
    duplicateGroups.set(key, [...(duplicateGroups.get(key) ?? []), item]);
  }
  for (const group of duplicateGroups.values()) {
    if (group.length < 2) continue;
    for (const item of group) item.ambiguous = true;
    appendIssue(issues, issue('duplicate-item', 'Duplicate normalized checkbox text in the same heading', { lines: group.slice(0, MAX_DUPLICATE_LINES).map((item) => item.line), ...(group.length > MAX_DUPLICATE_LINES ? { linesTruncated: true } : {}) }));
  }
  const fingerprint = sha(JSON.stringify(items.map(({ ref, identityHeading }) => [identityHeading, ref])));
  return { valid: issues.length === 0, fingerprint, items: items.map(({ identityText: _, identityHeading: __, ...item }) => item), issues };
}

export function runtimeFile(root, contextKey) {
  safePart(contextKey, 'context key');
  return path.join(path.resolve(root), '.local', 'nimloth-work-items', 'v1', rootFingerprint(root), `${contextKey}.json`);
}

async function canonicalRoot(root) {
  try { return await realpath(root); } catch { return path.resolve(root); }
}

async function expectedMeta(input) {
  safePart(input.contextKey, 'context key');
  bounded(input.taskRef, MAX_TASK_REF, 'task ref', { nullable: false });
  if (!/^\.trellis\/tasks\/[A-Za-z0-9._-]+$/u.test(input.taskRef)) throw new Error('Invalid task ref');
  if (input.sessionId != null) bounded(input.sessionId, MAX_SESSION, 'session id', { nullable: false });
  if (!input.plan || typeof input.plan.fingerprint !== 'string') throw new Error('Invalid plan');
  const canonical = await canonicalRoot(input.root);
  return { canonical, fingerprint: rootFingerprint(canonical) };
}

function validateRuntime(data, input, meta) {
  const exactKeys = (object, allowed) => object && typeof object === 'object' && !Array.isArray(object) && Object.keys(object).every((key) => allowed.has(key));
  const topKeys = new Set(['schemaVersion', 'rootFingerprint', 'contextKey', 'sessionId', 'taskRef', 'planFingerprint', 'assignment']);
  const assignmentKeys = new Set(['itemRef', 'executor', 'state', 'since', 'updatedAt', 'blocker', 'nextAction', 'evidence', 'releasedAt']);
  const evidenceKeys = new Set(['kind', 'ref', 'summary', 'at']);
  if (!exactKeys(data, topKeys) || data.schemaVersion !== SCHEMA_VERSION) return issue('runtime-schema', 'Unsupported runtime schema');
  if (data.rootFingerprint !== meta.fingerprint) return issue('root-mismatch', 'Runtime belongs to another root');
  if (data.contextKey !== input.contextKey || (input.sessionId != null && data.sessionId !== input.sessionId)) return issue('context-mismatch', 'Runtime belongs to another context/session');
  if (data.taskRef !== input.taskRef) return issue('task-mismatch', 'Runtime belongs to another task');
  if (data.planFingerprint !== input.plan.fingerprint) return issue('plan-stale', 'Runtime plan fingerprint is stale');
  const assignment = data.assignment;
  if (!exactKeys(assignment, assignmentKeys) || typeof assignment.itemRef !== 'string' || !/^[a-f0-9]{64}$/u.test(assignment.itemRef) || !STATES.has(assignment.state) || !Array.isArray(assignment.evidence) || assignment.evidence.length > MAX_EVIDENCE) return issue('runtime-schema', 'Runtime assignment is invalid');
  try {
    bounded(assignment.executor, MAX_EXECUTOR, 'executor', { nullable: false });
    timestamp(assignment.since, 'since');
    timestamp(assignment.updatedAt, 'updated at');
    bounded(assignment.blocker, MAX_TEXT, 'blocker');
    bounded(assignment.nextAction, MAX_TEXT, 'next action');
    if (assignment.releasedAt != null) timestamp(assignment.releasedAt, 'released at');
    for (const evidence of assignment.evidence) {
      if (!exactKeys(evidence, evidenceKeys) || !EVIDENCE_KINDS.has(evidence.kind)) throw new Error('Invalid evidence');
      bounded(evidence.ref, MAX_TEXT, 'evidence ref', { nullable: false });
      bounded(evidence.summary, MAX_TEXT, 'evidence summary');
      timestamp(evidence.at, 'evidence timestamp');
    }
  } catch { return issue('runtime-schema', 'Runtime assignment is invalid'); }
  return null;
}

export async function readRuntime(input) {
  let meta;
  try { meta = await expectedMeta(input); } catch (error) { return { issue: issue('invalid-input', error.message) }; }
  const file = runtimeFile(meta.canonical, input.contextKey);
  let raw;
  try { raw = await readFile(file, 'utf8'); }
  catch (error) { return error?.code === 'ENOENT' ? { issue: issue('runtime-missing', 'No live work-item assignment') } : { issue: issue('runtime-read', 'Unable to read runtime') }; }
  if (Buffer.byteLength(raw) > MAX_RUNTIME_BYTES) return { issue: issue('runtime-oversized', 'Runtime exceeds size limit') };
  let data;
  try { data = JSON.parse(raw); } catch { return { issue: issue('runtime-corrupt', 'Runtime JSON is corrupt') }; }
  const invalid = validateRuntime(data, input, meta);
  return invalid ? { issue: invalid, runtime: data } : data;
}

async function atomicWrite(file, value) {
  const json = `${JSON.stringify(value, null, 2)}\n`;
  if (Buffer.byteLength(json) > MAX_RUNTIME_BYTES) throw new Error('Runtime exceeds size limit');
  await mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${randomUUID()}.tmp`);
  let fd;
  try {
    fd = openSync(temp, 'wx', 0o600);
    const bytes = Buffer.from(json);
    let offset = 0;
    while (offset < bytes.length) offset += writeSync(fd, bytes, offset, bytes.length - offset);
    fsyncSync(fd); closeSync(fd); fd = undefined;
    renameSync(temp, file);
    const dirFd = openSync(path.dirname(file), 'r');
    try { fsyncSync(dirFd); } finally { closeSync(dirFd); }
  } finally {
    if (fd !== undefined) closeSync(fd);
    try { unlinkSync(temp); } catch (error) { if (error?.code !== 'ENOENT') throw error; }
  }
}

function activeItem(plan, ref) { return plan.items.find((item) => item.ref === ref); }
function fail(code, message) { return { ok: false, error: issue(code, message) }; }

export async function applyAction(input, action) {
  let meta;
  try { meta = await expectedMeta(input); } catch (error) { return fail('invalid-input', error.message); }
  if (input.sessionId == null) return fail('invalid-input', 'Session id is required');
  if (input.taskStatus !== 'in_progress' || !input.plan.valid) return fail('inactive-task', 'Task is not eligible for live work items');
  let now;
  try { now = timestamp(input.now ?? new Date().toISOString(), 'action timestamp'); }
  catch (error) { return fail('invalid-input', error.message); }
  let loaded = await readRuntime({ ...input, root: meta.canonical });
  const missing = loaded?.issue?.code === 'runtime-missing';
  if (action?.action === 'select') {
    if (!missing && !loaded.issue && !loaded.assignment.releasedAt) return fail('already-selected', 'An assignment is already active');
    if (!missing && loaded.issue && loaded.issue.code !== 'plan-stale' && loaded.issue.code !== 'task-mismatch') return fail(loaded.issue.code, loaded.issue.message);
    const item = activeItem(input.plan, action.itemRef);
    if (!item || item.checked || item.ambiguous) return fail('invalid-item', 'Item is missing, checked, or ambiguous');
    let executor;
    try { executor = bounded(action.executor, MAX_EXECUTOR, 'executor', { nullable: false }); } catch (error) { return fail('invalid-input', error.message); }
    loaded = {
      schemaVersion: SCHEMA_VERSION, rootFingerprint: meta.fingerprint, contextKey: input.contextKey,
      sessionId: input.sessionId, taskRef: input.taskRef, planFingerprint: input.plan.fingerprint,
      assignment: { itemRef: item.ref, executor, state: 'working', since: now, updatedAt: now, blocker: null, nextAction: null, evidence: [] },
    };
  } else {
    if (loaded.issue) return fail(loaded.issue.code, loaded.issue.message);
    const assignment = loaded.assignment;
    if (assignment.releasedAt) return fail('released', 'Assignment is released');
    const item = activeItem(input.plan, assignment.itemRef);
    if (!item || item.ambiguous) return fail('orphan', 'Assigned item is no longer selectable');
    if (item.checked) return fail('checked-conflict', 'Assigned item is checked in the plan');
    if (action?.action === 'update') {
      if (!STATES.has(action.state) || !TRANSITIONS[assignment.state]?.has(action.state)) return fail('invalid-transition', 'State transition is not allowed');
      let blocker, nextAction;
      try {
        blocker = bounded(action.blocker, MAX_TEXT, 'blocker');
        nextAction = bounded(action.nextAction, MAX_TEXT, 'next action');
      } catch (error) { return fail('invalid-input', error.message); }
      if (action.state === 'blocked' && !blocker) return fail('blocker-required', 'Blocked state requires a blocker');
      assignment.state = action.state; assignment.blocker = blocker; assignment.nextAction = nextAction; assignment.updatedAt = now;
    } else if (action?.action === 'evidence') {
      if (!EVIDENCE_KINDS.has(action.kind)) return fail('invalid-evidence', 'Evidence kind is not allowed');
      let ref, summary;
      try { ref = bounded(action.ref, MAX_TEXT, 'evidence ref', { nullable: false }); summary = bounded(action.summary, MAX_TEXT, 'evidence summary'); }
      catch (error) { return fail('invalid-input', error.message); }
      assignment.evidence = [...assignment.evidence, { kind: action.kind, ref, summary, at: now }].slice(-MAX_EVIDENCE);
      assignment.updatedAt = now;
    } else if (action?.action === 'release') {
      assignment.releasedAt = now; assignment.updatedAt = now;
    } else return fail('invalid-action', 'Unknown action');
  }
  const file = runtimeFile(meta.canonical, input.contextKey);
  try { await atomicWrite(file, loaded); } catch (error) { return fail('runtime-write', error.message); }
  return { ok: true, runtime: loaded };
}

export async function buildProjection(input) {
  const canonical = await canonicalRoot(input.root);
  const root = { path: canonical, fingerprint: rootFingerprint(canonical) };
  const rawTaskRef = typeof input.taskRef === 'string' ? input.taskRef : '';
  const rawStatus = typeof input.taskStatus === 'string' ? input.taskStatus : '';
  const safeTaskRef = rawTaskRef.slice(0, MAX_TASK_REF);
  const safeStatus = rawStatus.slice(0, MAX_STATUS);
  const fingerprint = typeof input.plan?.fingerprint === 'string' && /^[a-f0-9]{64}$/u.test(input.plan.fingerprint) ? input.plan.fingerprint : sha('invalid-plan-fingerprint');
  const task = { ref: safeTaskRef, status: safeStatus, planFingerprint: fingerprint };
  const issues = [];
  if (rawTaskRef.length > MAX_TASK_REF || typeof input.taskRef !== 'string') appendIssue(issues, issue('task-ref-too-long', `Task ref must be at most ${MAX_TASK_REF} characters`));
  if (rawStatus.length > MAX_STATUS || typeof input.taskStatus !== 'string') appendIssue(issues, issue('status-invalid', `Task status must be at most ${MAX_STATUS} characters`));
  const inactive = input.taskStatus !== 'in_progress';
  if (inactive) appendIssue(issues, issue('inactive-task', 'Live work-item activity is unavailable unless task status is in_progress'));

  const sourceItems = Array.isArray(input.plan?.items) ? input.plan.items : [];
  const items = sourceItems.slice(0, MAX_ITEMS).map((item) => ({
    ref: typeof item?.ref === 'string' && /^[a-f0-9]{64}$/u.test(item.ref) ? item.ref : sha(String(item?.ref ?? 'invalid-item')),
    text: String(item?.text ?? '').slice(0, MAX_ITEM_TEXT),
    checked: item?.checked === true,
    headingPath: (Array.isArray(item?.headingPath) ? item.headingPath : []).slice(0, MAX_HEADING_SEGMENTS).map((segment) => String(segment).slice(0, MAX_HEADING_SEGMENT)),
    line: Number.isSafeInteger(item?.line) && item.line > 0 ? item.line : 1,
    ambiguous: item?.ambiguous === true,
  }));
  if (sourceItems.length > MAX_ITEMS) appendIssue(issues, issue('items-truncated', `Plan exceeds ${MAX_ITEMS} items`));
  if (inactive) {
    for (const entry of Array.isArray(input.plan?.issues) ? input.plan.issues : []) appendIssue(issues, boundedIssue(entry));
    return { schemaVersion: SCHEMA_VERSION, root, task, items, current: null, issues };
  }

  const loaded = await readRuntime({ ...input, root: canonical });
  let current = null;
  if (loaded.issue) {
    appendIssue(issues, boundedIssue(loaded.issue));
    if (loaded.issue.code === 'plan-stale' && loaded.runtime?.assignment) {
      const same = activeItem(input.plan, loaded.runtime.assignment.itemRef);
      if (same?.checked) appendIssue(issues, issue('checked-conflict', 'Assigned item is now checked'));
      else if (!same) appendIssue(issues, issue('orphan', 'Assigned item no longer exists'));
    }
  } else if (!loaded.assignment.releasedAt) {
    const item = activeItem(input.plan, loaded.assignment.itemRef);
    const projectedItem = items.find((candidate) => candidate.ref === loaded.assignment.itemRef);
    if (!item || !projectedItem) appendIssue(issues, issue('orphan', 'Assigned item no longer exists'));
    else if (item.checked) appendIssue(issues, issue('checked-conflict', 'Assigned item is now checked'));
    else if (item.ambiguous) appendIssue(issues, issue('ambiguous-assignment', 'Assigned item is ambiguous'));
    else current = { itemRef: projectedItem.ref, text: projectedItem.text, headingPath: projectedItem.headingPath, executor: { name: loaded.assignment.executor, sessionId: loaded.sessionId }, state: loaded.assignment.state, since: loaded.assignment.since, updatedAt: loaded.assignment.updatedAt, blocker: loaded.assignment.blocker, nextAction: loaded.assignment.nextAction, evidence: loaded.assignment.evidence };
  }
  for (const entry of Array.isArray(input.plan?.issues) ? input.plan.issues : []) appendIssue(issues, boundedIssue(entry));
  return { schemaVersion: SCHEMA_VERSION, root, task, items, current, issues };
}
