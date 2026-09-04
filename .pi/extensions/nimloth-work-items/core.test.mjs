import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  parsePlan, runtimeFile, applyAction, readRuntime, buildProjection,
  MAX_RUNTIME_BYTES, MAX_ITEMS, MAX_ISSUES, MAX_ITEM_TEXT, MAX_HEADING_SEGMENT,
  MAX_HEADING_SEGMENTS, MAX_DUPLICATE_LINES, rootFingerprint,
} from './core.mjs';

const taskRef = '.trellis/tasks/09-03-demo';
const markdown = `# Plan\n## Build   parser\n- [ ]  Parse   ordinary markdown  \n- [x] Finished\n## Runtime\n* [ ] Persist state\n+ [X] Verify output\n`;

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'nimloth-wi-'));
  await mkdir(path.join(root, '.local'), { recursive: true });
  return root;
}

function firstPending(plan) { return plan.items.find((item) => !item.checked); }

const base = (root, plan, extra = {}) => ({
  root, contextKey: 'pi_session-1', sessionId: 'session-1', taskRef,
  taskStatus: 'in_progress', plan, now: '2026-09-03T00:00:00.000Z', ...extra,
});

test('parser recognizes ATX headings and ordinary checkboxes in order', () => {
  const plan = parsePlan(taskRef, markdown);
  assert.equal(plan.valid, true);
  assert.deepEqual(plan.items.map(({ text, checked, headingPath }) => ({ text, checked, headingPath })), [
    { text: 'Parse   ordinary markdown', checked: false, headingPath: ['Plan', 'Build parser'] },
    { text: 'Finished', checked: true, headingPath: ['Plan', 'Build parser'] },
    { text: 'Persist state', checked: false, headingPath: ['Plan', 'Runtime'] },
    { text: 'Verify output', checked: true, headingPath: ['Plan', 'Runtime'] },
  ]);
  assert.match(plan.items[0].ref, /^[a-f0-9]{64}$/);
  assert.doesNotMatch(plan.items[0].text, /\[W-/);
});

test('identity normalizes whitespace/case markers but changes with text or heading', () => {
  const a = parsePlan(taskRef, '## A\n- [ ] hello   world').items[0];
  const b = parsePlan(taskRef, '## A\n- [x]  hello world  ').items[0];
  const textChanged = parsePlan(taskRef, '## A\n- [ ] hello worlds').items[0];
  const headingChanged = parsePlan(taskRef, '## B\n- [ ] hello world').items[0];
  assert.equal(a.ref, b.ref);
  assert.notEqual(a.ref, textChanged.ref);
  assert.notEqual(a.ref, headingChanged.ref);
});

test('duplicates in one normalized heading are ambiguous and issues are visible', () => {
  const plan = parsePlan(taskRef, '## A\n- [ ] duplicate text\n- [ ] duplicate   text');
  assert.equal(plan.valid, false);
  assert.ok(plan.items.every((item) => item.ambiguous));
  assert.equal(plan.issues[0].code, 'duplicate-item');
});

test('projection-v1 parser bounds item text, headings, items, issues, and duplicate lines', () => {
  const itemBoundary = parsePlan(taskRef, `# ${'h'.repeat(MAX_HEADING_SEGMENT)}\n- [ ] ${'i'.repeat(MAX_ITEM_TEXT)}`);
  assert.equal(itemBoundary.valid, true);
  assert.equal(itemBoundary.items[0].text.length, MAX_ITEM_TEXT);
  assert.equal(itemBoundary.items[0].headingPath[0].length, MAX_HEADING_SEGMENT);

  const overContent = parsePlan(taskRef, `# ${'h'.repeat(MAX_HEADING_SEGMENT + 1)}\n- [ ] ${'i'.repeat(MAX_ITEM_TEXT + 1)}`);
  assert.equal(overContent.valid, false);
  assert.equal(overContent.items[0].text.length, MAX_ITEM_TEXT);
  assert.equal(overContent.items[0].headingPath[0].length, MAX_HEADING_SEGMENT);
  assert.ok(overContent.issues.some((entry) => entry.code === 'heading-too-long'));
  assert.ok(overContent.issues.some((entry) => entry.code === 'item-text-too-long'));
  assert.ok(overContent.items.every((item) => item.headingPath.length <= MAX_HEADING_SEGMENTS));

  const tooManyItems = parsePlan(taskRef, Array.from({length: MAX_ITEMS + 1}, (_, i) => `- [ ] item-${i}`).join('\n'));
  assert.equal(tooManyItems.items.length, MAX_ITEMS);
  assert.ok(tooManyItems.issues.some((entry) => entry.code === 'items-truncated'));

  const tooManyIssues = parsePlan(taskRef, Array.from({length: MAX_ISSUES + 20}, () => '- [bad] invalid').join('\n'));
  assert.equal(tooManyIssues.issues.length, MAX_ISSUES);
  assert.equal(tooManyIssues.issues.at(-1).code, 'issues-truncated');

  const duplicates = parsePlan(taskRef, Array.from({length: MAX_DUPLICATE_LINES + 1}, () => '- [ ] same').join('\n'));
  const duplicateIssue = duplicates.issues.find((entry) => entry.code === 'duplicate-item');
  assert.equal(duplicateIssue.lines.length, MAX_DUPLICATE_LINES);
  assert.equal(duplicateIssue.linesTruncated, true);
});

test('malformed checkbox is reported without inventing an item', () => {
  const plan = parsePlan(taskRef, '## A\n- [maybe] nope\n- [ ]\n- [x missing bracket');
  assert.equal(plan.items.length, 0);
  assert.deepEqual(plan.issues.map((issue) => issue.code), ['malformed-checkbox', 'empty-item', 'malformed-checkbox']);
});

test('runtime path is local, versioned, and rejects unsafe context/task refs', async () => {
  const root = await fixture();
  assert.equal(runtimeFile(root, 'safe_key'), path.join(root, '.local/nimloth-work-items/v1', rootFingerprint(root), 'safe_key.json'));
  assert.throws(() => runtimeFile(root, '../escape'), /context/i);
  const unsafeRef = '../../outside';
  const unsafePlan = parsePlan(unsafeRef, '- [ ] escape');
  const attempt = await applyAction(base(root, unsafePlan, { taskRef: unsafeRef }), { action: 'select', itemRef: unsafePlan.items[0].ref, executor: 'agent' });
  assert.equal(attempt.error.code, 'invalid-input');
});

test('select/update/evidence/release persist only bounded references', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  let result = await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'agent-a' });
  assert.equal(result.ok, true);
  result = await applyAction(base(root, plan, { now: '2026-09-03T00:01:00.000Z' }), { action: 'update', state: 'blocked', blocker: 'Needs API decision', nextAction: 'Ask owner' });
  assert.equal(result.runtime.assignment.state, 'blocked');
  result = await applyAction(base(root, plan), { action: 'evidence', kind: 'test', ref: 'core.test.mjs', summary: 'focused pass' });
  assert.equal(result.runtime.assignment.evidence.length, 1);
  const raw = await readFile(runtimeFile(root, 'pi_session-1'), 'utf8');
  for (const forbidden of ['Parse ordinary markdown', 'Finished', 'checkbox', 'assistant response', 'toolOutput', 'prompt']) assert.equal(raw.includes(forbidden), false);
  assert.ok(Buffer.byteLength(raw) <= MAX_RUNTIME_BYTES);
  result = await applyAction(base(root, plan), { action: 'release' });
  assert.ok(result.runtime.assignment.releasedAt);
});

test('runtime validation and transitions fail closed without rewriting', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  assert.equal((await applyAction(base(root, plan), { action: 'select', itemRef: plan.items[1].ref, executor: 'x' })).ok, false);
  assert.equal((await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'x' })).ok, true);
  const before = await readFile(runtimeFile(root, 'pi_session-1'), 'utf8');
  assert.equal((await applyAction(base(root, plan), { action: 'update', state: 'working' })).ok, false); // same-state transition
  assert.equal((await applyAction(base(root, plan), { action: 'update', state: 'blocked' })).ok, false); // blocker required
  assert.equal((await applyAction(base(root, plan, { taskRef: 'other' }), { action: 'update', state: 'verifying' })).ok, false);
  assert.equal((await applyAction(base(root, plan, { now: 'not-an-instant' }), { action: 'update', state: 'verifying' })).error.code, 'invalid-input');
  assert.equal(await readFile(runtimeFile(root, 'pi_session-1'), 'utf8'), before);

  const malformedTime = JSON.parse(before);
  malformedTime.assignment.updatedAt = 'not-an-instant';
  await writeFile(runtimeFile(root, 'pi_session-1'), JSON.stringify(malformedTime));
  assert.equal((await readRuntime(base(root, plan))).issue.code, 'runtime-schema');
});

test('bounds reject oversized fields and evidence stays FIFO capped', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  assert.equal((await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'x'.repeat(129) })).ok, false);
  assert.equal((await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'x' })).ok, true);
  assert.equal((await applyAction(base(root, plan), { action: 'evidence', kind: 'raw-output', ref: 'x' })).ok, false);
  for (let i = 0; i < 10; i++) await applyAction(base(root, plan), { action: 'evidence', kind: 'test', ref: `t${i}`, summary: 'ok' });
  const runtime = await readRuntime(base(root, plan));
  assert.equal(runtime.assignment.evidence.length, 8);
  assert.equal(runtime.assignment.evidence[0].ref, 't2');
});

test('consumer string boundaries accept exact maxima and reject max plus one', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  const exact = base(root, plan, { sessionId: 's'.repeat(256) });
  assert.equal((await applyAction(exact, { action: 'select', itemRef: item.ref, executor: 'e'.repeat(128) })).ok, true);
  assert.equal((await applyAction(exact, { action: 'update', state: 'verifying', nextAction: 'n'.repeat(512) })).ok, true);
  assert.equal((await applyAction(exact, { action: 'update', state: 'blocked', blocker: 'b'.repeat(512) })).ok, true);
  assert.equal((await applyAction(exact, { action: 'evidence', kind: 'test', ref: 'r'.repeat(512), summary: 'm'.repeat(512) })).ok, true);

  const anotherRoot = await fixture();
  assert.equal((await applyAction(base(anotherRoot, plan, { sessionId: 's'.repeat(257) }), { action: 'select', itemRef: item.ref, executor: 'agent' })).error.code, 'invalid-input');
  assert.equal((await applyAction(exact, { action: 'update', state: 'working', nextAction: 'n'.repeat(513) })).error.code, 'invalid-input');
  assert.equal((await applyAction(exact, { action: 'update', state: 'working', blocker: 'b'.repeat(513) })).error.code, 'invalid-input');
  assert.equal((await applyAction(exact, { action: 'evidence', kind: 'test', ref: 'r'.repeat(513) })).error.code, 'invalid-input');
  assert.equal((await applyAction(exact, { action: 'evidence', kind: 'test', ref: 'ok', summary: 'm'.repeat(513) })).error.code, 'invalid-input');
});

test('atomic writes leave no temporary payload behind', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  await applyAction(base(root, plan), { action: 'select', itemRef: firstPending(plan).ref, executor: 'agent' });
  const names = await readdir(path.dirname(runtimeFile(root, 'pi_session-1')));
  assert.deepEqual(names, ['pi_session-1.json']);
});

test('root, context, task, plan, and item mismatches fail visibly', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'agent' });
  const file = runtimeFile(root, 'pi_session-1');
  const original = JSON.parse(await readFile(file, 'utf8'));
  for (const [field, value, code] of [
    ['rootFingerprint', '0'.repeat(64), 'root-mismatch'],
    ['contextKey', 'another', 'context-mismatch'],
    ['taskRef', 'another-task', 'task-mismatch'],
    ['planFingerprint', '0'.repeat(64), 'plan-stale'],
  ]) {
    await writeFile(file, JSON.stringify({ ...original, [field]: value }));
    assert.equal((await readRuntime(base(root, plan))).issue.code, code);
  }
  await writeFile(file, JSON.stringify({ ...original, assignment: { ...original.assignment, itemRef: '0'.repeat(64) } }));
  assert.ok((await buildProjection(base(root, plan))).issues.some((entry) => entry.code === 'orphan'));
});

test('unknown runtime payload fields are rejected and never re-persisted', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'agent' });
  const file = runtimeFile(root, 'pi_session-1');
  const injected = { ...JSON.parse(await readFile(file, 'utf8')), assistantResponse: 'must not survive' };
  await writeFile(file, JSON.stringify(injected));
  assert.equal((await applyAction(base(root, plan), { action: 'update', state: 'verifying' })).error.code, 'runtime-schema');
  assert.equal(JSON.parse(await readFile(file, 'utf8')).assistantResponse, 'must not survive');
});

test('corrupt runtime is fail-visible and never auto-repaired', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const file = runtimeFile(root, 'pi_session-1');
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, '{bad');
  const loaded = await readRuntime(base(root, plan));
  assert.equal(loaded.issue.code, 'runtime-corrupt');
  assert.equal(await readFile(file, 'utf8'), '{bad');
});

test('projection suppresses current and reports every inactive task status', async () => {
  for (const status of ['planning', 'completed', 'cancelled', '', 'x'.repeat(65)]) {
    const root = await fixture();
    const plan = parsePlan(taskRef, markdown);
    await applyAction(base(root, plan), { action: 'select', itemRef: firstPending(plan).ref, executor: 'agent' });
    const projection = await buildProjection(base(root, plan, { taskStatus: status }));
    assert.equal(projection.current, null);
    assert.ok(projection.issues.some((entry) => entry.code === 'inactive-task'));
    assert.ok(projection.task.status.length <= 64);
  }
});

test('projection bounds task ref and issue arrays before returning schema v1', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  plan.issues = Array.from({length: MAX_ISSUES + 10}, (_, i) => ({code:'test', message:`issue-${i}`}));
  const projection = await buildProjection(base(root, plan, { taskRef: `.trellis/tasks/${'t'.repeat(600)}` }));
  assert.ok(projection.task.ref.length <= 512);
  assert.ok(projection.issues.length <= MAX_ISSUES);
  assert.ok(projection.issues.some((entry) => entry.code === 'task-ref-too-long'));
});

test('projection joins text at read time and reports stale/orphan/checked conflicts', async () => {
  const root = await fixture();
  const plan = parsePlan(taskRef, markdown);
  const item = firstPending(plan);
  await applyAction(base(root, plan), { action: 'select', itemRef: item.ref, executor: 'agent' });
  let projection = await buildProjection(base(root, plan));
  assert.equal(projection.schemaVersion, 1);
  assert.equal(projection.current.text, item.text);
  const changed = parsePlan(taskRef, markdown.replace('Parse   ordinary markdown', 'Changed text'));
  projection = await buildProjection(base(root, changed));
  assert.ok(projection.issues.some((issue) => issue.code === 'plan-stale'));
  assert.equal(projection.current, null);
  const checked = parsePlan(taskRef, markdown.replace('- [ ]  Parse', '- [x]  Parse'));
  projection = await buildProjection(base(root, checked));
  assert.ok(projection.issues.some((issue) => issue.code === 'checked-conflict'));
});
