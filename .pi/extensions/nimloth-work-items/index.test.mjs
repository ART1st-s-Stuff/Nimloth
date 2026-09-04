import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import extension, { reconcileTool } from './index.ts';

async function project({ status='in_progress', design=true, implement='## W\n- [ ] Do it', context='pi_s1' } = {}) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'nimloth-ext-'));
  const taskRef = '.trellis/tasks/09-03-demo';
  await mkdir(path.join(root, taskRef), {recursive:true});
  await mkdir(path.join(root, '.trellis/.runtime/sessions'), {recursive:true});
  await writeFile(path.join(root, taskRef, 'task.json'), JSON.stringify({status}));
  if (design) await writeFile(path.join(root, taskRef, 'design.md'), '# Design');
  if (implement !== null) await writeFile(path.join(root, taskRef, 'implement.md'), implement);
  await writeFile(path.join(root, '.trellis/.runtime/sessions', `${context}.json`), JSON.stringify({current_task: taskRef}));
  return {root, taskRef, context};
}

function harness(active=['read','bash']) {
  const handlers = new Map(); let tools = [...active]; let registered;
  const pi = { registerTool(t){registered=t}, on(name, fn){handlers.set(name, fn)}, getActiveTools(){return [...tools]}, setActiveTools(next){tools=[...next]} };
  return {pi, handlers, get tools(){return tools}, get registered(){return registered}};
}

test('activation matrix enables only complex in-progress valid plans', async () => {
  for (const [opts, enabled] of [
    [{}, true], [{status:'planning'}, false], [{status:'completed'}, false], [{design:false}, false],
    [{implement:null}, false], [{implement:'# none'}, false], [{implement:'- [ ] same\n- [ ] same'}, false],
  ]) {
    const p = await project(opts); const h = harness(['read','other']);
    await reconcileTool(h.pi, {cwd:p.root, contextKey:p.context});
    assert.equal(h.tools.includes('nimloth_work_item'), enabled, JSON.stringify(opts));
    assert.ok(h.tools.includes('read') && h.tools.includes('other'));
  }
});

test('reconcile removes only itself across task changes or missing task storage', async () => {
  const p = await project(); const h = harness(['read','nimloth_work_item','custom']);
  await reconcileTool(h.pi, {cwd:p.root, contextKey:'missing'});
  assert.deepEqual(h.tools, ['read','custom']);

  const emptyRoot = await mkdtemp(path.join(os.tmpdir(), 'nimloth-ext-empty-'));
  await mkdir(path.join(emptyRoot, '.trellis'));
  await assert.doesNotReject(reconcileTool(h.pi, {cwd:emptyRoot, contextKey:'pi_s1'}));
  assert.deepEqual(h.tools, ['read','custom']);
});

test('visibility guidance has no auto-discovered project skill metadata', () => {
  assert.equal(existsSync(path.resolve('.agents/skills/nimloth-work-item-visibility/SKILL.md')), false);
});

test('extension registers no prompt snippet/guidelines and reconciles lifecycle events', async () => {
  const h = harness(); extension(h.pi);
  assert.equal(h.registered.name, 'nimloth_work_item');
  for (const action of ['select', 'update', 'evidence', 'release']) assert.match(h.registered.description, new RegExp(action));
  assert.equal('promptSnippet' in h.registered, false);
  assert.equal('promptGuidelines' in h.registered, false);
  assert.ok(h.handlers.has('session_start'));
  assert.ok(h.handlers.has('before_agent_start'));
});

test('Pi session ids map to the upstream Trellis context key format and bound', async () => {
  for (const [sessionId, expected] of [
    ['session/id', 'pi_session_id'],
    ['s'.repeat(160), `pi_${'s'.repeat(160)}`],
  ]) {
    const p = await project({ context: expected }); const h = harness(); extension(h.pi);
    const ctx = { cwd:p.root, sessionManager:{ getSessionId(){return sessionId} } };
    await h.handlers.get('session_start')({}, ctx);
    assert.equal(h.tools.includes('nimloth_work_item'), true);
  }
});

test('select discovery is read-only, then the tool binds selection to current session', async () => {
  const p = await project(); const h = harness(); extension(h.pi);
  const ctx = { cwd:p.root, sessionManager:{ getSessionId(){return 's1'} } };
  await h.handlers.get('session_start')({}, ctx);
  const discovery = await h.registered.execute('call-1', {action:'select'}, null, null, ctx);
  assert.equal(discovery.details.ok, false);
  assert.equal(discovery.details.availableItems.length, 1);
  for (const action of ['select', 'update', 'evidence', 'release']) assert.match(discovery.details.guidance, new RegExp(action));
  const selected = await h.registered.execute('call-2', {action:'select', itemRef:discovery.details.availableItems[0].ref, executor:'agent'}, null, null, ctx);
  assert.equal(selected.details.ok, true);
  assert.equal(selected.details.runtime.contextKey, 'pi_s1');
});
