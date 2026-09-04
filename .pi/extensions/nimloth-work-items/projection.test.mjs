import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { applyAction, parsePlan } from './core.mjs';
const exec = promisify(execFile);
const cli = path.resolve('.pi/extensions/nimloth-work-items/projection.mjs');

async function fixture({status='in_progress', planText='## Work\n- [ ] Build projection'}={}) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'nimloth-proj-'));
  const taskRef = '.trellis/tasks/09-03-demo'; const contextKey = 'pi_projection';
  await mkdir(path.join(root, taskRef), {recursive:true});
  await mkdir(path.join(root, '.trellis/.runtime/sessions'), {recursive:true});
  await writeFile(path.join(root, taskRef, 'task.json'), JSON.stringify({status}));
  await writeFile(path.join(root, taskRef, 'implement.md'), planText);
  await writeFile(path.join(root, taskRef, 'design.md'), '# D');
  await writeFile(path.join(root, '.trellis/.runtime/sessions', `${contextKey}.json`), JSON.stringify({current_task:taskRef}));
  return {root, taskRef, contextKey, plan:parsePlan(taskRef, planText)};
}
async function run(f) {
  const {stdout, stderr} = await exec(process.execPath, [cli, '--context-key', f.contextKey], {cwd:f.root, maxBuffer:128*1024});
  assert.equal(stderr, ''); return JSON.parse(stdout);
}

test('CLI returns bounded versioned projection and joins runtime without mutating artifacts', async () => {
  const f = await fixture(); const item=f.plan.items[0];
  await applyAction({root:f.root, contextKey:f.contextKey, sessionId:f.contextKey, taskRef:f.taskRef, taskStatus:'in_progress', plan:f.plan}, {action:'select', itemRef:item.ref, executor:'agent'});
  const output=await run(f);
  assert.equal(output.schemaVersion,1); assert.equal(output.root.path,f.root); assert.equal(output.current.text,'Build projection');
  assert.ok(Buffer.byteLength(JSON.stringify(output)) < 64*1024);
});

test('missing runtime and malformed pointers are visible non-blocking issues', async () => {
  const f=await fixture(); let output=await run(f);
  assert.equal(output.current,null); assert.ok(output.issues.some(i=>i.code==='runtime-missing'));
  await writeFile(path.join(f.root,'.trellis/.runtime/sessions',`${f.contextKey}.json`), JSON.stringify({current_task:'../../outside'}));
  output=await run(f); assert.equal(output.task,null); assert.ok(output.issues.some(i=>i.code==='task-outside-root'));
});

test('CLI rejects traversal context and still emits versioned issue JSON', async () => {
  const f=await fixture(); f.contextKey='../bad'; const output=await run(f);
  assert.equal(output.schemaVersion,1); assert.equal(output.task,null); assert.ok(output.issues.some(i=>i.code==='invalid-context'));
});

test('released and orphan assignments never become current', async () => {
  const f=await fixture(); const item=f.plan.items[0]; const input={root:f.root,contextKey:f.contextKey,sessionId:f.contextKey,taskRef:f.taskRef,taskStatus:'in_progress',plan:f.plan};
  await applyAction(input,{action:'select',itemRef:item.ref,executor:'agent'}); await applyAction(input,{action:'release'});
  assert.equal((await run(f)).current,null);
});

test('oversized plans produce a bounded fail-visible projection', async () => {
  const lines = Array.from({length:1200}, (_, i) => `- [ ] item ${i} ${'x'.repeat(60)}`).join('\n');
  const f = await fixture({planText:`## Large\n${lines}`});
  const output = await run(f);
  assert.equal(output.items.length, 0);
  assert.ok(output.issues.some((entry) => entry.code === 'projection-overflow'));
  assert.ok(Buffer.byteLength(JSON.stringify(output)) < 64*1024);
});
