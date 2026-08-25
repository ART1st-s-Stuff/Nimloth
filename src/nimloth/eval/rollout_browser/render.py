"""Dependency-free offline HTML for evaluation manifests and rollout audits."""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Mapping
from typing import Any


def _encoded(value: Mapping[str, Any]) -> str:
    return base64.b64encode(
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).decode("ascii")


def render_rollout_html(audit: Mapping[str, Any]) -> str:
    capabilities = audit["capabilities"]
    task = html.escape(str(audit["task"]))
    optional_styles = ""
    direct_label = ""
    if capabilities["direct_q"]:
        direct_label = "<th>Direct Q</th>"
    planner_label = ""
    if capabilities["planner"]:
        planner_label = "<th>Planner root</th><th>Visits</th>"
    state_value_metric = ""
    if capabilities["state_value"]:
        state_value_metric = "${stateValue==null?'':`<div class=\\\"metric\\\"><span class=\\\"label\\\">Current state value</span><strong>${f(stateValue)}</strong></div>`}"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rollout · {html.escape(str(audit['identity'].get('record_id') or audit['identity'].get('rollout_sample_id')))}</title>
<style>
:root{{--bg:#0b1020;--panel:#131b31;--panel2:#192541;--text:#e8edf8;--muted:#9fb0cc;--blue:#61a8ff;--green:#55d6a0;--red:#ff758f;--gold:#ffd166;--border:#2a395d}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1020;color:var(--text);font-family:Inter,system-ui,sans-serif}}.shell{{max-width:1450px;margin:auto;padding:20px}}header,.card{{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:18px}}h1,h2,h3{{margin:0 0 12px}}.task{{background:#0e1629;border:1px solid var(--border);border-radius:12px;padding:13px 15px;font-size:17px;line-height:1.45}}.label,.note{{font-size:12px;color:var(--muted)}}.pills{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.pill{{border:1px solid var(--border);border-radius:999px;padding:5px 9px;font-size:12px}}.failure{{color:var(--red)}}.success{{color:var(--green)}}.layout{{display:grid;grid-template-columns:210px 1fr;gap:16px;margin-top:16px}}nav{{position:sticky;top:12px;align-self:start;max-height:96vh;overflow:auto;background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:8px}}nav button{{width:100%;border:0;background:transparent;color:var(--muted);text-align:left;padding:9px;border-radius:8px;cursor:pointer}}nav button.active,nav button:hover{{background:var(--panel2);color:white}}.card{{display:none}}.card.active{{display:block}}.top{{display:grid;grid-template-columns:minmax(280px,520px) 1fr;gap:16px}}img.obs{{width:100%;border:1px solid var(--border);border-radius:12px}}.metrics{{display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:8px}}.metric{{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:10px}}.metric strong{{display:block;margin-top:4px;color:var(--gold)}}details{{background:#0f172a;border:1px solid var(--border);border-radius:10px;padding:10px;margin-top:12px}}summary{{cursor:pointer;color:var(--blue)}}pre{{white-space:pre-wrap;word-break:break-word}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}tr.executed{{background:#2a3d35}}{optional_styles}@media(max-width:850px){{.layout{{grid-template-columns:1fr}}nav{{position:static;display:flex}}nav button{{min-width:120px}}.top{{grid-template-columns:1fr}}}}
</style></head><body><div class="shell"><header><h1>Evaluation rollout</h1><div class="label">Task</div><div class="task">{task}</div><div class="pills"><span class="pill">{html.escape(str(audit['policy_family']))}</span><span class="pill {'success' if audit['success'] else 'failure'}">{'SUCCESS' if audit['success'] else 'FAILURE'}</span><span class="pill">reward {audit['reward']}</span><span class="pill">{audit['turn_count']} steps</span><span class="pill">{html.escape(str(audit['stop_reason']))}</span></div></header><div class="layout"><nav id="nav"></nav><main id="cards"></main></div></div>
<script>
const audit=JSON.parse(atob('{_encoded(audit)}'));const nav=document.getElementById('nav'),cards=document.getElementById('cards');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));const f=x=>x==null?'—':Number(x).toFixed(5);const pct=x=>x==null?'—':(100*Math.exp(Number(x))).toFixed(2)+'%';
function show(i){{document.querySelectorAll('.card,nav button').forEach(e=>e.classList.remove('active'));document.getElementById('card-'+i).classList.add('active');document.getElementById('nav-'+i).classList.add('active')}}
function actions(t){{const names=audit.action_space?.names||[];const logs=t.action_distribution?.log_probabilities||[];const qs=t.direct_q?.values||[];const roots=t.planner?.root_scores||[];const visits=t.planner?.root_visits||[];const executed=t.executed_action?.id??-1;const n=Math.max(names.length,logs.length,qs.length,roots.length,executed+1);let rows='';for(let i=0;i<n;i++){{const name=names[i]||('action_'+i);rows+=`<tr class="${{i===executed?'executed':''}}"><td>${{esc(name)}}${{i===executed?' · EXECUTED':''}}</td><td>${{pct(logs[i])}}</td>{'<td>${f(qs[i])}</td>' if capabilities['direct_q'] else ''}{'<td>${f(roots[i])}</td><td>${visits[i]??"—"}</td>' if capabilities['planner'] else ''}</tr>`}}return `<table><thead><tr><th>Action</th><th>Behavior probability</th>{direct_label}{planner_label}</tr></thead><tbody>${{rows}}</tbody></table>`}}
function candidates(t){{if(!t.planner)return '';return `<details><summary>All planner candidates (${{t.planner.candidates.length}})</summary><table><thead><tr><th>Action sequence</th><th>Score</th><th>Visits</th></tr></thead><tbody>${{t.planner.candidates.map(c=>`<tr><td>${{c.actions.map(esc).join(' → ')}}</td><td>${{f(c.score)}}</td><td>${{c.visits??'—'}}</td></tr>`).join('')}}</tbody></table></details>`}}
function stateEvidence(t){{if(!t.model_state)return '';const p=t.planner?.mcts_process;return `<details><summary>Full current state and MCTS process</summary><p>Latent hidden: ${{t.model_state.arrays.latent_hidden.shape.join('×')}} · current WM state: ${{t.model_state.arrays.current_state.shape.join('×')}} · predicted node states: ${{t.model_state.arrays.mcts_node_states.shape.join('×')}}</p><p>Tree nodes: ${{p?.tree_nodes?.length??'—'}} · chronological simulations: ${{p?.simulations?.length??'—'}}</p><p><a href="${{esc(t.model_state.archive)}}">Download complete float32 state archive</a> · ${{esc(t.model_state.sha256)}}</p><pre>${{esc(JSON.stringify(p,null,2))}}</pre></details>`}}
audit.turns.forEach((t,i)=>{{const actionName=t.executed_action?.name||'no valid action executed';const b=document.createElement('button');b.id='nav-'+i;b.textContent=`Step ${{i+1}} · ${{actionName}}`;b.onclick=()=>show(i);nav.appendChild(b);const stateValue=t.direct_q?.state_value;const c=document.createElement('section');c.id='card-'+i;c.className='card';c.innerHTML=`<h2>Step ${{i+1}}</h2><div class="top"><div><img class="obs" src="${{esc(t.observation.image)}}"><div class="note">True environment observation before the executed action.</div></div><div><div class="metrics"><div class="metric"><span class="label">Executed action</span><strong>${{esc(actionName)}}</strong></div><div class="metric"><span class="label">Environment reward</span><strong>${{f(t.environment.reward)}}</strong></div>{state_value_metric}</div>${{audit.capabilities.cot?`<details open><summary>Actual generated CoT</summary><pre>${{esc(t.cot)}}</pre></details>`:`<details open><summary>Raw response · this policy did not provide CoT</summary><pre>${{esc(t.raw_response)}}</pre></details>`}}</div></div><details open><summary>All available action evidence</summary>${{actions(t)}}</details>${{candidates(t)}}${{stateEvidence(t)}}${{t.terminal?`<details open><summary>Terminal observation and CoT · no action executed</summary><img class="obs" style="max-width:520px;margin-top:10px" src="${{esc(t.terminal.observation.image)}}"><pre>${{esc(t.terminal.cot||'Terminal CoT was not captured by this policy.')}}</pre></details>`:''}}`;cards.appendChild(c)}});show(0);
</script></body></html>'''


def render_evaluation_index(manifest: Mapping[str, Any]) -> str:
    rows = "".join(
        f'<button class="rollout" data-path="{html.escape(str(row["artifact"]))}" '
        f'data-search="{html.escape((str(row.get("task") or "") + " " + str(row.get("data_source") or "") + " " + str(row["identity"])).lower())}" '
        f'data-success="{str(bool(row["success"])).lower()}">'
        f'<strong>{html.escape(str(row.get("task") or "Task unavailable"))}</strong>'
        f'<span>{html.escape(str(row.get("data_source") or "unknown"))} · '
        f' reward {row["reward"]} · {row["turn_count"]} steps · '
        f'{"success" if row["success"] else "failure"}</span></button>'
        for row in manifest["rollouts"]
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evaluation Rollout Browser</title><style>:root{{--bg:#09101e;--panel:#121c31;--text:#e8edf8;--muted:#9fb0cc;--border:#2a395d;--blue:#61a8ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}}header{{padding:14px 18px;border-bottom:1px solid var(--border);background:var(--panel)}}h1{{font-size:21px;margin:0 0 8px}}.summary{{color:var(--muted);font-size:13px}}.layout{{display:grid;grid-template-columns:360px 1fr;height:calc(100vh - 82px)}}aside{{padding:12px;border-right:1px solid var(--border);overflow:auto}}input,select{{width:100%;padding:10px;margin-bottom:8px;background:#0e1629;color:var(--text);border:1px solid var(--border);border-radius:8px}}button.rollout{{display:block;width:100%;text-align:left;padding:10px;margin:5px 0;border:1px solid transparent;border-radius:9px;background:var(--panel);color:var(--text);cursor:pointer}}button.rollout:hover,button.rollout.active{{border-color:var(--blue)}}button.rollout span{{display:block;color:var(--muted);font-size:12px;margin-top:5px}}iframe{{width:100%;height:100%;border:0;background:#0b1020}}@media(max-width:850px){{.layout{{grid-template-columns:1fr;height:auto}}aside{{max-height:320px}}iframe{{height:900px}}}}</style></head><body><header><h1>Evaluation Rollout Browser</h1><div class="summary">{manifest['rollout_count']} rollouts · {manifest['summary']['success_count']} success · status {html.escape(str(manifest['status']).upper())}</div></header><div class="layout"><aside><input id="search" placeholder="Search task, source, or identity"><select id="result"><option value="all">All results</option><option value="true">Success</option><option value="false">Failure</option></select><div id="rollouts">{rows}</div></aside><iframe id="viewer" title="Selected rollout"></iframe></div><script>const buttons=[...document.querySelectorAll('button.rollout')],viewer=document.getElementById('viewer'),search=document.getElementById('search'),result=document.getElementById('result');function select(b){{buttons.forEach(x=>x.classList.remove('active'));b.classList.add('active');viewer.src=b.dataset.path}}function filter(){{const q=search.value.toLowerCase(),r=result.value;buttons.forEach(b=>b.hidden=!(b.dataset.search.includes(q)&&(r==='all'||b.dataset.success===r)));const first=buttons.find(b=>!b.hidden);if(first&&!buttons.some(b=>b.classList.contains('active')&&!b.hidden))select(first)}}buttons.forEach(b=>b.onclick=()=>select(b));search.oninput=filter;result.onchange=filter;if(buttons.length)select(buttons[0]);</script></body></html>'''


__all__ = ["render_evaluation_index", "render_rollout_html"]
