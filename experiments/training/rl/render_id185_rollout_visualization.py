#!/usr/bin/env python3
"""Render a self-contained interactive HTML view of one audited rollout."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def render(audit_dir: Path, output: Path) -> None:
    audit = json.loads((audit_dir / "rollout_audit.json").read_text())
    if audit.get("schema") != "vagen_single_rollout_visualization_audit_v1":
        raise ValueError("unsupported rollout visualization audit")
    if audit.get("turn_count") != len(audit.get("turns", [])):
        raise ValueError("rollout visualization turn count mismatch")
    encoded = base64.b64encode(
        json.dumps(audit, ensure_ascii=False, allow_nan=False).encode()
    ).decode()
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ID185 rollout visualization</title>
<style>
:root{--bg:#0b1020;--panel:#131b31;--panel2:#192541;--text:#e8edf8;--muted:#9fb0cc;--blue:#61a8ff;--green:#55d6a0;--red:#ff758f;--gold:#ffd166;--border:#2a395d}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(140deg,#090d18,#111a30);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}.shell{max-width:1500px;margin:auto;padding:24px}.header{background:rgba(19,27,49,.94);border:1px solid var(--border);border-radius:18px;padding:20px 24px;box-shadow:0 14px 40px #0006}.header h1{margin:0 0 10px;font-size:25px}.meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}.pill{border:1px solid var(--border);background:#0e1629;border-radius:999px;padding:6px 10px}.failure{color:var(--red)}.success{color:var(--green)}.layout{display:grid;grid-template-columns:220px 1fr;gap:18px;margin-top:18px}.nav{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:10px}.nav button{display:block;width:100%;text-align:left;border:0;border-radius:10px;padding:10px;margin:3px 0;color:var(--muted);background:transparent;cursor:pointer}.nav button:hover,.nav button.active{background:var(--panel2);color:white}.card{display:none;background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:18px;margin-bottom:18px}.card.active{display:block}.top{display:grid;grid-template-columns:minmax(280px,520px) 1fr;gap:18px}.obs{width:100%;border-radius:13px;border:1px solid var(--border);image-rendering:auto}.metrics{display:grid;grid-template-columns:repeat(2,minmax(130px,1fr));gap:10px}.metric{background:var(--panel2);padding:12px;border-radius:12px;border:1px solid var(--border)}.metric .label{font-size:12px;color:var(--muted)}.metric .value{font-size:19px;margin-top:4px;font-variant-numeric:tabular-nums}.action{color:var(--gold)}h2,h3{margin:0 0 12px}details{background:#0f172a;border:1px solid var(--border);border-radius:12px;margin-top:14px;padding:12px}summary{cursor:pointer;color:var(--blue);font-weight:650}pre{white-space:pre-wrap;word-break:break-word;line-height:1.48;color:#dbe5f8;font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px}table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid var(--border);text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}tr.executed{background:#2a3d35}tr.prior td:first-child:after{content:'  · LLM prior';color:var(--blue);font-size:11px}.bar{display:inline-block;height:7px;border-radius:5px;background:var(--blue);min-width:1px;margin-left:6px}.seq{font-family:ui-monospace,monospace;text-align:left!important}.terminal{border-color:#7a5060}.note{color:var(--muted);font-size:12px;margin-top:8px}@media(max-width:900px){.layout{grid-template-columns:1fr}.nav{position:static;display:flex;overflow:auto}.nav button{min-width:100px}.top{grid-template-columns:1fr}}
</style></head><body><div class="shell"><header class="header"><h1>ID185 K4 Scheme-B rollout</h1><div class="meta" id="meta"></div></header><div class="layout"><nav class="nav" id="nav"></nav><main id="cards"></main></div></div>
<script>
const audit=JSON.parse(atob('__AUDIT_B64__'));const nav=document.getElementById('nav'),cards=document.getElementById('cards');
const f=x=>Number(x).toFixed(5),pct=x=>(100*Number(x)).toFixed(2)+'%';
document.getElementById('meta').innerHTML=`<span class="pill">sample ${audit.rollout_sample_id.slice(0,18)}…</span><span class="pill">${audit.turn_count} action steps</span><span class="pill ${audit.success?'success':'failure'}">${audit.success?'SUCCESS':'FAILURE'}</span><span class="pill">source step 796</span>`;
function show(i){document.querySelectorAll('.card,.nav button').forEach(x=>x.classList.remove('active'));document.getElementById('card-'+i).classList.add('active');document.getElementById('nav-'+i).classList.add('active')}
function actionTable(t){return `<table><thead><tr><th>Action</th><th>Prior</th><th>Guided</th><th>Direct Q</th><th>MCTS value</th><th>Visits</th></tr></thead><tbody>${t.action_ranking.map(a=>`<tr class="${a.is_executed_action?'executed ':''}${a.is_prior_action?'prior':''}"><td>${a.action}${a.is_executed_action?' · EXECUTED':''}</td><td>${pct(a.prior_probability)}</td><td>${pct(a.guided_probability)}<span class="bar" style="width:${Math.max(1,90*a.guided_probability)}px"></span></td><td>${f(a.direct_q)}</td><td>${f(a.predicted_root_value)}</td><td>${a.root_visits}</td></tr>`).join('')}</tbody></table>`}
function seqTable(t){return `<table><thead><tr><th>Predicted 4-action sequence</th><th>Value</th><th>Visits</th></tr></thead><tbody>${t.predicted_action_sequences.map(s=>`<tr><td class="seq">${s.actions.join(' → ')}</td><td>${f(s.predicted_value)}</td><td>${s.visits}</td></tr>`).join('')}</tbody></table>`}
audit.turns.forEach((t,i)=>{const b=document.createElement('button');b.id='nav-'+i;b.textContent=`Step ${i+1} · ${t.executed_action}`;b.onclick=()=>show(i);nav.appendChild(b);const c=document.createElement('section');c.className='card';c.id='card-'+i;c.innerHTML=`<h2>Step ${i+1}</h2><div class="top"><div><img class="obs" src="${t.observation_image}" alt="step ${i+1} observation"><div class="note">True environment observation before executing the action.</div></div><div><div class="metrics"><div class="metric"><div class="label">Executed action</div><div class="value action">${t.executed_action}</div></div><div class="metric"><div class="label">LLM prior action</div><div class="value">${t.prior_action}</div></div><div class="metric"><div class="label">Current state value Eπ[Qdirect]</div><div class="value">${f(t.current_state_value)}</div></div><div class="metric"><div class="label">Executed direct Q</div><div class="value">${f(t.executed_action_direct_q)}</div></div><div class="metric"><div class="label">Executed predicted MCTS value</div><div class="value">${f(t.executed_action_predicted_value)}</div></div><div class="metric"><div class="label">Reward / planner latency</div><div class="value">${f(t.env_turn_reward)} / ${f(t.planner_latency_seconds)}s</div></div></div><details open><summary>Actual generated CoT</summary><pre></pre></details></div></div><details open><summary>All actions: prior, guided policy, current Q, predicted value</summary>${actionTable(t)}</details><details><summary>Predicted action lists from K4 MCTS (${t.predicted_action_sequences.length})</summary>${seqTable(t)}</details>${t.terminal?`<details class="terminal" open><summary>Terminal observation and actual terminal CoT</summary><img class="obs" style="max-width:520px;margin-top:12px" src="${t.terminal.observation_image}"><pre class="terminal-cot"></pre></details>`:''}`;c.querySelector('pre').textContent=t.cot;if(t.terminal)c.querySelector('.terminal-cot').textContent=t.terminal.cot;cards.appendChild(c)});show(0);
</script></body></html>'''.replace('__AUDIT_B64__', encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.audit_dir, args.output)
    print(f"ID185_ROLLOUT_VISUALIZATION_HTML_OK path={args.output}")


if __name__ == "__main__":
    main()
