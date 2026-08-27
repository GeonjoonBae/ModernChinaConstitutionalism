#!/usr/bin/env python
"""Build the paper-specific lightweight pair-conditioned keyness dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "keyness" / "paper_robust_candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "dashboards" / "pair_conditioned_keyness_dashboard_lite.html"

DISPLAY_FIELDS = [
    "robust_score",
    "token",
    "included_metrics",
    "log_likelihood",
    "log_odds_z",
    "log_ratio",
    "chi_square",
    "tfidf",
    "count_period",
    "count_ref",
    "edge_a",
    "edge_b",
    "shared_strength",
    "dominant_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        numeric_fields = {
            "period_sort_order",
            "robust_score",
            "log_likelihood",
            "log_odds_z",
            "log_ratio",
            "chi_square",
            "tfidf",
            "count_period",
            "count_ref",
            "edge_a",
            "edge_b",
            "shared_strength",
        }
        keep = [
            "period_id",
            "period_label",
            "period_sort_order",
            "period_start_date",
            "period_end_date",
            "pair_id",
            "focus_a_label",
            "focus_b_label",
            *DISPLAY_FIELDS,
        ]
        for source in reader:
            row: dict[str, object] = {}
            for field in keep:
                value = source.get(field, "")
                if field in numeric_fields:
                    try:
                        row[field] = float(value)
                    except (TypeError, ValueError):
                        row[field] = None
                else:
                    row[field] = value
            label = str(row.get("period_label", ""))
            start = str(row.get("period_start_date", ""))
            end = str(row.get("period_end_date", ""))
            if label.startswith("long_period_manual_") and start and end:
                row["period_label"] = f"{start.replace('-', '.')} - {end.replace('-', '.')}"
            rows.append(row)
    return rows


def build_html(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pair-Conditioned Keyness Dashboard - Paper Lite</title>
<style>
:root {{ color-scheme: light; --ink:#10233f; --muted:#5e6f86; --line:#d5dde8; --soft:#f5f8fb; --accent:#1f6f8b; --strong:#a52a2a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#fff; color:var(--ink); font-family:Inter,"Noto Sans KR","Noto Sans CJK KR",Arial,sans-serif; font-size:14px; }}
.page {{ max-width:1320px; margin:0 auto; padding:24px 20px 40px; }}
h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:0; }}
.subtitle {{ margin:0 0 16px; color:var(--muted); }}
details {{ border:1px solid var(--line); border-radius:6px; background:var(--soft); margin-bottom:12px; }}
summary {{ cursor:pointer; padding:11px 14px; font-weight:700; }}
.controls {{ display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:10px; padding:0 14px 14px; }}
label {{ display:grid; gap:5px; color:var(--muted); font-size:12px; }}
select,input,button {{ min-height:36px; border:1px solid #c7d2df; border-radius:5px; background:#fff; color:var(--ink); padding:7px 9px; font:inherit; }}
button {{ cursor:pointer; font-weight:600; }}
.badges {{ display:flex; flex-wrap:wrap; gap:7px; margin:10px 0 14px; }}
.badge {{ border:1px solid var(--line); border-radius:5px; padding:7px 9px; background:#fff; color:#455a73; }}
.panel {{ border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.panel-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; background:var(--soft); border-bottom:1px solid var(--line); }}
.panel-head h2 {{ margin:0; font-size:17px; }}
.table-wrap {{ overflow:auto; max-height:72vh; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
thead th {{ position:sticky; top:0; z-index:2; background:#eaf0f6; border-bottom:1px solid var(--line); padding:9px 8px; text-align:right; cursor:pointer; }}
thead th:nth-child(2), thead th:nth-child(3), tbody td:nth-child(2), tbody td:nth-child(3) {{ text-align:left; }}
tbody td {{ border-bottom:1px solid #e3e8ef; padding:8px; text-align:right; }}
tbody tr:hover {{ background:#f7fafc; }}
.score5 {{ color:var(--strong); font-weight:700; }}
.token {{ font-weight:700; font-size:15px; }}
.metrics {{ color:#4e647d; }}
.empty {{ padding:30px; text-align:center; color:var(--muted); }}
.note {{ margin:10px 2px 0; color:var(--muted); font-size:12px; line-height:1.55; }}
@media (max-width:900px) {{ .controls {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .page {{ padding:16px 10px 28px; }} }}
</style>
</head>
<body>
<main class="page">
  <h1>Pair-Conditioned Keyness Dashboard</h1>
  <p class="subtitle">Paper-specific lightweight edition: full · w10 · top100 · min · same_period_other_pairs</p>
  <details open>
    <summary>Conditions and display</summary>
    <div class="controls">
      <label>Period<select id="period"></select></label>
      <label>Pair<select id="pair"></select></label>
      <label>Minimum robust score<select id="score"><option value="4">4</option><option value="5">5</option></select></label>
      <label>Rows<select id="limit"><option value="20">Top 20</option><option value="0">All candidates</option></select></label>
      <label>Token / POS filter<input id="search" type="search" placeholder="token or POS"></label>
    </div>
  </details>
  <div class="badges" id="badges"></div>
  <section class="panel">
    <div class="panel-head"><h2>Robust Candidates</h2><button id="copy">Copy table</button></div>
    <div class="table-wrap"><table id="table"><thead></thead><tbody></tbody></table></div>
  </section>
  <p class="note">Candidates appear in at least four of five positive-direction metric top-30 lists. Default order is log-likelihood descending, with absolute log odds z as the tie-breaker. Click a column title to sort.</p>
</main>
<script>
const DATA={payload};
const columns=[
 ['rank','rank'],['token','token'],['robust_score','score'],['included_metrics','metrics'],
 ['log_likelihood','log-likelihood'],['log_odds_z','z'],['log_ratio','log ratio'],
 ['chi_square','chi-square'],['tfidf','TF-IDF'],['count_period','count'],['count_ref','ref'],
 ['edge_a','edge A'],['edge_b','edge B'],['shared_strength','shared'],['dominant_pos','POS']
];
const $=id=>document.getElementById(id);
let sortKey='log_likelihood', sortAsc=false;
const periods=[...new Map(DATA.slice().sort((a,b)=>a.period_sort_order-b.period_sort_order).map(r=>[r.period_id,r.period_label])).entries()];
for(const [id,label] of periods) $('period').add(new Option(label,id));
function refreshPairs() {{
 const current=$('pair').value;
 const pairs=[...new Set(DATA.filter(r=>r.period_id===$('period').value).map(r=>r.pair_id))];
 $('pair').replaceChildren(...pairs.map(v=>new Option(v,v)));
 if(pairs.includes(current)) $('pair').value=current;
}}
function number(value,digits=3) {{
 if(value===null || value==='' || Number.isNaN(Number(value))) return '';
 return Number(value).toLocaleString(undefined,{{maximumFractionDigits:digits}});
}}
function filtered() {{
 const q=$('search').value.trim().toLowerCase();
 const min=Number($('score').value);
 return DATA.filter(r=>r.period_id===$('period').value && r.pair_id===$('pair').value && Number(r.robust_score)>=min && (!q || String(r.token).toLowerCase().includes(q) || String(r.dominant_pos).toLowerCase().includes(q)));
}}
function compare(a,b) {{
 const av=a[sortKey], bv=b[sortKey];
 let c;
 if(typeof av==='number' && typeof bv==='number') c=av-bv;
 else c=String(av??'').localeCompare(String(bv??''),'ko');
 return sortAsc?c:-c;
}}
function render() {{
 let rows=filtered();
 if(sortKey==='log_likelihood') rows.sort((a,b)=>{{const d=Number(b.log_likelihood)-Number(a.log_likelihood);return sortAsc?-d:(d || Math.abs(Number(b.log_odds_z))-Math.abs(Number(a.log_odds_z)));}});
 else rows.sort(compare);
 const total=rows.length, limit=Number($('limit').value);
 if(limit) rows=rows.slice(0,limit);
 $('table').querySelector('thead').innerHTML='<tr>'+columns.map(([k,l])=>`<th data-key="${{k}}">${{l}}${{sortKey===k?(sortAsc?' ↑':' ↓'):''}}</th>`).join('')+'</tr>';
 const body=$('table').querySelector('tbody');
 if(!rows.length) body.innerHTML='<tr><td class="empty" colspan="15">No candidates match the selected condition.</td></tr>';
 else body.innerHTML=rows.map((r,i)=>`<tr><td>${{i+1}}</td><td class="token">${{r.token}}</td><td class="${{Number(r.robust_score)===5?'score5':''}}">${{number(r.robust_score,0)}}</td><td class="metrics">${{String(r.included_metrics).replaceAll(';',', ')}}</td><td>${{number(r.log_likelihood)}}</td><td>${{number(r.log_odds_z)}}</td><td>${{number(r.log_ratio)}}</td><td>${{number(r.chi_square)}}</td><td>${{number(r.tfidf,6)}}</td><td>${{number(r.count_period,0)}}</td><td>${{number(r.count_ref,0)}}</td><td>${{number(r.edge_a,0)}}</td><td>${{number(r.edge_b,0)}}</td><td>${{number(r.shared_strength,6)}}</td><td>${{r.dominant_pos??''}}</td></tr>`).join('');
 $('badges').innerHTML=[`condition: full | w10 | top100 | min`,`period: ${{$('period').selectedOptions[0]?.text||''}}`,`pair: ${{$('pair').value}}`,`matching candidates: ${{total}}`,`displayed: ${{rows.length}}`].map(v=>`<span class="badge">${{v}}</span>`).join('');
 document.querySelectorAll('th[data-key]').forEach(th=>th.onclick=()=>{{const key=th.dataset.key;if(key==='rank')return;if(sortKey===key)sortAsc=!sortAsc;else{{sortKey=key;sortAsc=false;}}render();}});
}}
$('period').onchange=()=>{{refreshPairs();render();}};
for(const id of ['pair','score','limit','search']) $(id).addEventListener(id==='search'?'input':'change',render);
$('copy').onclick=async()=>{{
 const lines=[columns.map(x=>x[1]).join('\\t')];
 document.querySelectorAll('#table tbody tr').forEach(tr=>lines.push([...tr.cells].map(td=>td.innerText).join('\\t')));
 await navigator.clipboard.writeText(lines.join('\\n')); $('copy').textContent='Copied'; setTimeout(()=>$('copy').textContent='Copy table',1200);
}};
refreshPairs();render();
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv.resolve())
    output = args.output_html.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(rows), encoding="utf-8")
    print(f"Wrote {output} ({len(rows):,} candidates)")


if __name__ == "__main__":
    main()
