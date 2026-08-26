#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an HTML dashboard for pair-conditioned keyness outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
SHENBAO = ROOT / "shenbao"
DEFAULT_INPUT_DIR = SHENBAO / "shenbao_interpretation" / "pair_conditioned_keyness"
DEFAULT_KEYNESS_CSV = DEFAULT_INPUT_DIR / "pair_conditioned_keyness.csv"
DEFAULT_ROBUST_CSV = DEFAULT_INPUT_DIR / "pair_conditioned_keyness_robust_candidates.csv"
DEFAULT_CONTRAST_KEYNESS_CSV = DEFAULT_INPUT_DIR / "core_contrast_keyness.csv"
DEFAULT_CONTRAST_ROBUST_CSV = DEFAULT_INPUT_DIR / "core_contrast_keyness_robust_candidates.csv"
DEFAULT_OUTPUT_HTML = DEFAULT_INPUT_DIR / "pair_conditioned_keyness_dashboard.html"

PROFILE_ORDER = ["regex-only", "strict", "full"]
NEIGHBOR_SCOPE_ORDER = ["all", "top20", "top50", "top100"]
COMPARISON_ORDER = ["same_pair_other_periods", "same_period_other_pairs"]
METRICS = ["log_odds_z", "log_likelihood", "log_ratio", "chi_square", "tfidf"]
PAIR_ORDER = [
    "\u7acb\u61b2-\u61b2\u653f",
    "\u7acb\u61b2-\u61b2\u6cd5",
    "\u7acb\u61b2-\u5236\u61b2",
    "\u61b2\u653f-\u61b2\u6cd5",
    "\u61b2\u653f-\u5236\u61b2",
    "\u61b2\u6cd5-\u5236\u61b2",
]

KEYNESS_COLUMNS = [
    "analysis_mode",
    "token",
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "pair_id",
    "focus_a_label",
    "focus_b_label",
    "candidate_scope",
    "neighbor_scope",
    "topn",
    "pair_count_mode",
    "comparison_type",
    "comparison_id",
    "count_period",
    "count_ref",
    "total_period",
    "total_ref",
    "rate_period",
    "rate_ref",
    "log_odds_z",
    "log_likelihood",
    "log_ratio",
    "chi_square",
    "tfidf",
    "tfidf_a",
    "tfidf_b",
    "direction",
    "rank_positive",
    "rank_negative",
    "rank_log_likelihood",
    "rank_log_ratio",
    "rank_chi_square",
    "rank_tfidf",
    "edge_a",
    "edge_b",
    "norm_a",
    "norm_b",
    "shared_strength",
    "rank_a",
    "rank_b",
    "period_keyness_log_odds_z",
    "period_keyness_rank_positive",
    "period_keyness_rank_negative",
    "period_keyness_count",
    "period_keyness_rate",
    "dominant_pos",
    "dict_lv1",
    "dict_lv2",
    "count_a",
    "count_b",
    "total_a",
    "total_b",
    "rate_a",
    "rate_b",
    "signed_log_likelihood",
    "signed_chi_square",
    "rank_log_likelihood_a",
    "rank_log_likelihood_b",
    "rank_log_ratio_a",
    "rank_log_ratio_b",
    "rank_chi_square_a",
    "rank_chi_square_b",
    "rank_tfidf_a",
    "rank_tfidf_b",
]

ROBUST_COLUMNS = [
    "analysis_mode",
    "token",
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "pair_id",
    "focus_a_label",
    "focus_b_label",
    "candidate_scope",
    "neighbor_scope",
    "topn",
    "pair_count_mode",
    "comparison_type",
    "comparison_id",
    "robust_topn",
    "robust_min_metrics",
    "robust_score",
    "robust_class",
    "included_metrics",
    "log_odds_z",
    "log_likelihood",
    "log_ratio",
    "chi_square",
    "tfidf",
    "direction",
    "rank_positive",
    "rank_log_likelihood",
    "rank_log_ratio",
    "rank_chi_square",
    "rank_tfidf",
    "count_period",
    "count_ref",
    "rate_period",
    "rate_ref",
    "edge_a",
    "edge_b",
    "shared_strength",
    "period_keyness_log_odds_z",
    "period_keyness_rank_positive",
    "dominant_pos",
    "dict_lv1",
    "dict_lv2",
    "count_a",
    "count_b",
    "rate_a",
    "rate_b",
    "rank_a",
    "rank_b",
]


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value


def period_display(start: object, end: object, period_id: object) -> str:
    period_text = str(period_id)
    if period_text == "global" or period_text.startswith("global_"):
        return "global"
    if pd.isna(start) or pd.isna(end) or not str(start).strip() or not str(end).strip():
        return period_text
    return f"{str(start)[:10].replace('-', '.') } - {str(end)[:10].replace('-', '.')}"


def ordered_unique(values: Sequence[object], preferred: Sequence[str] | None = None) -> List[str]:
    seen = {str(value) for value in values if not pd.isna(value) and str(value) != ""}
    out: List[str] = []
    if preferred:
        for value in preferred:
            if value in seen:
                out.append(value)
                seen.remove(value)
    out.extend(sorted(seen))
    return out


def prepare_keyness_rows(df: pd.DataFrame, top_per_metric: int) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [col for col in KEYNESS_COLUMNS if col in df.columns]
    df = df[keep].copy()
    group_cols = [
        "analysis_mode",
        "token_profile",
        "network_window",
        "period_id",
        "pair_id",
        "candidate_scope",
        "neighbor_scope",
        "pair_count_mode",
        "comparison_type",
    ]
    selected_parts: List[pd.DataFrame] = []
    for _key, group in df.groupby(group_cols, dropna=False):
        selected_idx = set()
        if str(group["analysis_mode"].iloc[0]) == "core_contrast":
            focus_a = str(group["focus_a_label"].iloc[0])
            for direction, is_a in [(focus_a, True), (str(group["focus_b_label"].iloc[0]), False)]:
                side = group[group["direction"].astype(str) == direction]
                for metric in METRICS:
                    ascending = (not is_a) if metric in {"log_odds_z", "log_ratio", "tfidf"} else False
                    selected_idx.update(side.sort_values(metric, ascending=ascending, kind="mergesort").head(top_per_metric).index)
        else:
            positive = group[group["direction"] == "positive"]
            negative = group[group["direction"] == "negative"]
            if not positive.empty:
                for metric in METRICS:
                    selected_idx.update(positive.sort_values(metric, ascending=False, kind="mergesort").head(top_per_metric).index)
            if not negative.empty:
                selected_idx.update(negative.sort_values("log_odds_z", ascending=True, kind="mergesort").head(top_per_metric).index)
                selected_idx.update(negative.sort_values("log_ratio", ascending=True, kind="mergesort").head(top_per_metric).index)
        selected_parts.append(group.loc[sorted(selected_idx)])
    out = pd.concat(selected_parts, ignore_index=True) if selected_parts else df.iloc[0:0].copy()
    out["period_display"] = out.apply(
        lambda row: period_display(row.get("period_start_date"), row.get("period_end_date"), row.get("period_id")), axis=1
    )
    return out


def prepare_robust_rows(df: pd.DataFrame, max_rows_per_group: int) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [col for col in ROBUST_COLUMNS if col in df.columns]
    df = df[keep].copy()
    group_cols = [
        "analysis_mode",
        "token_profile",
        "network_window",
        "period_id",
        "pair_id",
        "candidate_scope",
        "neighbor_scope",
        "pair_count_mode",
        "comparison_type",
    ]
    parts = []
    for _key, group in df.groupby(group_cols, dropna=False):
        parts.append(
            group.sort_values(["robust_score", "log_odds_z"], ascending=[False, False], kind="mergesort").head(
                max_rows_per_group
            )
        )
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    out["period_display"] = out.apply(
        lambda row: period_display(row.get("period_start_date"), row.get("period_end_date"), row.get("period_id")), axis=1
    )
    return out


def build_group_summary(keyness_df: pd.DataFrame, robust_df: pd.DataFrame) -> List[Dict[str, object]]:
    if keyness_df.empty:
        return []
    group_cols = [
        "analysis_mode",
        "token_profile",
        "network_window",
        "period_id",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
        "pair_id",
        "candidate_scope",
        "neighbor_scope",
        "pair_count_mode",
        "comparison_type",
    ]
    summary = keyness_df.groupby(group_cols, dropna=False).agg(
        keyness_rows=("token", "size"),
        positive_rows=("direction", lambda values: int((values == "positive").sum())),
        negative_rows=("direction", lambda values: int((values == "negative").sum())),
        max_log_odds_z=("log_odds_z", "max"),
        max_log_likelihood=("log_likelihood", "max"),
        max_log_ratio=("log_ratio", "max"),
        max_chi_square=("chi_square", "max"),
        max_tfidf=("tfidf", "max"),
    ).reset_index()
    if not robust_df.empty:
        robust_counts = robust_df.groupby(group_cols, dropna=False).agg(
            robust_rows=("token", "size"),
            strong_robust_rows=("robust_class", lambda values: int((values == "strong").sum())),
        ).reset_index()
        summary = summary.merge(robust_counts, on=group_cols, how="left")
    else:
        summary["robust_rows"] = 0
        summary["strong_robust_rows"] = 0
    summary[["robust_rows", "strong_robust_rows"]] = summary[["robust_rows", "strong_robust_rows"]].fillna(0).astype(int)
    summary["period_display"] = summary.apply(
        lambda row: period_display(row.get("period_start_date"), row.get("period_end_date"), row.get("period_id")), axis=1
    )
    return summary.to_dict(orient="records")


def build_payload(args: argparse.Namespace) -> Dict[str, object]:
    keyness_path = Path(args.keyness_csv).expanduser().resolve()
    robust_path = Path(args.robust_csv).expanduser().resolve()
    keyness_df = pd.read_csv(keyness_path, encoding="utf-8-sig", low_memory=False)
    robust_df = pd.read_csv(robust_path, encoding="utf-8-sig", low_memory=False) if robust_path.is_file() else pd.DataFrame()
    keyness_df["analysis_mode"] = "shared_pair"
    if not robust_df.empty:
        robust_df["analysis_mode"] = "shared_pair"
    contrast_keyness_path = Path(args.contrast_keyness_csv).expanduser().resolve()
    contrast_robust_path = Path(args.contrast_robust_csv).expanduser().resolve()
    if contrast_keyness_path.is_file():
        contrast_keyness = pd.read_csv(contrast_keyness_path, encoding="utf-8-sig", low_memory=False)
        contrast_keyness["analysis_mode"] = "core_contrast"
        contrast_keyness["pair_count_mode"] = "not_applicable"
        keyness_df = pd.concat([keyness_df, contrast_keyness], ignore_index=True, sort=False)
    if contrast_robust_path.is_file():
        contrast_robust = pd.read_csv(contrast_robust_path, encoding="utf-8-sig", low_memory=False)
        contrast_robust["analysis_mode"] = "core_contrast"
        contrast_robust["pair_count_mode"] = "not_applicable"
        robust_df = pd.concat([robust_df, contrast_robust], ignore_index=True, sort=False)
    if "pair_count_mode" not in keyness_df.columns:
        keyness_df["pair_count_mode"] = "sum"
    if not robust_df.empty and "pair_count_mode" not in robust_df.columns:
        robust_df["pair_count_mode"] = "sum"

    slim_keyness = prepare_keyness_rows(keyness_df, args.top_per_metric)
    slim_robust = prepare_robust_rows(robust_df, args.max_robust_rows_per_group)
    summary = build_group_summary(keyness_df, robust_df)

    periods = (
        keyness_df[["period_id", "period_sort_order", "period_start_date", "period_end_date"]]
        .drop_duplicates()
        .sort_values(["period_sort_order", "period_id"])
    )
    periods["period_display"] = periods.apply(
        lambda row: period_display(row["period_start_date"], row["period_end_date"], row["period_id"]), axis=1
    )
    period_order = {str(row.period_id): idx for idx, row in enumerate(periods.itertuples(index=False))}
    periods_by_comparison = {}
    for comparison, group in keyness_df[["comparison_type", "period_id"]].drop_duplicates().groupby("comparison_type"):
        period_ids = [str(value) for value in group["period_id"].tolist()]
        periods_by_comparison[str(comparison)] = sorted(
            period_ids,
            key=lambda value: (period_order.get(value, 10**9), value),
        )

    keyness_cols = list(slim_keyness.columns)
    robust_cols = list(slim_robust.columns)
    slim_keyness_values = slim_keyness.where(pd.notna(slim_keyness), None).values.tolist()
    slim_robust_values = slim_robust.where(pd.notna(slim_robust), None).values.tolist()

    payload = {
        "meta": {
            "title": "Core Keyword Keyness Dashboard",
            "primary_metric": "log_odds_z",
            "metrics": METRICS,
            "top_per_metric": args.top_per_metric,
            "max_robust_rows_per_group": args.max_robust_rows_per_group,
            "source_keyness_csv": str(keyness_path),
            "source_robust_csv": str(robust_path),
            "source_contrast_keyness_csv": str(contrast_keyness_path),
            "source_contrast_robust_csv": str(contrast_robust_path),
            "full_keyness_rows": int(len(keyness_df)),
            "shown_keyness_rows": int(len(slim_keyness)),
            "full_robust_rows": int(len(robust_df)),
            "shown_robust_rows": int(len(slim_robust)),
        },
        "options": {
            "analysis_modes": ordered_unique(keyness_df["analysis_mode"].unique(), ["shared_pair", "core_contrast"]),
            "profiles": ordered_unique(keyness_df["token_profile"].unique(), PROFILE_ORDER),
            "windows": sorted(int(value) for value in keyness_df["network_window"].dropna().unique()),
            "periods": periods.to_dict(orient="records"),
            "pairs": ordered_unique(keyness_df["pair_id"].unique(), PAIR_ORDER),
            "candidate_scopes": ordered_unique(keyness_df["candidate_scope"].unique()),
            "neighbor_scopes": ordered_unique(keyness_df["neighbor_scope"].unique(), NEIGHBOR_SCOPE_ORDER),
            "pair_count_modes": ordered_unique(keyness_df["pair_count_mode"].unique(), ["sum", "min"]),
            "comparison_types": ordered_unique(keyness_df["comparison_type"].unique(), COMPARISON_ORDER),
            "periods_by_comparison": periods_by_comparison,
            "comparison_types_by_mode": {
                str(mode): ordered_unique(group["comparison_type"].unique(), COMPARISON_ORDER + ["core_a_vs_core_b"])
                for mode, group in keyness_df.groupby("analysis_mode", dropna=False)
            },
            "pair_count_modes_by_mode": {
                str(mode): ordered_unique(group["pair_count_mode"].unique(), ["sum", "min", "not_applicable"])
                for mode, group in keyness_df.groupby("analysis_mode", dropna=False)
            },
        },
        "summary_rows": json_ready(summary),
        "keyness_columns": keyness_cols,
        "keyness_rows": json_ready(slim_keyness_values),
        "robust_columns": robust_cols,
        "robust_rows": json_ready(slim_robust_values),
    }
    return payload


def render_html(payload: Dict[str, object]) -> str:
    payload_json = json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Core Keyword Keyness Dashboard</title>
  <style>
    :root {{ color-scheme: light; --line:#d8dee8; --muted:#64748b; --ink:#111827; --panel:#fff; --bg:#f8fafc; --head:#eef2f7; --blue:#2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line); padding:14px 18px 12px; }}
    h1 {{ margin:0 0 10px; font-size:21px; letter-spacing:0; }}
    h2 {{ margin:0 0 9px; font-size:17px; }}
    .controls {{ display:grid; grid-template-columns: repeat(8, minmax(120px, 1fr)); gap:9px; align-items:end; }}
    label {{ display:grid; gap:4px; font-size:12px; color:#475569; }}
    select, input {{ width:100%; min-width:0; padding:7px 8px; border:1px solid #cbd5e1; border-radius:6px; background:#fff; font:inherit; }}
    main {{ padding:16px 18px 30px; }}
    .summary {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:13px; }}
    .pill {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 9px; font-size:13px; }}
    .grid {{ display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap:14px; margin-bottom:14px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:14px; overflow:auto; }}
    .note {{ color:var(--muted); font-size:12px; margin-top:7px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; }}
    th, td {{ border-bottom:1px solid #e5e7eb; padding:6px 7px; text-align:left; vertical-align:top; font-size:12px; }}
    th {{ background:var(--head); position:sticky; top:0; z-index:2; white-space:nowrap; }}
    th.sortable {{ cursor:pointer; user-select:none; }}
    th.sortable::after {{ content:" ↕"; color:#94a3b8; font-size:10px; }}
    th.sortable.sort-asc::after {{ content:" ↑"; color:#2563eb; }}
    th.sortable.sort-desc::after {{ content:" ↓"; color:#2563eb; }}
    .num {{ text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }}
    .token {{ font-size:13px; font-weight:600; }}
    .muted {{ color:var(--muted); }}
    .badge {{ display:inline-block; border:1px solid #cbd5e1; border-radius:999px; padding:1px 6px; margin:1px 2px 1px 0; background:#f8fafc; white-space:nowrap; }}
    .strong {{ color:#991b1b; font-weight:700; }}
    .stable {{ color:#1d4ed8; font-weight:700; }}
    .matrix td, .matrix th {{ text-align:center; }}
    .heat {{ background: color-mix(in srgb, #ef4444 calc(var(--v) * 1%), white); }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:0 0 8px; }}
    button {{ border:1px solid #cbd5e1; border-radius:6px; background:#fff; padding:6px 9px; cursor:pointer; font:inherit; }}
    button:hover {{ border-color:#94a3b8; }}
    .codex-controls-toggle {{
      display:inline-flex; align-items:center; gap:6px; margin:8px 0 10px; padding:6px 10px;
      border:1px solid #cbd5e1; border-radius:6px; background:#fff; color:#334155; font-size:12px; line-height:1.2;
    }}
    .codex-controls-toggle[aria-expanded="false"] {{ margin-bottom:0; }}
    .codex-controls-hidden {{ display:none !important; }}
    @media (max-width: 1300px) {{ .controls {{ grid-template-columns: repeat(4, minmax(120px, 1fr)); }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>Core Keyword Keyness Dashboard</h1>
  <div class="controls">
    <label>Analysis mode<select id="analysisMode"></select></label>
    <label>Profile<select id="profile"></select></label>
    <label>Window<select id="window"></select></label>
    <label>Period<select id="period"></select></label>
    <label>Pair<select id="pair"></select></label>
    <label>Neighbor scope<select id="neighborScope"></select></label>
    <label>Pair count<select id="pairCountMode"></select></label>
    <label>Comparison<select id="comparison"></select></label>
    <label>Metric<select id="metric"></select></label>
    <label>Direction<select id="direction"></select></label>
    <label>Rows<select id="limit"><option>25</option><option selected>50</option><option>100</option><option>200</option></select></label>
    <label>Filter<input id="filterText" placeholder="token, pos, annotation"></label>
  </div>
</header>
<main>
  <div id="summary" class="summary"></div>
  <section class="grid">
    <div class="panel">
      <h2>Robust Candidates</h2>
      <div id="robust"></div>
      <div class="note">Tokens included in at least the configured number of metric top lists. Strong means four or more metrics in the current output.</div>
    </div>
    <div class="panel">
      <h2>Metric Robustness</h2>
      <div id="robustness"></div>
      <div class="note">Overlap among top tokens under each metric for the selected condition.</div>
    </div>
  </section>
  <section class="panel">
    <div class="toolbar">
      <h2 style="margin-right:auto">Keyness Table</h2>
      <button id="copyKeyness">Copy table</button>
    </div>
    <div id="keyness"></div>
  </section>
  <section class="panel">
    <h2>Group Summary</h2>
    <div id="groupSummary"></div>
  </section>
</main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
function inflateRows(columns, rows) {{
  return (rows || []).map(values => {{
    const row = {{}};
    columns.forEach((col, idx) => row[col] = values[idx]);
    return row;
  }});
}}
DATA.keyness_rows = inflateRows(DATA.keyness_columns || [], DATA.keyness_rows || []);
DATA.robust_rows = inflateRows(DATA.robust_columns || [], DATA.robust_rows || []);
const METRICS = DATA.meta.metrics;
const METRIC_RANK = {{
  log_odds_z: "rank_positive",
  log_likelihood: "rank_log_likelihood",
  log_ratio: "rank_log_ratio",
  chi_square: "rank_chi_square",
  tfidf: "rank_tfidf"
}};
function byId(id) {{ return document.getElementById(id); }}
function esc(text) {{
  return String(text ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
}}
function fmt(value, digits=3) {{
  const n = Number(value);
  if (!Number.isFinite(n)) return value === null || value === undefined ? "" : String(value);
  return n.toLocaleString(undefined, {{maximumFractionDigits: digits}});
}}
function pct(value) {{
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  return (100 * n).toFixed(3) + "%";
}}
function attr(text) {{
  return esc(text).replace(/`/g, "&#96;");
}}
function setSelect(id, values, selected) {{
  const el = byId(id);
  el.innerHTML = values.map(v => `<option value="${{esc(v)}}">${{esc(v)}}</option>`).join("");
  if (selected !== undefined && values.map(String).includes(String(selected))) el.value = String(selected);
}}
function periodIdsForComparison(comparison) {{
  const byComparison = DATA.options.periods_by_comparison || {{}};
  const ids = byComparison[String(comparison)];
  return ids && ids.length ? ids : DATA.options.periods.map(p => p.period_id);
}}
function periodLabel(periodId) {{
  const row = DATA.options.periods.find(p => String(p.period_id) === String(periodId));
  return row ? row.period_display : periodId;
}}
function refreshPeriodOptions(selected) {{
  const currentValue = selected !== undefined ? selected : byId("period").value;
  const ids = periodIdsForComparison(byId("comparison").value);
  setSelect("period", ids, currentValue);
  for (const opt of byId("period").options) opt.textContent = periodLabel(opt.value);
}}
function refreshDirectionOptions(selected) {{
  const mode = byId("analysisMode").value;
  const values = mode === "core_contrast" ? [byId("pair").value.split("-")[0], byId("pair").value.split("-")[1], "both"] : ["positive", "negative", "both"];
  setSelect("direction", values.filter(Boolean), selected ?? values[0]);
}}
function selectFirstAvailableCondition(preferredPeriod) {{
  const mode = byId("analysisMode").value;
  let rows = DATA.summary_rows.filter(row =>
    String(row.analysis_mode || "shared_pair") === mode &&
    row.token_profile === byId("profile").value &&
    Number(row.network_window) === Number(byId("window").value) &&
    String(row.neighbor_scope) === byId("neighborScope").value &&
    String(row.pair_count_mode || "sum") === byId("pairCountMode").value &&
    String(row.comparison_type) === byId("comparison").value
  );
  if (preferredPeriod && rows.some(row => String(row.period_id) === preferredPeriod)) {{
    rows = rows.filter(row => String(row.period_id) === preferredPeriod);
  }}
  rows.sort((a,b) => Number(a.period_sort_order)-Number(b.period_sort_order) || String(a.pair_id).localeCompare(String(b.pair_id)));
  const row = rows[0];
  if (!row) {{ refreshPeriodOptions(preferredPeriod); refreshDirectionOptions(); return; }}
  byId("pair").value = String(row.pair_id);
  refreshPeriodOptions(String(row.period_id));
  refreshDirectionOptions();
}}
function refreshModeOptions() {{
  const mode = byId("analysisMode").value;
  const comparisons = DATA.options.comparison_types_by_mode[mode] || DATA.options.comparison_types;
  const countModes = DATA.options.pair_count_modes_by_mode[mode] || DATA.options.pair_count_modes;
  setSelect("comparison", comparisons, comparisons[0]);
  setSelect("pairCountMode", countModes, countModes[0]);
  byId("pairCountMode").disabled = mode === "core_contrast";
  selectFirstAvailableCondition(mode === "core_contrast" ? "global_all_p000" : undefined);
}}
function current() {{
  return {{
    analysisMode: byId("analysisMode").value,
    profile: byId("profile").value,
    window: Number(byId("window").value),
    period: byId("period").value,
    pair: byId("pair").value,
    neighborScope: byId("neighborScope").value,
    pairCountMode: byId("pairCountMode").value,
    comparison: byId("comparison").value,
    metric: byId("metric").value,
    direction: byId("direction").value,
    limit: Number(byId("limit").value),
    q: byId("filterText").value.trim().toLowerCase()
  }};
}}
function rowMatches(row, c) {{
  if (String(row.analysis_mode || "shared_pair") !== c.analysisMode) return false;
  if (row.token_profile !== c.profile) return false;
  if (Number(row.network_window) !== c.window) return false;
  if (String(row.period_id) !== c.period) return false;
  if (String(row.pair_id) !== c.pair) return false;
  if (String(row.neighbor_scope) !== c.neighborScope) return false;
  if (String(row.pair_count_mode || "sum") !== c.pairCountMode) return false;
  if (String(row.comparison_type) !== c.comparison) return false;
  if (c.direction !== "both" && String(row.direction || "positive") !== c.direction) return false;
  if (c.q) {{
    const hay = [row.token,row.dominant_pos,row.dict_lv1,row.dict_lv2,row.included_metrics].map(v => String(v || "").toLowerCase()).join(" ");
    if (!hay.includes(c.q)) return false;
  }}
  return true;
}}
function sortRows(rows, metric, direction) {{
  const copy = rows.slice();
  const first = copy[0];
  const descendingSide = first && first.analysis_mode === "core_contrast" && direction === String(first.focus_b_label);
  if ((direction === "negative" || descendingSide) && (metric === "log_odds_z" || metric === "log_ratio" || metric === "tfidf")) {{
    copy.sort((a,b) => Number(a[metric] || 0) - Number(b[metric] || 0));
  }} else {{
    copy.sort((a,b) => Number(b[metric] || 0) - Number(a[metric] || 0));
  }}
  return copy;
}}
let tableCounter = 0;
function plainText(value) {{
  const div = document.createElement("div");
  div.innerHTML = String(value ?? "");
  return div.textContent || div.innerText || "";
}}
function sortCellValue(value, numeric) {{
  if (numeric) {{
    const n = Number(String(value ?? "").replace(/[% ,]/g, ""));
    return Number.isFinite(n) ? n : Number.NEGATIVE_INFINITY;
  }}
  return plainText(value).toLowerCase();
}}
function sortTableByHeader(th, index, numeric) {{
  const table = th.closest("table");
  const tbody = table ? table.querySelector("tbody") : null;
  if (!tbody) return;
  const ascending = th.dataset.sortDirection !== "asc";
  table.querySelectorAll("th").forEach(cell => {{
    cell.classList.remove("sort-asc", "sort-desc");
    delete cell.dataset.sortDirection;
  }});
  th.dataset.sortDirection = ascending ? "asc" : "desc";
  th.classList.add(ascending ? "sort-asc" : "sort-desc");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  rows.sort((a, b) => {{
    const left = a.children[index]?.dataset.sortValue ?? a.children[index]?.textContent ?? "";
    const right = b.children[index]?.dataset.sortValue ?? b.children[index]?.textContent ?? "";
    const av = sortCellValue(left, numeric);
    const bv = sortCellValue(right, numeric);
    let cmp = 0;
    if (numeric) cmp = av - bv;
    else cmp = String(av).localeCompare(String(bv));
    return ascending ? cmp : -cmp;
  }});
  rows.forEach(row => tbody.appendChild(row));
}}
function table(rows, cols) {{
  if (!rows.length) return `<div class="note">No rows for this condition.</div>`;
  const id = `table-${{++tableCounter}}`;
  const head = `<thead><tr>${{cols.map((c, idx) => `<th class="${{c.num ? "num " : ""}}sortable" onclick="sortTableByHeader(this, ${{idx}}, ${{c.num ? "true" : "false"}})">${{esc(c.label)}}</th>`).join("")}}</tr></thead>`;
  const body = rows.map(row => `<tr>${{cols.map(c => {{
    const raw = c.value ? c.value(row) : row[c.key];
    const sortRaw = c.sortValue ? c.sortValue(row) : (c.key ? row[c.key] : raw);
    const html = c.html ? raw : esc(raw);
    return `<td class="${{c.num ? "num" : ""}}" data-sort-value="${{attr(sortRaw)}}">${{html}}</td>`;
  }}).join("")}}</tr>`).join("");
  return `<table id="${{id}}">${{head}}<tbody>${{body}}</tbody></table>`;
}}
function metricRank(row, metric) {{
  if (row.analysis_mode === "core_contrast") {{
    const side = String(row.direction) === String(row.focus_a_label) ? "a" : "b";
    if (metric === "log_odds_z") return row[`rank_${{side}}`];
    return row[`rank_${{metric}}_${{side}}`];
  }}
  if (metric === "log_odds_z") return row.direction === "negative" ? row.rank_negative : row.rank_positive;
  return row[METRIC_RANK[metric]];
}}
function keynessRows() {{
  const c = current();
  return sortRows(DATA.keyness_rows.filter(row => rowMatches(row, c)), c.metric, c.direction).slice(0, c.limit);
}}
function robustRows() {{
  const c = current();
  return DATA.robust_rows.filter(row => rowMatches(row, c))
    .sort((a,b) => Number(b.robust_score || 0) - Number(a.robust_score || 0) || Math.abs(Number(b.log_odds_z || 0)) - Math.abs(Number(a.log_odds_z || 0)))
    .slice(0, c.limit);
}}
function renderSummary() {{
  const c = current();
  const row = DATA.summary_rows.find(r =>
    String(r.analysis_mode || "shared_pair") === c.analysisMode &&
    r.token_profile === c.profile && Number(r.network_window) === c.window && String(r.period_id) === c.period &&
    String(r.pair_id) === c.pair && String(r.neighbor_scope) === c.neighborScope && String(r.comparison_type) === c.comparison
    && String(r.pair_count_mode || "sum") === c.pairCountMode
  );
  const meta = DATA.meta;
  const items = [
    ["condition", `${{c.analysisMode}} | ${{c.profile}} | w${{c.window}} | ${{periodLabel(c.period)}} | ${{c.pair}}${{c.analysisMode === "shared_pair" ? ` | ${{c.pairCountMode}}` : ""}}`],
    ["source rows", `${{fmt(meta.full_keyness_rows,0)}} keyness / ${{fmt(meta.full_robust_rows,0)}} robust`],
    ["shown payload", `${{fmt(meta.shown_keyness_rows,0)}} keyness / ${{fmt(meta.shown_robust_rows,0)}} robust`]
  ];
  if (row) {{
    items.push(["group rows", `${{fmt(row.keyness_rows,0)}} keyness / ${{fmt(row.robust_rows,0)}} robust`]);
    items.push(["max z", fmt(row.max_log_odds_z,3)]);
    items.push(["max LL", fmt(row.max_log_likelihood,2)]);
  }}
  byId("summary").innerHTML = items.map(([k,v]) => `<div class="pill"><span class="muted">${{esc(k)}}:</span> ${{esc(v)}}</div>`).join("");
}}
function renderKeyness() {{
  const c = current();
  const rows = keynessRows();
  const common = [
    {{label:"rank", num:true, value:r=>fmt(metricRank(r, c.metric),0)}},
    {{label:"token", html:true, value:r=>`<span class="token">${{esc(r.token)}}</span>`}},
    {{key:c.metric, label:c.metric, num:true, value:r=>fmt(r[c.metric], c.metric === "tfidf" ? 5 : 3)}},
    {{key:"log_odds_z", label:"z", num:true, value:r=>fmt(r.log_odds_z,3)}},
    {{key:"log_likelihood", label:"LL", num:true, value:r=>fmt(r.log_likelihood,2)}},
    {{key:"log_ratio", label:"log ratio", num:true, value:r=>fmt(r.log_ratio,3)}},
    {{key:"chi_square", label:"chi-square", num:true, value:r=>fmt(r.chi_square,2)}},
    {{key:"tfidf", label:c.analysisMode === "core_contrast" ? "TF-IDF delta" : "TF-IDF", num:true, value:r=>fmt(r.tfidf,5)}}
  ];
  const shared = [
    {{key:"count_period", label:"count", num:true, value:r=>fmt(r.count_period,0)}},
    {{key:"count_ref", label:"ref", num:true, value:r=>fmt(r.count_ref,0)}},
    {{key:"rate_period", label:"rate", num:true, value:r=>pct(r.rate_period)}},
    {{key:"rate_ref", label:"ref rate", num:true, value:r=>pct(r.rate_ref)}},
    {{key:"edge_a", label:"edge A", num:true, value:r=>fmt(r.edge_a,0)}},
    {{key:"edge_b", label:"edge B", num:true, value:r=>fmt(r.edge_b,0)}},
    {{key:"shared_strength", label:"shared", num:true, value:r=>fmt(r.shared_strength,5)}}
  ];
  const pairParts = c.pair.split("-");
  const contrast = [
    {{key:"count_a", label:`${{pairParts[0]}} count`, num:true, value:r=>fmt(r.count_a,0)}},
    {{key:"count_b", label:`${{pairParts[1]}} count`, num:true, value:r=>fmt(r.count_b,0)}},
    {{key:"rate_a", label:`${{pairParts[0]}} rate`, num:true, value:r=>pct(r.rate_a)}},
    {{key:"rate_b", label:`${{pairParts[1]}} rate`, num:true, value:r=>pct(r.rate_b)}},
    {{key:"norm_a", label:`${{pairParts[0]}} norm`, num:true, value:r=>fmt(r.norm_a,5)}},
    {{key:"norm_b", label:`${{pairParts[1]}} norm`, num:true, value:r=>fmt(r.norm_b,5)}}
  ];
  const tail = [
    {{key:"period_keyness_log_odds_z", label:"period z", num:true, value:r=>fmt(r.period_keyness_log_odds_z,3)}},
    {{key:"dominant_pos", label:"pos"}},
    {{key:"dict_lv2", label:"annotation"}}
  ];
  byId("keyness").innerHTML = table(rows, [...common, ...(c.analysisMode === "core_contrast" ? contrast : shared), ...tail]);
}}
function renderRobust() {{
  const c = current();
  const rows = robustRows();
  byId("robust").innerHTML = table(rows, [
    {{label:"score", key:"robust_score", num:true, value:r=>fmt(r.robust_score,0)}},
    {{label:"class", html:true, value:r=>`<span class="${{esc(r.robust_class)}}">${{esc(r.robust_class)}}</span>`}},
    {{label:"token", html:true, value:r=>`<span class="token">${{esc(r.token)}}</span>`}},
    {{label:"metrics", html:true, value:r=>String(r.included_metrics || "").split(";").map(m=>`<span class="badge">${{esc(m)}}</span>`).join("")}},
    {{label:"z", key:"log_odds_z", num:true, value:r=>fmt(r.log_odds_z,3)}},
    {{label:c.analysisMode === "core_contrast" ? "A count" : "count", num:true, value:r=>fmt(c.analysisMode === "core_contrast" ? r.count_a : r.count_period,0)}},
    {{label:c.analysisMode === "core_contrast" ? "B count" : "ref", num:true, value:r=>fmt(c.analysisMode === "core_contrast" ? r.count_b : r.count_ref,0)}},
    {{label:c.analysisMode === "core_contrast" ? "direction" : "shared", value:r=>c.analysisMode === "core_contrast" ? r.direction : fmt(r.shared_strength,5)}},
    {{label:"period z", key:"period_keyness_log_odds_z", num:true, value:r=>fmt(r.period_keyness_log_odds_z,3)}},
    {{label:"pos", key:"dominant_pos"}},
    {{label:"annotation", key:"dict_lv2"}}
  ]);
}}
function metricTopSet(rows, metric, direction, n) {{
  return new Set(sortRows(rows, metric, direction).slice(0, n).map(r => String(r.token)));
}}
function renderRobustness() {{
  const c = current();
  const rows = DATA.keyness_rows.filter(row => rowMatches(row, {{...c, direction:"both", q:""}}));
  const n = Math.min(30, Number(byId("limit").value || 30));
  const targetDirection = c.direction === "both" ? (c.analysisMode === "core_contrast" ? c.pair.split("-")[0] : "positive") : c.direction;
  const sets = new Map(METRICS.map(metric => [metric, metricTopSet(rows.filter(r => String(r.direction) === targetDirection), metric, targetDirection, n)]));
  const head = `<tr><th>metric</th>${{METRICS.map(m=>`<th>${{esc(m)}}</th>`).join("")}}</tr>`;
  const body = METRICS.map(a => `<tr><th>${{esc(a)}}</th>${{METRICS.map(b => {{
    const A = sets.get(a), B = sets.get(b);
    const inter = [...A].filter(x => B.has(x)).length;
    const union = new Set([...A, ...B]).size;
    const value = union ? inter / union : 0;
    return `<td class="heat" style="--v:${{Math.round(value*100)}}">${{inter}}/${{union}}</td>`;
  }}).join("")}}</tr>`).join("");
  const score = new Map();
  for (const [metric,set] of sets.entries()) for (const token of set) score.set(token, (score.get(token)||0)+1);
  const repeated = [...score.entries()].filter(([_t,s]) => s >= 3).sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0])).slice(0, 40);
  byId("robustness").innerHTML = `<table class="matrix"><tbody>${{head}}${{body}}</tbody></table>` +
    `<div class="note">Repeated in at least 3 metric top-${{n}} lists: ${{repeated.map(([t,s])=>`<span class="badge">${{esc(t)}} (${{s}})</span>`).join(" ") || "none"}}</div>`;
}}
function renderGroupSummary() {{
  const c = current();
  const rows = DATA.summary_rows.filter(r =>
    String(r.analysis_mode || "shared_pair") === c.analysisMode &&
    r.token_profile === c.profile && Number(r.network_window) === c.window &&
    String(r.neighbor_scope) === c.neighborScope && String(r.pair_count_mode || "sum") === c.pairCountMode && String(r.comparison_type) === c.comparison
  ).sort((a,b) => Number(a.period_sort_order)-Number(b.period_sort_order) || String(a.pair_id).localeCompare(String(b.pair_id)));
  byId("groupSummary").innerHTML = table(rows, [
    {{label:"period", key:"period_display"}},
    {{label:"pair", key:"pair_id"}},
    {{label:"keyness rows", key:"keyness_rows", num:true, value:r=>fmt(r.keyness_rows,0)}},
    {{label:"robust", key:"robust_rows", num:true, value:r=>fmt(r.robust_rows,0)}},
    {{label:"strong", key:"strong_robust_rows", num:true, value:r=>fmt(r.strong_robust_rows,0)}},
    {{label:"max z", key:"max_log_odds_z", num:true, value:r=>fmt(r.max_log_odds_z,3)}},
    {{label:"max LL", key:"max_log_likelihood", num:true, value:r=>fmt(r.max_log_likelihood,2)}},
    {{label:"max TF-IDF", key:"max_tfidf", num:true, value:r=>fmt(r.max_tfidf,5)}}
  ]);
}}
function copyTable() {{
  const rows = keynessRows();
  const c = current();
  const cols = c.analysisMode === "core_contrast"
    ? ["token","direction","log_odds_z","log_likelihood","log_ratio","chi_square","tfidf","count_a","count_b","rate_a","rate_b","norm_a","norm_b","period_keyness_log_odds_z","dominant_pos","dict_lv2"]
    : ["token","log_odds_z","log_likelihood","log_ratio","chi_square","tfidf","count_period","count_ref","rate_period","rate_ref","edge_a","edge_b","shared_strength","period_keyness_log_odds_z","dominant_pos","dict_lv2"];
  const text = [cols.join("\\t"), ...rows.map(r => cols.map(c => r[c] ?? "").join("\\t"))].join("\\n");
  navigator.clipboard.writeText(text);
}}
function renderAll() {{
  renderSummary();
  renderRobust();
  renderRobustness();
  renderKeyness();
  renderGroupSummary();
}}
function installControlsCollapse() {{
  const header = document.querySelector("header");
  if (!header || header.dataset.codexControlsCollapseReady === "1") return;
  const controls = header.querySelector(".controls");
  if (!controls) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "codex-controls-toggle";
  button.setAttribute("aria-expanded", "true");
  button.textContent = "Hide controls";
  button.addEventListener("click", () => {{
    const expanded = button.getAttribute("aria-expanded") === "true";
    controls.classList.toggle("codex-controls-hidden", expanded);
    button.setAttribute("aria-expanded", expanded ? "false" : "true");
    button.textContent = expanded ? "Show controls" : "Hide controls";
  }});
  header.insertBefore(button, controls);
  header.dataset.codexControlsCollapseReady = "1";
}}
function init() {{
  setSelect("analysisMode", DATA.options.analysis_modes, "shared_pair");
  setSelect("profile", DATA.options.profiles, "strict");
  setSelect("window", DATA.options.windows.map(String), "10");
  setSelect("pair", DATA.options.pairs, DATA.options.pairs[0]);
  setSelect("neighborScope", DATA.options.neighbor_scopes, "top100");
  refreshModeOptions();
  setSelect("metric", METRICS, "log_odds_z");
  ["period","metric","direction","limit"].forEach(id => byId(id).addEventListener("change", renderAll));
  ["profile","window","neighborScope","pairCountMode"].forEach(id => byId(id).addEventListener("change", () => {{ selectFirstAvailableCondition(); renderAll(); }}));
  byId("analysisMode").addEventListener("change", () => {{ refreshModeOptions(); renderAll(); }});
  byId("pair").addEventListener("change", () => {{ refreshDirectionOptions(); renderAll(); }});
  byId("comparison").addEventListener("change", () => {{ selectFirstAvailableCondition(); renderAll(); }});
  byId("filterText").addEventListener("input", renderAll);
  byId("copyKeyness").addEventListener("click", copyTable);
  installControlsCollapse();
  renderAll();
}}
init();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pair-conditioned keyness dashboard HTML.")
    parser.add_argument("--keyness-csv", default=str(DEFAULT_KEYNESS_CSV), help="pair_conditioned_keyness.csv path.")
    parser.add_argument("--robust-csv", default=str(DEFAULT_ROBUST_CSV), help="robust candidates CSV path.")
    parser.add_argument(
        "--contrast-keyness-csv",
        default=str(DEFAULT_CONTRAST_KEYNESS_CSV),
        help="Direct core A-vs-B keyness CSV path.",
    )
    parser.add_argument(
        "--contrast-robust-csv",
        default=str(DEFAULT_CONTRAST_ROBUST_CSV),
        help="Direct core A-vs-B robust candidates CSV path.",
    )
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), help="Output HTML path.")
    parser.add_argument("--top-per-metric", type=int, default=50, help="Rows retained per metric and condition.")
    parser.add_argument("--max-robust-rows-per-group", type=int, default=80, help="Robust rows retained per condition.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    html = render_html(payload)
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {output_html}", flush=True)
    print(
        f"Keyness rows in payload: {payload['meta']['shown_keyness_rows']} / {payload['meta']['full_keyness_rows']}",
        flush=True,
    )
    print(
        f"Robust rows in payload: {payload['meta']['shown_robust_rows']} / {payload['meta']['full_robust_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
