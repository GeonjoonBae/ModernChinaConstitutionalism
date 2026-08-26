#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build a static dashboard for combined Shenbao network overlap metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "shenbao" / "shenbao_network" / "network_overlap_metrics"
DEFAULT_CORE_CSV = DEFAULT_INPUT_DIR / "network_core_overlap_metrics_combined.csv"
DEFAULT_GROUP_CSV = DEFAULT_INPUT_DIR / "network_group_overlap_metrics_combined.csv"
DEFAULT_OUTPUT_HTML = DEFAULT_INPUT_DIR / "network_overlap_metrics_dashboard.html"
DEFAULT_PERIOD_CSV = ROOT / "shenbao" / "shenbao_interpretation" / "focus_anchor_dashboard" / "focus_period_counts.csv"


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML dashboard from combined overlap metric CSVs."
    )
    parser.add_argument("--core-csv", default=str(DEFAULT_CORE_CSV))
    parser.add_argument("--group-csv", default=str(DEFAULT_GROUP_CSV))
    parser.add_argument("--period-csv", default=str(DEFAULT_PERIOD_CSV))
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML))
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if fieldnames and fieldnames[0].startswith("\ufeff"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    return fieldnames, rows


def date_label(start_date: str, end_date: str) -> str:
    if start_date and end_date:
        return f"{start_date.replace('-', '.')} - {end_date.replace('-', '.')}"
    return ""


def build_period_labels(period_csv: Path) -> Dict[str, str]:
    labels = {
        "global": "global",
        "global_all": "global",
        "global_all_p000": "global",
    }
    if not period_csv.is_file():
        return labels
    _, rows = read_csv(period_csv)
    for row in rows:
        period_id = row.get("period_id", "").strip()
        if not period_id or period_id in labels:
            continue
        if period_id.startswith("long_period_manual"):
            label = date_label(row.get("period_start_date", "").strip(), row.get("period_end_date", "").strip())
            if label:
                labels[period_id] = label
        elif period_id == "global":
            labels[period_id] = "global"
    return labels


def build_payload(core_csv: Path, group_csv: Path, period_csv: Path) -> Dict[str, object]:
    core_columns, core_rows = read_csv(core_csv)
    group_columns, group_rows = read_csv(group_csv)
    return {
        "period_labels": build_period_labels(period_csv),
        "core": {
            "path": str(core_csv),
            "columns": core_columns,
            "rows": core_rows,
        },
        "group": {
            "path": str(group_csv),
            "columns": group_columns,
            "rows": group_rows,
        },
    }


def write_html(payload: Dict[str, object], output_html: Path) -> None:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shenbao Network Overlap Metrics</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #1f2937;
      --muted: #667085;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --warn: #b45309;
      --warn-bg: #fff7ed;
      --ok: #047857;
      --heat: 214, 92, 92;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); }
    main { padding: 18px 24px 32px; }
    .grid {
      display: grid;
      gap: 14px;
    }
    .cards {
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      margin-bottom: 14px;
    }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .card { padding: 12px 14px; }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .card .value {
      font-size: 21px;
      font-weight: 700;
    }
    .panel { padding: 14px; margin-bottom: 14px; }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }
    .field {
      display: grid;
      gap: 4px;
      min-width: 130px;
    }
    label {
      font-size: 12px;
      color: var(--muted);
    }
    select, input[type="search"] {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 9px;
      font: inherit;
    }
    .segmented {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      background: #fff;
    }
    .segmented button {
      border: 0;
      border-right: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      height: 34px;
      padding: 0 12px;
      cursor: pointer;
      font: inherit;
    }
    .segmented button:last-child { border-right: 0; }
    .segmented button.active {
      background: var(--accent);
      color: #fff;
    }
    .viz-grid {
      display: grid;
      grid-template-columns: minmax(360px, 520px) minmax(360px, 1fr);
      gap: 14px;
      align-items: start;
    }
    .matrix {
      display: grid;
      grid-template-columns: 76px repeat(4, minmax(68px, 1fr));
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .matrix > div {
      min-height: 58px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 8px;
      display: grid;
      align-content: center;
      justify-items: center;
      text-align: center;
    }
    .matrix > div:nth-child(5n) { border-right: 0; }
    .matrix .head {
      min-height: 38px;
      background: #f8fafc;
      font-weight: 700;
    }
    .matrix .diag { color: #9aa3b2; background: #fafafa; }
    .matrix .heat {
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }
    .matrix .pair {
      font-size: 11px;
      color: var(--muted);
      font-weight: 500;
    }
    .low-support {
      outline: 2px solid var(--warn);
      outline-offset: -2px;
    }
    .trend-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .trend {
      min-width: 720px;
      display: grid;
      background: #fff;
    }
    .trend > div {
      min-height: 44px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 7px;
      display: grid;
      align-content: center;
      justify-items: center;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }
    .trend .head {
      background: #f8fafc;
      font-weight: 700;
      color: var(--text);
    }
    .line-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 6px 0 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      cursor: pointer;
    }
    .legend-item input { accent-color: currentColor; }
    .legend-muted { opacity: 0.45; }
    .legend-swatch {
      width: 18px;
      height: 3px;
      border-radius: 999px;
      background: currentColor;
    }
    .line-reset {
      height: 26px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    .line-reset:hover { border-color: #98a2b3; }
    .line-charts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }
    .line-chart {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .line-chart h3 {
      margin: 0;
      padding: 10px 12px 0;
      font-size: 13px;
      letter-spacing: 0;
    }
    .line-chart svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .axis-line {
      stroke: #98a2b3;
      stroke-width: 1;
      shape-rendering: crispEdges;
    }
    .grid-line {
      stroke: #e4e7ec;
      stroke-width: 1;
      shape-rendering: crispEdges;
    }
    .axis-text {
      fill: var(--muted);
      font-size: 11px;
    }
    .series-line {
      fill: none;
      stroke-width: 2;
    }
    .series-point {
      fill: #fff;
      stroke-width: 2;
    }
    .series-point.low-support-point {
      fill: #f2f4f7;
      stroke: #c7ced8;
      stroke-width: 2;
    }
    .table-tools {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .checkbox {
      display: inline-flex;
      gap: 7px;
      align-items: center;
      color: var(--muted);
    }
    .table-wrap {
      max-height: 720px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table {
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      min-width: 1900px;
    }
    th, td {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
      text-align: left;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }
    td.num {
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    td.metric {
      font-weight: 650;
    }
    td.status-ok { color: var(--ok); font-weight: 650; }
    td.status-low { color: var(--warn); background: var(--warn-bg); font-weight: 650; }
    .path {
      font-size: 12px;
      color: #475467;
      min-width: 360px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #174ea6;
      font-size: 12px;
      font-weight: 650;
    }
    .note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    @media (max-width: 1000px) {
      main { padding: 14px; }
      .cards { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .viz-grid { grid-template-columns: 1fr; }
      .line-charts { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Shenbao Network Overlap Metrics</h1>
    <div class="subtle">Core token pairs and core-containing token group pairs</div>
  </header>
  <main>
    <section class="grid cards" id="cards"></section>

    <section class="panel">
      <div class="toolbar">
        <div class="field">
          <label>Dataset</label>
          <div class="segmented">
            <button type="button" id="unit-core" class="active">Core</button>
            <button type="button" id="unit-group">Group</button>
          </div>
        </div>
        <div class="field">
          <label for="profile-select">Profile</label>
          <select id="profile-select"></select>
        </div>
        <div class="field">
          <label for="window-select">Window</label>
          <select id="window-select"></select>
        </div>
        <div class="field">
          <label for="period-select">Period</label>
          <select id="period-select"></select>
        </div>
        <div class="field">
          <label for="topn-select">Top N</label>
          <select id="topn-select"></select>
        </div>
        <div class="field">
          <label for="metric-select">Metric</label>
          <select id="metric-select"></select>
        </div>
      </div>
      <div class="viz-grid">
        <div>
          <h2>Pair Heatmap</h2>
          <div id="matrix" class="matrix"></div>
          <div class="note" id="condition-note"></div>
        </div>
        <div>
          <h2>Period Comparison</h2>
          <div class="trend-wrap">
            <div id="trend" class="trend"></div>
          </div>
          <div class="note">Rows are periods; columns are the six keyword pairs under the selected dataset, profile, window, and metric.</div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Period Line Profiles</h2>
      <div class="note">Global is excluded. The chart follows the selected Metric, with long-period units on the x-axis and metric values on the y-axis.</div>
      <div id="line-legend" class="line-legend"></div>
      <div id="line-charts" class="line-charts"></div>
    </section>

    <section class="panel">
      <h2>Selected Condition Values</h2>
      <div class="table-wrap" style="max-height: 360px;">
        <table id="pair-table"></table>
      </div>
    </section>

    <section class="panel">
      <div class="table-tools">
        <div>
          <h2 style="margin-bottom: 4px;">Full Data Table</h2>
          <div class="subtle" id="table-count"></div>
        </div>
        <div class="toolbar" style="margin: 0;">
          <label class="checkbox">
            <input type="checkbox" id="condition-filter">
            selected condition only
          </label>
          <div class="field" style="min-width: 260px;">
            <label for="search-input">Search</label>
            <input type="search" id="search-input" placeholder="token, period, source path">
          </div>
        </div>
      </div>
      <div class="table-wrap">
        <table id="full-table"></table>
      </div>
    </section>
  </main>

  <script>
    const DATA = __DATA_JSON__;
    const ORDER = ["lixian", "xianzheng", "xianfa", "zhixian"];
    const PROFILE_ORDER = ["regex-only", "strict", "full"];
    const LABELS = {lixian: "立憲", xianzheng: "憲政", xianfa: "憲法", zhixian: "制憲"};
    const METRICS = ["direct_strength", "weighted_jaccard", "jaccard", "cosine", "neighbor_mean"];
    const METRIC_LABELS = {
      direct_strength: "direct_strength",
      weighted_jaccard: "weighted_jaccard",
      jaccard: "jaccard",
      cosine: "cosine",
      neighbor_mean: "neighbor_mean = (weighted_jaccard + jaccard + cosine) / 3"
    };
    const PAIR_COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"];
    const UNIT_META = {
      core: {aNorm: "focus_a_norm", aLabel: "focus_a_label", bNorm: "focus_b_norm", bLabel: "focus_b_label"},
      group: {aNorm: "group_a_norm", aLabel: "group_a_label", bNorm: "group_b_norm", bLabel: "group_b_label"}
    };
    const state = {
      unit: "core",
      profile: "",
      window: "",
      period: "",
      topn: "",
      metric: "weighted_jaccard",
      search: "",
      conditionOnly: false,
      sortKey: "",
      sortDir: 1,
      visiblePairs: null
    };

    function rows(unit = state.unit) { return DATA[unit].rows; }
    function cols(unit = state.unit) { return DATA[unit].columns; }
    function unique(values) { return [...new Set(values.filter(Boolean))]; }
    function numeric(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : 0;
    }
    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }
    function formatNumber(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "";
      if (Math.abs(number) >= 1000) return number.toLocaleString("en-US", {maximumFractionDigits: 0});
      if (number === 0) return "0";
      return number.toLocaleString("en-US", {maximumSignificantDigits: 5});
    }
    function sortWindows(values) {
      return values.sort((a, b) => Number(a) - Number(b));
    }
    function sortTopns(values) {
      return values.sort((a, b) => {
        const an = Number(a);
        const bn = Number(b);
        if (an === 0 && bn !== 0) return 1;
        if (bn === 0 && an !== 0) return -1;
        return an - bn;
      });
    }
    function sortProfiles(values) {
      return values.sort((a, b) => {
        const ai = PROFILE_ORDER.includes(a) ? PROFILE_ORDER.indexOf(a) : PROFILE_ORDER.length;
        const bi = PROFILE_ORDER.includes(b) ? PROFILE_ORDER.indexOf(b) : PROFILE_ORDER.length;
        if (ai !== bi) return ai - bi;
        return a.localeCompare(b);
      });
    }
    function sortPeriods(values) {
      return values.sort((a, b) => {
        const ga = a.includes("global") ? -1 : 1;
        const gb = b.includes("global") ? -1 : 1;
        if (ga !== gb) return ga - gb;
        return a.localeCompare(b);
      });
    }
    function periodLabel(periodId) {
      return DATA.period_labels[periodId] || periodId;
    }
    function topnLabel(topn) {
      return String(topn) === "0" ? "all" : String(topn);
    }
    function pairKey(a, b) { return [a, b].sort((x, y) => ORDER.indexOf(x) - ORDER.indexOf(y)).join("|"); }
    function pairLabelFromKey(key) {
      const [a, b] = key.split("|");
      return `${LABELS[a]}-${LABELS[b]}`;
    }
    function pairLabelFromRow(row) {
      const meta = UNIT_META[state.unit];
      return `${row[meta.aLabel]}-${row[meta.bLabel]}`;
    }
    function pairKeyFromRow(row, unit = state.unit) {
      const meta = UNIT_META[unit];
      return pairKey(row[meta.aNorm], row[meta.bNorm]);
    }
    function metricValue(row, metricKey) {
      if (!row) return 0;
      if (metricKey === "neighbor_mean") {
        return (numeric(row.weighted_jaccard) + numeric(row.jaccard) + numeric(row.cosine)) / 3;
      }
      return numeric(row[metricKey]);
    }
    function metricMax(unit = state.unit, metric = state.metric) {
      return Math.max(0, ...rows(unit)
        .filter(row => !state.topn || row.topn === state.topn)
        .map(row => metricValue(row, metric)));
    }
    function periodLinePeriods(currentRows) {
      return sortPeriods(unique(currentRows.map(row => row.period_id))).filter(period => !period.includes("global"));
    }
    function svgText(x, y, text, extra = "") {
      return `<text x="${x}" y="${y}" class="axis-text" ${extra}>${escapeHtml(text)}</text>`;
    }
    function heatColor(value, max) {
      if (!max || value <= 0) return "rgba(255,255,255,1)";
      const t = Math.min(1, value / max);
      const alpha = 0.08 + t * 0.78;
      return `rgba(var(--heat), ${alpha})`;
    }
    function isMetricColumn(column) {
      return METRICS.includes(column) || column === "shared_neighbor_count";
    }
    function setupCards() {
      const all = [...DATA.core.rows, ...DATA.group.rows];
      const conditionKeys = unique(all.map(row => `${row.token_profile}|${row.network_window}|${row.period_id}`));
      const lowSupport = all.filter(row => row.support_status && row.support_status !== "ok").length;
      const cardData = [
        ["Core rows", DATA.core.rows.length],
        ["Group rows", DATA.group.rows.length],
        ["Conditions", conditionKeys.length],
        ["Low support rows", lowSupport]
      ];
      document.getElementById("cards").innerHTML = cardData.map(([label, value]) => `
        <div class="card"><div class="label">${escapeHtml(label)}</div><div class="value">${formatNumber(value)}</div></div>
      `).join("");
    }
    function populateSelect(id, values, selected, labeler = value => value) {
      const select = document.getElementById(id);
      select.innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(labeler(value))}</option>`).join("");
      if (values.includes(selected)) select.value = selected;
      else if (values.length) select.value = values[0];
      return select.value;
    }
    function refreshControls() {
      document.getElementById("unit-core").classList.toggle("active", state.unit === "core");
      document.getElementById("unit-group").classList.toggle("active", state.unit === "group");
      const currentRows = rows();
      state.profile = populateSelect("profile-select", sortProfiles(unique(currentRows.map(row => row.token_profile))), state.profile);
      state.window = populateSelect("window-select", sortWindows(unique(currentRows.map(row => row.network_window))), state.window);
      state.period = populateSelect("period-select", sortPeriods(unique(currentRows.map(row => row.period_id))), state.period, periodLabel);
      state.topn = populateSelect("topn-select", sortTopns(unique(currentRows.map(row => row.topn))), state.topn, topnLabel);
      state.metric = populateSelect("metric-select", METRICS, state.metric);
    }
    function conditionRows() {
      return rows().filter(row =>
        row.token_profile === state.profile &&
        row.network_window === state.window &&
        row.period_id === state.period &&
        row.topn === state.topn
      );
    }
    function renderMatrix() {
      const matrix = document.getElementById("matrix");
      const current = conditionRows();
      const max = metricMax();
      const byPair = new Map(current.map(row => [pairKeyFromRow(row), row]));
      let html = `<div class="head"></div>${ORDER.map(norm => `<div class="head">${LABELS[norm]}</div>`).join("")}`;
      for (const rowNorm of ORDER) {
        html += `<div class="head">${LABELS[rowNorm]}</div>`;
        for (const colNorm of ORDER) {
          if (rowNorm === colNorm) {
            html += `<div class="diag">-</div>`;
            continue;
          }
          const item = byPair.get(pairKey(rowNorm, colNorm));
          const value = metricValue(item, state.metric);
          const low = item && item.support_status !== "ok" ? " low-support" : "";
          html += `<div class="heat${low}" style="background:${heatColor(value, max)}">
            <div>${formatNumber(value)}</div>
            <div class="pair">${item ? escapeHtml(item.pair_class) : ""}</div>
          </div>`;
        }
      }
      matrix.innerHTML = html;
      const lowCount = current.filter(row => row.support_status !== "ok").length;
      document.getElementById("condition-note").textContent =
        `${state.unit} | ${state.profile} | w${state.window} | top ${topnLabel(state.topn)} | ${periodLabel(state.period)} | ${state.metric} | rows ${current.length} | low support ${lowCount}`;
    }
    function renderTrend() {
      const trend = document.getElementById("trend");
      const currentRows = rows().filter(row =>
        row.token_profile === state.profile &&
        row.network_window === state.window &&
        row.topn === state.topn
      );
      const periods = sortPeriods(unique(currentRows.map(row => row.period_id)));
      const pairKeys = ORDER.flatMap((a, index) => ORDER.slice(index + 1).map(b => pairKey(a, b)));
      const max = metricMax();
      const byPeriodPair = new Map(currentRows.map(row => [`${row.period_id}|${pairKeyFromRow(row)}`, row]));
      trend.style.gridTemplateColumns = `150px repeat(${pairKeys.length}, minmax(82px, 1fr))`;
      let html = `<div class="head">period</div>${pairKeys.map(key => {
        const [a, b] = key.split("|");
        return `<div class="head">${LABELS[a]}-${LABELS[b]}</div>`;
      }).join("")}`;
      for (const period of periods) {
        html += `<div class="head">${escapeHtml(periodLabel(period))}</div>`;
        for (const key of pairKeys) {
          const item = byPeriodPair.get(`${period}|${key}`);
          const value = metricValue(item, state.metric);
          const low = item && item.support_status !== "ok" ? " low-support" : "";
          html += `<div class="${low}" style="background:${heatColor(value, max)}">${formatNumber(value)}</div>`;
        }
      }
      trend.innerHTML = html;
    }
    function renderLineLegend(pairKeys) {
      const selected = state.visiblePairs === null ? new Set(pairKeys) : new Set(state.visiblePairs);
      const legend = document.getElementById("line-legend");
      legend.innerHTML = `
        <button type="button" class="line-reset" id="line-show-all">Show all</button>
        ${pairKeys.map((key, index) => {
          const checked = selected.has(key);
          return `<label class="legend-item${checked ? "" : " legend-muted"}" style="color:${PAIR_COLORS[index % PAIR_COLORS.length]}">
            <input type="checkbox" data-pair="${escapeHtml(key)}" ${checked ? "checked" : ""}>
            <span class="legend-swatch"></span>${escapeHtml(pairLabelFromKey(key))}
          </label>`;
        }).join("")}
      `;
      document.getElementById("line-show-all").addEventListener("click", () => {
        state.visiblePairs = null;
        renderLineCharts();
      });
      legend.querySelectorAll("input[data-pair]").forEach(input => {
        input.addEventListener("change", () => {
          state.visiblePairs = Array.from(legend.querySelectorAll("input[data-pair]:checked")).map(item => item.dataset.pair);
          renderLineCharts();
        });
      });
    }
    function visiblePairKeys(pairKeys) {
      if (state.visiblePairs === null) return pairKeys;
      return pairKeys.filter(key => state.visiblePairs.includes(key));
    }
    function renderLineCharts() {
      const container = document.getElementById("line-charts");
      const currentRows = rows().filter(row =>
        row.token_profile === state.profile &&
        row.network_window === state.window &&
        row.topn === state.topn
      );
      const periods = periodLinePeriods(currentRows);
      const pairKeys = ORDER.flatMap((a, index) => ORDER.slice(index + 1).map(b => pairKey(a, b)));
      const activePairKeys = visiblePairKeys(pairKeys);
      renderLineLegend(pairKeys);
      if (!periods.length) {
        container.innerHTML = `<div class="note">No period rows are available for this condition.</div>`;
        return;
      }
      if (!activePairKeys.length) {
        container.innerHTML = `<div class="note">Select one or more pairs to display.</div>`;
        return;
      }
      const byPeriodPair = new Map(currentRows.map(row => [`${row.period_id}|${pairKeyFromRow(row)}`, row]));
      const width = 860;
      const height = 330;
      const margin = {top: 22, right: 30, bottom: 58, left: 72};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xStep = periods.length > 1 ? plotWidth / (periods.length - 1) : 0;
      const values = [];
      for (const period of periods) {
        for (const key of activePairKeys) values.push(metricValue(byPeriodPair.get(`${period}|${key}`), state.metric));
      }
      const maxValue = Math.max(0, ...values);
      const yMax = maxValue > 0 ? maxValue : 1;
      const ticks = [0, 0.25, 0.5, 0.75, 1].map(tick => tick * yMax);
      const xPos = index => margin.left + index * xStep;
      const yPos = value => margin.top + plotHeight - (value / yMax) * plotHeight;
      let svg = "";
      for (const tick of ticks) {
        const y = yPos(tick);
        svg += `<line class="grid-line" x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}"></line>`;
        svg += svgText(margin.left - 8, y + 4, formatNumber(tick), 'text-anchor="end"');
      }
      for (const [index, period] of periods.entries()) {
        const x = xPos(index);
        const labelY = margin.top + plotHeight + 28;
        svg += `<line class="grid-line" x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}"></line>`;
        svg += `<text x="${x - 4}" y="${labelY}" class="axis-text" text-anchor="end" transform="rotate(-28 ${x - 4} ${labelY})">${escapeHtml(periodLabel(period))}</text>`;
      }
      svg += `<line class="axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}"></line>`;
      svg += `<line class="axis-line" x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}"></line>`;
      activePairKeys.forEach((key) => {
        const pairIndex = pairKeys.indexOf(key);
        const points = periods.map((period, periodIndex) => {
          const row = byPeriodPair.get(`${period}|${key}`);
          const value = metricValue(row, state.metric);
          return {
            row,
            period,
            value,
            x: xPos(periodIndex),
            y: yPos(value)
          };
        });
        const color = PAIR_COLORS[pairIndex % PAIR_COLORS.length];
        svg += `<polyline class="series-line" stroke="${color}" points="${points.map(point => `${point.x},${point.y}`).join(" ")}"></polyline>`;
        for (const point of points) {
          const lowClass = point.row && point.row.support_status !== "ok" ? " low-support-point" : "";
          svg += `<circle class="series-point${lowClass}" cx="${point.x}" cy="${point.y}" r="4" stroke="${color}">
            <title>${escapeHtml(`${pairLabelFromKey(key)} | ${periodLabel(point.period)} | ${state.metric}: ${formatNumber(point.value)} | ${point.row ? point.row.support_status : "missing"}`)}</title>
          </circle>`;
        }
      });
      container.innerHTML = `<div class="line-chart">
        <h3>${escapeHtml(METRIC_LABELS[state.metric] || state.metric)}</h3>
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(state.metric)} period line chart">${svg}</svg>
      </div>`;
    }
    function renderPairTable() {
      const columns = [
        "pair_class", UNIT_META[state.unit].aLabel, UNIT_META[state.unit].bLabel,
        "direct_strength", "weighted_jaccard", "jaccard", "cosine", "neighbor_mean",
        "shared_neighbor_count", "shared_neighbors", "support_status", "support_notes"
      ];
      const current = conditionRows().slice().sort((a, b) => {
        const classOrder = a.pair_class.localeCompare(b.pair_class);
        if (classOrder !== 0) return classOrder;
        return pairLabelFromRow(a).localeCompare(pairLabelFromRow(b));
      });
      renderTable("pair-table", columns, current, false);
    }
    function fullTableRows() {
      let out = rows();
      if (state.conditionOnly) {
        out = out.filter(row =>
          row.token_profile === state.profile &&
          row.network_window === state.window &&
          row.period_id === state.period &&
          row.topn === state.topn
        );
      }
      const query = state.search.trim().toLowerCase();
      if (query) {
        out = out.filter(row => {
          const corpus = Object.values(row).concat([periodLabel(row.period_id), formatNumber(metricValue(row, "neighbor_mean"))]);
          return corpus.some(value => String(value).toLowerCase().includes(query));
        });
      }
      if (state.sortKey) {
        const key = state.sortKey;
        const dir = state.sortDir;
        out = out.slice().sort((a, b) => {
          const na = isMetricColumn(key) ? metricValue(a, key) : Number(a[key]);
          const nb = isMetricColumn(key) ? metricValue(b, key) : Number(b[key]);
          if (Number.isFinite(na) && Number.isFinite(nb)) return (na - nb) * dir;
          return String(a[key] ?? "").localeCompare(String(b[key] ?? "")) * dir;
        });
      }
      return out;
    }
    function fullTableColumns() {
      const columns = cols().filter(column => column !== "source_file");
      if (!columns.includes("neighbor_mean")) {
        const cosineIndex = columns.indexOf("cosine");
        if (cosineIndex >= 0) columns.splice(cosineIndex + 1, 0, "neighbor_mean");
        else columns.push("neighbor_mean");
      }
      return columns;
    }
    function cellClass(column, value) {
      const classes = [];
      if (value !== "" && Number.isFinite(Number(value))) classes.push("num");
      if (isMetricColumn(column)) classes.push("metric");
      if (column === "support_status") {
        classes.push(value === "ok" ? "status-ok" : "status-low");
      }
      if (column === "source_file") classes.push("path");
      return classes.join(" ");
    }
    function renderTable(tableId, columns, tableRows, sortable = true) {
      const table = document.getElementById(tableId);
      const maxes = Object.fromEntries(columns.filter(isMetricColumn).map(column => [
        column, Math.max(0, ...tableRows.map(row => metricValue(row, column)))
      ]));
      const header = `<thead><tr>${columns.map(column => {
        const marker = state.sortKey === column ? (state.sortDir > 0 ? " ▲" : " ▼") : "";
        return `<th data-column="${escapeHtml(column)}">${escapeHtml(column)}${sortable ? marker : ""}</th>`;
      }).join("")}</tr></thead>`;
      const body = `<tbody>${tableRows.map(row => `<tr>${columns.map(column => {
        const raw = column === "neighbor_mean" ? metricValue(row, column) : row[column] ?? "";
        let display = Number.isFinite(Number(raw)) && raw !== "" ? formatNumber(raw) : raw;
        if (column === "period_id") display = periodLabel(raw);
        if (column === "topn") display = topnLabel(raw);
        const style = isMetricColumn(column) ? ` style="background:${heatColor(metricValue(row, column), maxes[column])}"` : "";
        return `<td class="${cellClass(column, raw)}"${style}>${escapeHtml(display)}</td>`;
      }).join("")}</tr>`).join("")}</tbody>`;
      table.innerHTML = header + body;
      if (sortable) {
        table.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
          const column = th.dataset.column;
          if (state.sortKey === column) state.sortDir *= -1;
          else {
            state.sortKey = column;
            state.sortDir = 1;
          }
          renderFullTable();
        }));
      }
    }
    function renderFullTable() {
      const current = fullTableRows();
      document.getElementById("table-count").textContent = `${current.length} visible rows from ${rows().length} ${state.unit} rows`;
      renderTable("full-table", fullTableColumns(), current, true);
    }
    function renderAll() {
      refreshControls();
      renderMatrix();
      renderTrend();
      renderLineCharts();
      renderPairTable();
      renderFullTable();
    }
    function bindEvents() {
      document.getElementById("unit-core").addEventListener("click", () => { state.unit = "core"; state.sortKey = ""; renderAll(); });
      document.getElementById("unit-group").addEventListener("click", () => { state.unit = "group"; state.sortKey = ""; renderAll(); });
      document.getElementById("profile-select").addEventListener("change", event => { state.profile = event.target.value; renderAll(); });
      document.getElementById("window-select").addEventListener("change", event => { state.window = event.target.value; renderAll(); });
      document.getElementById("period-select").addEventListener("change", event => { state.period = event.target.value; renderAll(); });
      document.getElementById("topn-select").addEventListener("change", event => { state.topn = event.target.value; renderAll(); });
      document.getElementById("metric-select").addEventListener("change", event => { state.metric = event.target.value; renderAll(); });
      document.getElementById("search-input").addEventListener("input", event => { state.search = event.target.value; renderFullTable(); });
      document.getElementById("condition-filter").addEventListener("change", event => { state.conditionOnly = event.target.checked; renderFullTable(); });
    }
    function initializeDefaults() {
      const currentRows = rows();
      state.profile = sortProfiles(unique(currentRows.map(row => row.token_profile)))[0] || "";
      state.window = sortWindows(unique(currentRows.map(row => row.network_window))).includes("10") ? "10" : sortWindows(unique(currentRows.map(row => row.network_window)))[0] || "";
      state.period = sortPeriods(unique(currentRows.map(row => row.period_id)))[0] || "";
      const topns = sortTopns(unique(currentRows.map(row => row.topn)));
      state.topn = topns.includes("100") ? "100" : topns[0] || "";
    }
    setupCards();
    initializeDefaults();
    bindEvents();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    core_csv = Path(args.core_csv).expanduser().resolve()
    group_csv = Path(args.group_csv).expanduser().resolve()
    period_csv = Path(args.period_csv).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    payload = build_payload(core_csv, group_csv, period_csv)
    write_html(payload, output_html)
    print(f"[ok] dashboard: {output_html}")
    print(f"[ok] core rows: {len(payload['core']['rows'])}")
    print(f"[ok] group rows: {len(payload['group']['rows'])}")


if __name__ == "__main__":
    main()
