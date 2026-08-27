#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from shenbao_html_controls import inject_controls_collapse
from shenbao_region_analysis_utils import (
    DEFAULT_FOREIGN_REGION_DIR,
    log,
    normalize_profile,
    parse_bool,
    parse_int_list,
    parse_list,
    read_csv,
    write_json,
)


MAJOR_REGIONS = [
    "United States",
    "Japan",
    "United Kingdom/Britain",
    "India",
    "Germany",
    "France",
    "Russia",
    "Soviet Union",
]

DEFAULT_INPUT_CSV = (
    DEFAULT_FOREIGN_REGION_DIR / "network_summary" / "region_network_top_neighbors.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_FOREIGN_REGION_DIR / "network_html"
DEFAULT_OUTPUT_HTML = DEFAULT_OUTPUT_DIR / "region_multi_ego_network_dashboard.html"
DEFAULT_TOKENS_ROOT = Path(__file__).resolve().parent / "shenbao" / "shenbao_network" / "applied_tokens"

POS_GROUP_ORDER = [
    "noun_entity",
    "action_process",
    "state_attitude",
    "function_quantity_other",
    "missing",
]
POS_GROUP_LABELS = {
    "noun_entity": "Nouns, institutions, actors",
    "action_process": "Actions and procedures",
    "state_attitude": "States, evaluations, attitudes",
    "function_quantity_other": "Function, quantity, other",
    "missing": "Missing POS",
}
DEFAULT_POS_GROUPS = ["noun_entity", "action_process", "state_attitude", "missing"]
STATE_ATTITUDE_POS = {"VH", "VHC", "VJ", "VK", "VL"}
FUNCTION_QUANTITY_POS = {
    "D",
    "Da",
    "Dfa",
    "Dfb",
    "Di",
    "Dk",
    "Neu",
    "Neqa",
    "Neqb",
    "Caa",
    "Cab",
    "Cba",
    "Cbb",
    "P",
    "T",
    "I",
    "DE",
    "SHI",
    "FW",
}

REQUIRED_COLUMNS = [
    "data_scope",
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "region_norm",
    "alter_token",
    "edge_weight",
]
OPTIONAL_COLUMNS = [
    "center_mode",
    "period_label",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "macro_region",
    "neighbor_rank",
    "neighbor_share_within_region",
    "pmi_mean",
    "ppmi_mean",
    "avg_distance_mean",
    "alter_is_region_reference",
    "alter_region_norm",
    "alter_macro_region",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML dashboard for multi-region ego-network exploration."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-html", default="")
    parser.add_argument(
        "--title",
        default="Multi-Region Ego Networks for Foreign-Related Mentions in Shun Pao",
    )
    parser.add_argument("--data-scopes", default="body")
    parser.add_argument("--profiles", default="regex-only,strict,full")
    parser.add_argument("--period-set-ids", default="global,long_period_manual")
    parser.add_argument("--windows", default="5,10,20")
    parser.add_argument(
        "--region-mode",
        choices=["major", "all", "custom"],
        default="major",
        help="major uses the default eight regions; custom uses --region-norms.",
    )
    parser.add_argument(
        "--region-norms",
        default="",
        help="Comma-separated region_norm list. Spaces inside names are preserved.",
    )
    parser.add_argument("--max-neighbors-per-region", type=int, default=100)
    parser.add_argument("--default-topn-neighbors", type=int, default=20)
    parser.add_argument("--min-edge-weight", type=float, default=1.0)
    parser.add_argument("--tokens-root", default=str(DEFAULT_TOKENS_ROOT))
    parser.add_argument("--attach-pos", type=parse_bool, default=True)
    parser.add_argument("--ambiguous-pos-threshold", type=float, default=0.6)
    parser.add_argument(
        "--default-include-region-reference-neighbors",
        type=parse_bool,
        default=False,
        help="Whether region-reference neighbor tokens are shown by default in the HTML.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path for a generation summary JSON.",
    )
    parser.add_argument(
        "--ui-mode",
        choices=["region", "core"],
        default="region",
        help="Visible UI terminology in the rendered HTML.",
    )
    parser.add_argument(
        "--hide-data-scope",
        type=parse_bool,
        default=False,
        help="Hide the data-scope control in the rendered HTML and use the first available scope.",
    )
    parser.add_argument(
        "--edge-context-csv",
        default="",
        help="Optional CSV of edge-level representative contexts to show when an edge is clicked.",
    )
    parser.add_argument(
        "--max-edge-contexts-per-edge-payload",
        type=int,
        default=8,
        help="Maximum context rows per edge to embed in the HTML payload.",
    )
    parser.add_argument(
        "--public-release",
        type=parse_bool,
        default=False,
        help="Omit context excerpts and redact local source paths for a public dashboard.",
    )
    parser.add_argument(
        "--prepared-output-csv",
        default="",
        help="Optional path for the filtered, POS-enriched aggregate rows embedded in the HTML.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out["token_profile"] = out["token_profile"].map(normalize_profile)
    if "center_mode" not in out.columns:
        out["center_mode"] = "grouped_focus"
    out["network_window"] = pd.to_numeric(out["network_window"], errors="coerce").astype("Int64").astype(str)
    out["network_window"] = out["network_window"].replace({"<NA>": ""})
    out["edge_weight"] = pd.to_numeric(out["edge_weight"], errors="coerce").fillna(0.0)
    for column in [
        "neighbor_rank",
        "neighbor_share_within_region",
        "pmi_mean",
        "ppmi_mean",
        "avg_distance_mean",
        "period_sort_order",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["alter_is_region_reference"] = normalize_bool_series(out["alter_is_region_reference"])
    for column in [
        "data_scope",
        "center_mode",
        "period_set_id",
        "period_id",
        "period_label",
        "period_start_date",
        "period_end_date",
        "region_norm",
        "macro_region",
        "alter_token",
        "alter_region_norm",
        "alter_macro_region",
    ]:
        out[column] = out[column].fillna("").astype(str)
    out.loc[out["period_label"].eq(""), "period_label"] = out["period_id"]
    return out


def pos_group_for_pos(pos: object) -> str:
    text = str(pos or "").strip()
    if not text or text == "(missing)":
        return "missing"
    if text in FUNCTION_QUANTITY_POS:
        return "function_quantity_other"
    if text.startswith("N"):
        return "noun_entity"
    if text in STATE_ATTITUDE_POS:
        return "state_attitude"
    if text.startswith("V"):
        return "action_process"
    if text == "A":
        return "state_attitude"
    return "function_quantity_other"


def empty_pos_lookup(tokens: Sequence[str], profile: str, threshold: float) -> pd.DataFrame:
    rows = []
    for token in sorted(set(str(value) for value in tokens if str(value))):
        rows.append(
            {
                "token_profile": profile,
                "alter_token": token,
                "dominant_pos": "(missing)",
                "dominant_pos_count": 0,
                "dominant_pos_token_count": 0,
                "dominant_pos_share": 0.0,
                "pos_group": "missing",
                "pos_group_label": POS_GROUP_LABELS["missing"],
                "pos_is_ambiguous": True,
            }
        )
    return pd.DataFrame(rows)


def dominant_pos_lookup(tokens_root: Path, profile: str, tokens: Sequence[str], threshold: float) -> pd.DataFrame:
    token_set = {str(value) for value in tokens if str(value)}
    if not token_set:
        return empty_pos_lookup([], profile, threshold)
    path = tokens_root / profile.replace("-", "_") / "tokens.parquet"
    if not path.exists():
        log(f"Warning: POS token parquet not found for profile {profile}: {path}")
        return empty_pos_lookup(token_set, profile, threshold)
    try:
        work = pd.read_parquet(path, columns=["token", "pos"])
    except Exception as exc:
        log(f"Warning: failed to read POS token parquet for profile {profile}: {exc}")
        return empty_pos_lookup(token_set, profile, threshold)
    work["token"] = work["token"].fillna("").astype(str)
    work["pos"] = work["pos"].fillna("").astype(str)
    work = work[work["token"].isin(token_set) & work["pos"].ne("")]
    if work.empty:
        return empty_pos_lookup(token_set, profile, threshold)
    counts = work.groupby(["token", "pos"], dropna=False).size().reset_index(name="dominant_pos_count")
    counts = counts.sort_values(["token", "dominant_pos_count", "pos"], ascending=[True, False, True], kind="mergesort")
    dominant = counts.drop_duplicates("token", keep="first").rename(columns={"token": "alter_token", "pos": "dominant_pos"})
    totals = work.groupby("token", dropna=False).size().reset_index(name="dominant_pos_token_count")
    dominant = dominant.merge(totals.rename(columns={"token": "alter_token"}), on="alter_token", how="left")
    dominant["dominant_pos_share"] = dominant["dominant_pos_count"] / dominant["dominant_pos_token_count"].where(
        dominant["dominant_pos_token_count"].ne(0)
    )
    dominant["dominant_pos_share"] = dominant["dominant_pos_share"].fillna(0.0)
    dominant["token_profile"] = profile
    dominant["pos_group"] = dominant["dominant_pos"].map(pos_group_for_pos)
    dominant["pos_group_label"] = dominant["pos_group"].map(POS_GROUP_LABELS).fillna(POS_GROUP_LABELS["function_quantity_other"])
    dominant["pos_is_ambiguous"] = dominant["dominant_pos_share"].lt(float(threshold))
    missing = sorted(token_set.difference(set(dominant["alter_token"])))
    if missing:
        dominant = pd.concat([dominant, empty_pos_lookup(missing, profile, threshold)], ignore_index=True)
    return dominant[
        [
            "token_profile",
            "alter_token",
            "dominant_pos",
            "dominant_pos_count",
            "dominant_pos_token_count",
            "dominant_pos_share",
            "pos_group",
            "pos_group_label",
            "pos_is_ambiguous",
        ]
    ]


def attach_pos_metadata(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    threshold = max(0.0, min(1.0, float(args.ambiguous_pos_threshold)))
    if not args.attach_pos:
        existing_columns = {
            "dominant_pos",
            "dominant_pos_count",
            "dominant_pos_token_count",
            "dominant_pos_share",
            "pos_group",
            "pos_group_label",
            "pos_is_ambiguous",
            "region_dominant_pos",
            "region_dominant_pos_count",
            "region_dominant_pos_token_count",
            "region_dominant_pos_share",
            "region_pos_group",
            "region_pos_group_label",
            "region_pos_is_ambiguous",
        }
        if existing_columns.issubset(out.columns):
            for column in [
                "dominant_pos_count",
                "dominant_pos_token_count",
                "region_dominant_pos_count",
                "region_dominant_pos_token_count",
            ]:
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
            for column in ["dominant_pos_share", "region_dominant_pos_share"]:
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
            out["pos_is_ambiguous"] = normalize_bool_series(out["pos_is_ambiguous"])
            out["region_pos_is_ambiguous"] = normalize_bool_series(out["region_pos_is_ambiguous"])
            return out
        out["dominant_pos"] = "(missing)"
        out["dominant_pos_count"] = 0
        out["dominant_pos_token_count"] = 0
        out["dominant_pos_share"] = 0.0
        out["pos_group"] = "missing"
        out["pos_group_label"] = POS_GROUP_LABELS["missing"]
        out["pos_is_ambiguous"] = True
        return out
    tokens_root = Path(args.tokens_root).expanduser().resolve()
    lookups = []
    for profile in ordered_unique(out["token_profile"]):
        profile_rows = out.loc[out["token_profile"].eq(profile)]
        profile_tokens = pd.concat(
            [
                profile_rows["alter_token"].dropna().astype(str),
                profile_rows["region_norm"].dropna().astype(str),
            ],
            ignore_index=True,
        ).unique()
        log(f"Attaching dominant POS for {profile}: {len(profile_tokens):,} tokens")
        lookups.append(dominant_pos_lookup(tokens_root, profile, profile_tokens, threshold))
    if not lookups:
        return attach_pos_metadata(out.assign(), argparse.Namespace(attach_pos=False, ambiguous_pos_threshold=threshold))
    lookup = pd.concat(lookups, ignore_index=True)
    region_lookup = lookup.rename(
        columns={
            "alter_token": "region_norm",
            "dominant_pos": "region_dominant_pos",
            "dominant_pos_count": "region_dominant_pos_count",
            "dominant_pos_token_count": "region_dominant_pos_token_count",
            "dominant_pos_share": "region_dominant_pos_share",
            "pos_group": "region_pos_group",
            "pos_group_label": "region_pos_group_label",
            "pos_is_ambiguous": "region_pos_is_ambiguous",
        }
    )
    out = out.merge(lookup, on=["token_profile", "alter_token"], how="left")
    out = out.merge(
        region_lookup[
            [
                "token_profile",
                "region_norm",
                "region_dominant_pos",
                "region_dominant_pos_count",
                "region_dominant_pos_token_count",
                "region_dominant_pos_share",
                "region_pos_group",
                "region_pos_group_label",
                "region_pos_is_ambiguous",
            ]
        ],
        on=["token_profile", "region_norm"],
        how="left",
    )
    out["dominant_pos"] = out["dominant_pos"].fillna("(missing)")
    out["dominant_pos_count"] = pd.to_numeric(out["dominant_pos_count"], errors="coerce").fillna(0).astype(int)
    out["dominant_pos_token_count"] = pd.to_numeric(out["dominant_pos_token_count"], errors="coerce").fillna(0).astype(int)
    out["dominant_pos_share"] = pd.to_numeric(out["dominant_pos_share"], errors="coerce").fillna(0.0)
    out["pos_group"] = out["pos_group"].fillna("missing")
    out["pos_group_label"] = out["pos_group_label"].fillna(POS_GROUP_LABELS["missing"])
    out["pos_is_ambiguous"] = normalize_bool_series(out["pos_is_ambiguous"])
    out["region_dominant_pos"] = out["region_dominant_pos"].fillna("(missing)")
    out["region_dominant_pos_count"] = pd.to_numeric(out["region_dominant_pos_count"], errors="coerce").fillna(0).astype(int)
    out["region_dominant_pos_token_count"] = (
        pd.to_numeric(out["region_dominant_pos_token_count"], errors="coerce").fillna(0).astype(int)
    )
    out["region_dominant_pos_share"] = pd.to_numeric(out["region_dominant_pos_share"], errors="coerce").fillna(0.0)
    out["region_pos_group"] = out["region_pos_group"].fillna("missing")
    out["region_pos_group_label"] = out["region_pos_group_label"].fillna(POS_GROUP_LABELS["missing"])
    out["region_pos_is_ambiguous"] = normalize_bool_series(out["region_pos_is_ambiguous"])
    return out


def selected_regions(args: argparse.Namespace, df: pd.DataFrame) -> List[str]:
    available = list(dict.fromkeys(df["region_norm"].dropna().astype(str).sort_values()))
    if args.region_mode == "all":
        return available
    if args.region_mode == "custom":
        requested = [part.strip() for part in str(args.region_norms).split(",") if part.strip()]
        missing = [region for region in requested if region not in set(available)]
        if missing:
            log(f"Warning: requested regions not found in input data: {missing}")
        return [region for region in requested if region in set(available)]
    return [region for region in MAJOR_REGIONS if region in set(available)]


def filter_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data_scopes = parse_list(args.data_scopes)
    profiles = [normalize_profile(value) for value in parse_list(args.profiles)]
    period_set_ids = parse_list(args.period_set_ids)
    windows = [str(value) for value in parse_int_list(args.windows)]

    out = df.copy()
    if data_scopes:
        out = out[out["data_scope"].isin(data_scopes)]
    if profiles:
        out = out[out["token_profile"].isin(profiles)]
    if period_set_ids:
        out = out[out["period_set_id"].isin(period_set_ids)]
    if windows:
        out = out[out["network_window"].isin(windows)]
    out = out[out["edge_weight"].ge(float(args.min_edge_weight))]

    regions = selected_regions(args, out)
    if not regions:
        raise ValueError("No regions remain after filtering.")
    out = out[out["region_norm"].isin(regions)].copy()
    out = out.sort_values(
        [
            "data_scope",
            "token_profile",
            "network_window",
            "period_set_id",
            "period_sort_order",
            "period_id",
            "region_norm",
            "neighbor_rank",
            "edge_weight",
            "alter_token",
        ],
        ascending=[True, True, True, True, True, True, True, True, False, True],
        kind="mergesort",
    )
    if args.max_neighbors_per_region > 0:
        group_cols = [
            "data_scope",
            "token_profile",
            "network_window",
            "center_mode",
            "period_set_id",
            "period_id",
            "region_norm",
        ]
        out = out[out.groupby(group_cols, dropna=False).cumcount().lt(args.max_neighbors_per_region)]
    return out


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype(str)
    out = out.where(pd.notna(out), None)
    records = out.to_dict(orient="records")
    for row in records:
        for key, value in list(row.items()):
            if hasattr(value, "item"):
                row[key] = value.item()
    return records


def dataframe_values(df: pd.DataFrame) -> List[List[object]]:
    if df.empty:
        return []
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype(str)
    out = out.where(pd.notna(out), None)
    return json_ready(out.values.tolist())


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_ready(value.item())
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def ordered_unique(values: Sequence[object]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value) != ""))


EDGE_CONTEXT_KEY_COLUMNS = [
    "data_scope",
    "token_profile",
    "network_window",
    "center_mode",
    "period_set_id",
    "period_id",
    "region_norm",
    "alter_token",
]

EDGE_CONTEXT_COLUMNS = [
    *EDGE_CONTEXT_KEY_COLUMNS,
    "edge_weight",
    "neighbor_rank",
    "context_rank",
    "available_context_count",
    "available_article_count",
    "available_match_event_count",
    "context_uid",
    "article_uid",
    "article_id",
    "date",
    "issue_page",
    "publish_variant",
    "source_labels",
    "context_index",
    "context_start",
    "context_end",
    "context_char_len",
    "focus_tokens",
    "alter_token_mentions",
    "match_event_count",
    "min_token_distance",
    "context_text",
]


def edge_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in EDGE_CONTEXT_KEY_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out["network_window"] = out["network_window"].astype(str)
    return out[EDGE_CONTEXT_KEY_COLUMNS].drop_duplicates()


def load_edge_context_payload(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    if args.public_release:
        return {"columns": [], "rows": [], "source_csv": "", "full_rows": 0, "shown_rows": 0}
    path_text = str(args.edge_context_csv or "").strip()
    if not path_text:
        return {"columns": [], "rows": [], "source_csv": "", "full_rows": 0, "shown_rows": 0}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        log(f"WARNING: edge context CSV not found: {path}")
        return {"columns": [], "rows": [], "source_csv": str(path), "full_rows": 0, "shown_rows": 0}
    contexts = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if contexts.empty:
        return {"columns": EDGE_CONTEXT_COLUMNS, "rows": [], "source_csv": str(path), "full_rows": 0, "shown_rows": 0}
    for column in EDGE_CONTEXT_COLUMNS:
        if column not in contexts.columns:
            contexts[column] = ""
    contexts["token_profile"] = contexts["token_profile"].map(normalize_profile)
    contexts["network_window"] = pd.to_numeric(contexts["network_window"], errors="coerce").astype("Int64").astype(str)
    contexts["network_window"] = contexts["network_window"].replace({"<NA>": ""})
    contexts["context_rank"] = pd.to_numeric(contexts["context_rank"], errors="coerce").fillna(0).astype(int)
    full_rows = len(contexts)
    contexts = contexts.merge(edge_key_frame(df), on=EDGE_CONTEXT_KEY_COLUMNS, how="inner")
    limit = int(args.max_edge_contexts_per_edge_payload)
    if limit > 0:
        contexts = contexts[contexts["context_rank"].le(limit)].copy()
    contexts = contexts[EDGE_CONTEXT_COLUMNS].sort_values(
        EDGE_CONTEXT_KEY_COLUMNS + ["context_rank"],
        kind="mergesort",
    )
    contexts = contexts.where(pd.notna(contexts), None)
    return {
        "columns": EDGE_CONTEXT_COLUMNS,
        "rows": contexts.values.tolist(),
        "source_csv": str(path),
        "full_rows": full_rows,
        "shown_rows": len(contexts),
    }


def build_payload(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    regions = selected_regions(args, df)
    center_modes = ordered_unique(df["center_mode"])
    default_center_mode = "grouped_focus" if "grouped_focus" in center_modes else (center_modes[0] if center_modes else "grouped_focus")
    ui_mode = str(args.ui_mode).strip().lower()
    is_core_ui = ui_mode == "core"
    period_cols = [
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
    ]
    periods = (
        df[period_cols]
        .drop_duplicates()
        .sort_values(["period_set_id", "period_sort_order", "period_id"], kind="mergesort")
    )
    region_meta = (
        df[["region_norm", "macro_region"]]
        .drop_duplicates()
        .sort_values(["region_norm", "macro_region"], kind="mergesort")
    )
    topn_choices = list(range(1, min(20, args.max_neighbors_per_region) + 1))
    topn_choices.extend(value for value in [30, 50, 100] if value <= args.max_neighbors_per_region)
    if args.default_topn_neighbors not in topn_choices and args.default_topn_neighbors <= args.max_neighbors_per_region:
        topn_choices.append(args.default_topn_neighbors)
    topn_choices = sorted(set(topn_choices))
    available_pos_groups = [group for group in POS_GROUP_ORDER if group in set(df.get("pos_group", pd.Series(dtype=str)).astype(str))]
    if not available_pos_groups:
        available_pos_groups = POS_GROUP_ORDER[:]
    edge_context_payload = load_edge_context_payload(df, args)
    input_path = Path(args.input_csv).expanduser().resolve()
    return {
        "meta": {
            "title": args.title,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_csv": input_path.name if args.public_release else str(input_path),
            "edge_context_csv": edge_context_payload["source_csv"],
            "edge_context_full_rows": edge_context_payload["full_rows"],
            "edge_context_shown_rows": edge_context_payload["shown_rows"],
            "public_release": bool(args.public_release),
            "default_selected_regions": regions,
            "default_center_mode": default_center_mode,
            "center_mode_labels": {
                "grouped_focus": "Grouped core + derived",
                "exact_core_only": "Exact core only",
            },
            "major_regions": [region for region in MAJOR_REGIONS if region in set(regions)],
            "default_topn_neighbors": min(args.default_topn_neighbors, args.max_neighbors_per_region),
            "max_neighbors_per_region": args.max_neighbors_per_region,
            "topn_choices": topn_choices,
            "ambiguous_pos_threshold": max(0.0, min(1.0, float(args.ambiguous_pos_threshold))),
            "pos_group_labels": POS_GROUP_LABELS,
            "pos_group_order": POS_GROUP_ORDER,
            "default_selected_pos_groups": [
                group for group in DEFAULT_POS_GROUPS if group in set(available_pos_groups)
            ],
            "default_include_region_reference_neighbors": bool(
                args.default_include_region_reference_neighbors
            ),
            "ui": {
                "mode": ui_mode,
                "hide_data_scope": bool(args.hide_data_scope),
                "entity_singular": "core" if is_core_ui else "region",
                "entity_plural": "cores" if is_core_ui else "regions",
                "entity_singular_title": "Core" if is_core_ui else "Region",
                "entity_plural_title": "Cores" if is_core_ui else "Regions",
                "checkbox_label": "Include core keyword tokens" if is_core_ui else "Include region-reference tokens",
                "search_placeholder": "core or token" if is_core_ui else "region or token",
            },
        },
        "options": {
            "data_scopes": ordered_unique(df["data_scope"]),
            "profiles": ordered_unique(df["token_profile"]),
            "windows": ordered_unique(df["network_window"]),
            "center_modes": center_modes,
            "period_sets": ordered_unique(df["period_set_id"]),
            "periods": dataframe_records(periods),
            "regions": dataframe_records(region_meta),
            "pos_groups": available_pos_groups,
        },
        "row_columns": list(df.columns),
        "rows": dataframe_values(df),
        "edge_context_columns": edge_context_payload["columns"],
        "edge_context_rows": json_ready(edge_context_payload["rows"]),
    }


def html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_template(title: str, payload_json: str) -> str:
    safe_title = html_escape(title)
    payload = json.loads(payload_json)
    ui = payload.get("meta", {}).get("ui", {})
    hide_data_scope = bool(ui.get("hide_data_scope"))
    entity_singular = html_escape(ui.get("entity_singular", "region"))
    entity_plural = html_escape(ui.get("entity_plural", "regions"))
    checkbox_label = html_escape(ui.get("checkbox_label", "Include region-reference tokens"))
    search_placeholder = html_escape(ui.get("search_placeholder", "region or token"))
    public_release = bool(payload.get("meta", {}).get("public_release"))
    graph_hint = (
        "Mouse wheel zooms the graph. Drag the background to pan. Click a node or edge to highlight its connections. "
        "Edge-level context excerpts are omitted from the public edition."
        if public_release
        else "Mouse wheel zooms the graph. Drag the background to pan. Click a node to highlight its incident edges; click an edge to show keyword-centered contexts."
    )
    data_scope_control = ""
    if not hide_data_scope:
        data_scope_control = '      <label>Data scope<select id="dataScope"></select></label>\n'
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65727e;
      --line: #d7ded8;
      --accent: #0f6c63;
      --soft: #e8f0ed;
      --warn: #ad5d20;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Malgun Gothic", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px 12px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 22px;
      font-weight: 700;
    }
    main {
      padding: 16px 18px 24px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      align-items: end;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    select, input[type="number"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 8px;
      font: inherit;
    }
    .checkbox-line {
      flex-direction: row;
      align-items: center;
      gap: 7px;
      min-height: 35px;
    }
    .region-toolbar, .pos-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 0;
      align-items: center;
    }
    .pos-toolbar {
      margin-top: 10px;
    }
    .region-list, .pos-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .region-list label, .pos-list label {
      flex-direction: row;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 5px 8px;
      color: var(--ink);
      font-weight: 500;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 9px;
      cursor: pointer;
      font: inherit;
    }
    button:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 3.6fr) minmax(280px, 0.6fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 17px;
    }
    .graph-wrap {
      width: 100%;
      height: 700px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf8;
      overflow: hidden;
      position: relative;
    }
    .graph-toolbar {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 10px;
    }
    .zoom-controls {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .zoom-controls button {
      min-width: 34px;
      height: 34px;
      padding: 0 8px;
    }
    svg {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      touch-action: none;
    }
    svg.panning {
      cursor: grabbing;
    }
    .edge {
      stroke-opacity: 0.48;
      cursor: pointer;
    }
    .edge.search-hit {
      stroke-opacity: 0.88;
    }
    .edge.selected-edge {
      stroke-opacity: 1;
      stroke-width: 8px;
      filter: drop-shadow(0 0 3px rgba(17, 24, 39, 0.35));
    }
    .edge.dim, .node.dim, .node-label.dim {
      opacity: 0.16;
    }
    .region-node {
      stroke: #263238;
      stroke-width: 1.2;
    }
    .token-node {
      stroke: #ffffff;
      stroke-width: 1.2;
    }
    .node {
      cursor: pointer;
    }
    .node-label {
      pointer-events: none;
      paint-order: stroke;
      stroke: #fbfbf8;
      stroke-width: 3px;
      stroke-linejoin: round;
      fill: #1f2933;
      font-weight: 700;
      dominant-baseline: middle;
    }
    .region-label {
      fill: #ffffff;
      stroke: rgba(0, 0, 0, 0.38);
      stroke-width: 3px;
    }
    .node.search-hit .node-shape {
      stroke: #f2b705;
      stroke-width: 4px;
    }
    .token-node.ambiguous-pos {
      stroke: #111827;
      stroke-width: 2.6px;
      stroke-dasharray: 4 2;
    }
    .node-label.search-hit {
      fill: #111827;
      stroke: #f8d463;
      stroke-width: 4px;
    }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfbf8;
    }
    .metric .value {
      font-size: 20px;
      font-weight: 700;
      color: var(--accent);
    }
    .metric .label {
      font-size: 12px;
      color: var(--muted);
    }
    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #fafbf9;
      color: var(--muted);
      font-size: 12px;
    }
    th.sortable {
      cursor: pointer;
      user-select: none;
    }
    th.sortable:hover {
      color: var(--accent);
    }
    .sort-indicator {
      color: var(--accent);
      margin-left: 4px;
    }
    .table-wrap {
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-top: 10px;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 12px;
      color: var(--muted);
      background: #fbfbf8;
    }
    .context-list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .context-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
    }
    .context-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .context-text {
      font-size: 13px;
      line-height: 1.65;
      word-break: break-all;
    }
    mark.focus-mark, mark.neighbor-mark {
      padding: 0 2px;
      border-radius: 2px;
    }
    mark.focus-mark {
      background: #fef08a;
    }
    mark.neighbor-mark {
      background: #bfdbfe;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }
    .legend-shape {
      width: 14px;
      height: 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    @media (max-width: 1180px) {
      .controls { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      header { position: static; }
      main { padding: 12px; }
      .controls { grid-template-columns: 1fr; }
      .graph-toolbar { grid-template-columns: 1fr; }
      .graph-wrap { height: 580px; }
      .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="controls">
__DATA_SCOPE_CONTROL__      <label>Profile<select id="profile"></select></label>
      <label>Window<select id="window"></select></label>
      <label id="centerModeWrap">Center mode<select id="centerMode"></select></label>
      <label>Period set<select id="periodSet"></select></label>
      <label>Period<select id="period"></select></label>
      <label>Top N per __ENTITY_SINGULAR__<select id="topN"></select></label>
      <label>Minimum shared __ENTITY_PLURAL__<input id="minShared" type="number" min="1" step="1" value="1"></label>
      <label>Maximum shared __ENTITY_PLURAL__<input id="maxShared" type="number" min="1" step="1" placeholder="All"></label>
    </div>
    <label class="checkbox-line"><input id="includeRegionRefs" type="checkbox"> __CHECKBOX_LABEL__</label>
    <div class="region-toolbar">
      <button id="selectMajor" type="button">Select major</button>
      <button id="selectAll" type="button">Select all available</button>
      <button id="clearRegions" type="button">Clear</button>
      <span class="small" id="selectionNote"></span>
    </div>
    <div class="region-list" id="regions"></div>
    <div class="pos-toolbar">
      <span class="small">POS filters</span>
      <button id="selectDefaultPos" type="button">Default POS</button>
      <button id="selectAllPos" type="button">Select all POS</button>
      <button id="clearPos" type="button">Clear POS</button>
      <label class="checkbox-line"><input id="markAmbiguousPos" type="checkbox"> Mark ambiguous dominant POS</label>
      <label class="checkbox-line"><input id="shapeByPos" type="checkbox"> Shape by POS group</label>
    </div>
    <div class="pos-list" id="posGroups"></div>
  </header>
  <main>
    <section class="grid">
      <div class="panel">
        <h2>Multi-core ego network</h2>
        <div class="graph-toolbar">
          <label>Search node<input id="nodeSearch" type="text" placeholder="__SEARCH_PLACEHOLDER__"></label>
          <div class="zoom-controls">
            <button id="zoomOut" type="button">-</button>
            <button id="zoomReset" type="button">Reset</button>
            <button id="zoomIn" type="button">+</button>
            <button id="captureGraph" type="button">Capture</button>
          </div>
        </div>
        <div class="graph-wrap"><svg id="graph"></svg></div>
        <div class="legend" id="legend"></div>
        <div class="hint">__GRAPH_HINT__</div>
      </div>
      <div class="panel">
        <h2>Graph summary</h2>
        <div class="stats" id="stats"></div>
        <div id="sharedTokens"></div>
        <div id="edgeTable"></div>
        <div id="edgeContexts"></div>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    function inflateRows(columns, rows) {
      return (rows || []).map((values) => {
        const row = {};
        columns.forEach((column, index) => {
          row[column] = values[index];
        });
        return row;
      });
    }
    DATA.rows = inflateRows(DATA.row_columns || [], DATA.rows || []);
    DATA.edge_context_rows = inflateRows(DATA.edge_context_columns || [], DATA.edge_context_rows || []);
    const UI = DATA.meta.ui || {
      entity_singular: "region",
      entity_plural: "regions",
      entity_singular_title: "Region",
      entity_plural_title: "Regions",
      checkbox_label: "Include region-reference tokens",
      search_placeholder: "region or token",
    };
    const byId = (id) => document.getElementById(id);
    const fields = ["dataScope", "profile", "window", "periodSet", "centerMode", "period"];
    const CORE_REGION_COLORS = {
      "立憲": "#00a651",
      "憲政": "#0057ff",
      "憲法": "#ef0000",
      "制憲": "#c79a00",
    };
    const fallbackRegionColors = [
      "#2364aa", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e",
      "#e6ab02", "#a6761d", "#0f6c63", "#8d6e63", "#5c6bc0", "#00897b"
    ];
    const state = {
      selectedRegions: DATA.meta.default_selected_regions.slice(),
      selectedPosGroups: (DATA.meta.default_selected_pos_groups || []).slice(),
      highlightedNode: "",
      highlightedEdge: "",
      nodeSearch: "",
      zoom: {scale: 1, x: 0, y: 0},
      pan: {active: false, moved: false, startX: 0, startY: 0, originX: 0, originY: 0},
    };
    const tableSortState = {};
    const EDGE_KEY_SEPARATOR = "\u001f";
    function edgeContextKeyFromParts(dataScope, tokenProfile, networkWindow, centerMode, periodSetId, periodId, regionNorm, alterToken) {
      return [dataScope, tokenProfile, networkWindow, centerMode, periodSetId, periodId, regionNorm, alterToken]
        .map((value) => String(value ?? ""))
        .join(EDGE_KEY_SEPARATOR);
    }
    function edgeContextKeyFromRow(row) {
      return edgeContextKeyFromParts(
        row.data_scope,
        row.token_profile,
        row.network_window,
        row.center_mode,
        row.period_set_id,
        row.period_id,
        row.region_norm,
        row.alter_token
      );
    }
    function edgeContextKeyFromEdge(edge) {
      return edgeContextKeyFromParts(
        edge.dataScope,
        edge.tokenProfile,
        edge.networkWindow,
        edge.centerMode,
        edge.periodSetId,
        edge.periodId,
        edge.region,
        edge.alterToken
      );
    }
    const EDGE_CONTEXT_LOOKUP = new Map();
    (DATA.edge_context_rows || []).forEach((row) => {
      const key = edgeContextKeyFromRow(row);
      const rows = EDGE_CONTEXT_LOOKUP.get(key) || [];
      rows.push(row);
      EDGE_CONTEXT_LOOKUP.set(key, rows);
    });
    function uniq(values) {
      return Array.from(new Set(values.filter((value) => value !== null && value !== undefined && String(value) !== "")));
    }
    function centerModeLabel(value) {
      return DATA.meta.center_mode_labels?.[value] || value;
    }
    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
    function fmt(value, digits = 3) {
      const num = Number(value);
      if (!Number.isFinite(num)) return String(value ?? "");
      if (Math.abs(num) >= 1000) return Math.round(num).toLocaleString();
      return num.toLocaleString(undefined, { maximumFractionDigits: digits });
    }
    function formatDateLabel(value) {
      const text = String(value ?? "").trim();
      if (!text) return "";
      const compact = text.match(/^(\\d{4})(\\d{2})(\\d{2})$/);
      if (compact) return `${compact[1]}.${compact[2]}.${compact[3]}`;
      const iso = text.match(/^(\\d{4})-(\\d{2})-(\\d{2})/);
      if (iso) return `${iso[1]}.${iso[2]}.${iso[3]}`;
      return text;
    }
    function formatPeriodLabel(row) {
      const start = formatDateLabel(row.period_start_date);
      const end = formatDateLabel(row.period_end_date);
      if (start && end) return `${start} - ${end}`;
      return row.period_label || row.period_id;
    }
    function setOptions(id, values, preferred, labelFn = null) {
      const el = byId(id);
      const keys = values.map((value) => value.value);
      if (!el) {
        state[id] = keys.includes(preferred) ? preferred : (keys.length ? keys[0] : "");
        return;
      }
      el.innerHTML = "";
      values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value.value;
        option.textContent = labelFn ? labelFn(value) : value.label;
        el.appendChild(option);
      });
      if (keys.includes(preferred)) el.value = preferred;
      else if (keys.length) el.value = keys[0];
      state[id] = el.value;
    }
    function updateStateFromControls() {
      fields.forEach((id) => {
        const el = byId(id);
        if (el && el.options.length) state[id] = el.value;
      });
      state.topN = Math.max(1, Number(byId("topN").value || DATA.meta.default_topn_neighbors));
      state.minShared = Math.max(1, Number(byId("minShared").value || 1));
      const maxSharedText = byId("maxShared").value.trim();
      const maxSharedValue = Number(maxSharedText);
      state.maxShared = maxSharedText && Number.isFinite(maxSharedValue) ? Math.max(1, maxSharedValue) : Infinity;
      state.includeRegionRefs = byId("includeRegionRefs").checked;
      state.markAmbiguousPos = byId("markAmbiguousPos").checked;
      state.shapeByPos = byId("shapeByPos").checked;
      state.nodeSearch = byId("nodeSearch").value.trim().toLowerCase();
    }
    function rowsForCurrentCore() {
      return DATA.rows.filter((row) =>
        row.data_scope === state.dataScope &&
        row.token_profile === state.profile &&
        String(row.network_window) === String(state.window) &&
        row.period_set_id === state.periodSet &&
        String(row.center_mode || "grouped_focus") === String(state.centerMode || DATA.meta.default_center_mode || "grouped_focus")
      );
    }
    function rowsForCurrentPeriod() {
      return rowsForCurrentCore().filter((row) => row.period_id === state.period);
    }
    function updateOptions() {
      updateStateFromControls();
      setOptions("dataScope", DATA.options.data_scopes.map((value) => ({value, label: value})), state.dataScope || "body");
      const profileOrder = ["regex-only", "strict", "full"];
      const profiles = uniq(DATA.rows.filter((row) => row.data_scope === state.dataScope).map((row) => row.token_profile))
        .sort((a, b) => {
          const ai = profileOrder.indexOf(a);
          const bi = profileOrder.indexOf(b);
          return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a.localeCompare(b);
        });
      setOptions("profile", profiles.map((value) => ({value, label: value})), state.profile || "strict");
      const windows = uniq(DATA.rows.filter((row) => row.data_scope === state.dataScope && row.token_profile === state.profile).map((row) => String(row.network_window)))
        .sort((a, b) => Number(a) - Number(b));
      setOptions("window", windows.map((value) => ({value, label: value})), state.window || "10");
      const windowRows = DATA.rows
        .filter((row) => row.data_scope === state.dataScope && row.token_profile === state.profile && String(row.network_window) === String(state.window))
      const periodSets = uniq(windowRows.map((row) => row.period_set_id));
      setOptions("periodSet", periodSets.map((value) => ({value, label: value})), state.periodSet || "global");
      const periodSetRows = windowRows.filter((row) => row.period_set_id === state.periodSet);
      const centerModes = uniq(periodSetRows.map((row) => String(row.center_mode || "grouped_focus")));
      setOptions(
        "centerMode",
        centerModes.map((value) => ({value, label: centerModeLabel(value)})),
        state.centerMode || DATA.meta.default_center_mode || "grouped_focus"
      );
      byId("centerModeWrap").style.display = centerModes.length > 1 ? "" : "none";
      const periodRows = DATA.options.periods
        .filter((row) => row.period_set_id === state.periodSet)
        .filter((period) => DATA.rows.some((row) =>
          row.data_scope === state.dataScope &&
          row.token_profile === state.profile &&
          String(row.network_window) === String(state.window) &&
          row.period_set_id === state.periodSet &&
          String(row.center_mode || "grouped_focus") === String(state.centerMode) &&
          row.period_id === period.period_id
        ))
        .sort((a, b) => Number(a.period_sort_order || 0) - Number(b.period_sort_order || 0) || String(a.period_id).localeCompare(String(b.period_id)));
      setOptions("period", periodRows.map((row) => ({value: row.period_id, label: formatPeriodLabel(row)})), state.period || "global");
      renderRegionControls();
      renderPosControls();
    }
    function renderTopNOptions() {
      const topN = byId("topN");
      const choices = (DATA.meta.topn_choices || []).length
        ? DATA.meta.topn_choices
        : Array.from({length: Number(DATA.meta.max_neighbors_per_region || 20)}, (_, idx) => idx + 1);
      const current = String(topN.value || DATA.meta.default_topn_neighbors);
      topN.innerHTML = choices.map((value) => `<option value="${esc(value)}">Top ${esc(value)}</option>`).join("");
      if (choices.map(String).includes(current)) topN.value = current;
      else topN.value = String(DATA.meta.default_topn_neighbors);
    }
    function availableRegions() {
      return uniq(rowsForCurrentPeriod().map((row) => row.region_norm));
    }
    function renderRegionControls() {
      const regions = availableRegions();
      const selected = new Set(state.selectedRegions.filter((region) => regions.includes(region)));
      if (!selected.size) {
        DATA.meta.default_selected_regions.forEach((region) => {
          if (regions.includes(region)) selected.add(region);
        });
      }
      state.selectedRegions = Array.from(selected);
      byId("regions").innerHTML = regions.map((region) => {
        const checked = selected.has(region) ? " checked" : "";
        return `<label><input type="checkbox" data-region="${esc(region)}"${checked}>${esc(region)}</label>`;
      }).join("");
      byId("selectionNote").textContent = `${state.selectedRegions.length} selected / ${regions.length} available`;
      byId("regions").querySelectorAll("input[data-region]").forEach((input) => {
        input.addEventListener("change", () => {
          const region = input.dataset.region;
          const current = new Set(state.selectedRegions);
          if (input.checked) current.add(region);
          else current.delete(region);
          state.selectedRegions = Array.from(current);
          state.highlightedNode = "";
          renderAll(false);
        });
      });
    }
    function availablePosGroups() {
      const groups = uniq(rowsForCurrentPeriod().map((row) => row.pos_group || "missing"));
      const order = DATA.meta.pos_group_order || [];
      return groups.sort((a, b) => {
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a.localeCompare(b);
      });
    }
    function posGroupLabel(group) {
      return DATA.meta.pos_group_labels?.[group] || group;
    }
    function regionPosMeta(region) {
      const row = rowsForCurrentPeriod().find((item) => item.region_norm === region && item.region_dominant_pos)
        || DATA.rows.find((item) =>
          item.data_scope === state.dataScope &&
          item.token_profile === state.profile &&
          String(item.network_window) === String(state.window) &&
          String(item.center_mode || "grouped_focus") === String(state.centerMode || DATA.meta.default_center_mode || "grouped_focus") &&
          item.region_norm === region &&
          item.region_dominant_pos
        )
        || {};
      return {
        dominantPos: row.region_dominant_pos || "(missing)",
        dominantPosShare: Number(row.region_dominant_pos_share || 0),
        posGroup: row.region_pos_group || "missing",
        posGroupLabel: row.region_pos_group_label || posGroupLabel(row.region_pos_group || "missing"),
        posAmbiguous: Boolean(row.region_pos_is_ambiguous),
      };
    }
    function renderPosControls(allowDefault = true) {
      const groups = availablePosGroups();
      const selected = new Set(state.selectedPosGroups.filter((group) => groups.includes(group)));
      if (!selected.size && allowDefault) {
        (DATA.meta.default_selected_pos_groups || groups).forEach((group) => {
          if (groups.includes(group)) selected.add(group);
        });
      }
      state.selectedPosGroups = Array.from(selected);
      byId("posGroups").innerHTML = groups.map((group) => {
        const checked = selected.has(group) ? " checked" : "";
        return `<label title="${esc(group)}"><input type="checkbox" data-pos-group="${esc(group)}"${checked}>${esc(posGroupLabel(group))}</label>`;
      }).join("");
      byId("posGroups").querySelectorAll("input[data-pos-group]").forEach((input) => {
        input.addEventListener("change", () => {
          const group = input.dataset.posGroup;
          const current = new Set(state.selectedPosGroups);
          if (input.checked) current.add(group);
          else current.delete(group);
          state.selectedPosGroups = Array.from(current);
          state.highlightedNode = "";
          renderAll(false);
        });
      });
    }
    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }
    function hexToRgb(hex) {
      const value = String(hex || "").replace("#", "");
      if (value.length !== 6) return {r: 127, g: 127, b: 127};
      return {
        r: parseInt(value.slice(0, 2), 16),
        g: parseInt(value.slice(2, 4), 16),
        b: parseInt(value.slice(4, 6), 16),
      };
    }
    function rgbToHex(rgb) {
      const parts = [rgb.r, rgb.g, rgb.b].map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0"));
      return `#${parts.join("")}`;
    }
    function mixRegionColors(regions, lighten = 0) {
      const valid = (regions || []).filter((region) => region);
      if (!valid.length) return "#8b97a3";
      const colors = valid.map((region) => hexToRgb(regionColor(region)));
      const mixed = colors.reduce((acc, color) => ({
        r: acc.r + color.r,
        g: acc.g + color.g,
        b: acc.b + color.b,
      }), {r: 0, g: 0, b: 0});
      const mean = {
        r: mixed.r / colors.length,
        g: mixed.g / colors.length,
        b: mixed.b / colors.length,
      };
      const ratio = clamp(lighten, 0, 1);
      return rgbToHex({
        r: mean.r + (255 - mean.r) * ratio,
        g: mean.g + (255 - mean.g) * ratio,
        b: mean.b + (255 - mean.b) * ratio,
      });
    }
    function regionColor(region) {
      if (CORE_REGION_COLORS[region]) return CORE_REGION_COLORS[region];
      const regions = availableRegions();
      const idx = Math.max(0, regions.indexOf(region));
      return fallbackRegionColors[idx % fallbackRegionColors.length];
    }
    function selectedRows() {
      const selected = new Set(state.selectedRegions);
      const byRegion = new Map();
      rowsForCurrentPeriod()
        .filter((row) => selected.has(row.region_norm))
        .filter((row) => state.includeRegionRefs || !row.alter_is_region_reference)
        .sort((a, b) => Number(a.neighbor_rank || 0) - Number(b.neighbor_rank || 0) || Number(b.edge_weight || 0) - Number(a.edge_weight || 0))
        .forEach((row) => {
          const rows = byRegion.get(row.region_norm) || [];
          if (rows.length < state.topN) rows.push(row);
          byRegion.set(row.region_norm, rows);
        });
      const rows = Array.from(byRegion.values()).flat();
      const selectedPos = new Set(state.selectedPosGroups || []);
      const posFiltered = rows.filter((row) => selectedPos.has(row.pos_group || "missing"));
      if (!posFiltered.length) return [];
      if (state.minShared <= 1 && !Number.isFinite(state.maxShared)) return posFiltered;
      const tokenRegions = new Map();
      posFiltered.forEach((row) => {
        const set = tokenRegions.get(row.alter_token) || new Set();
        set.add(row.region_norm);
        tokenRegions.set(row.alter_token, set);
      });
      return posFiltered.filter((row) => {
        const sharedCount = (tokenRegions.get(row.alter_token) || new Set()).size;
        return sharedCount >= state.minShared && sharedCount <= state.maxShared;
      });
    }
    function buildGraph(rows) {
      const nodes = new Map();
      const edges = [];
      const tokenRegions = new Map();
      const tokenWeights = new Map();
      state.selectedRegions.forEach((region) => {
        const posMeta = regionPosMeta(region);
        nodes.set(`region::${region}`, {
          id: `region::${region}`,
          label: region,
          type: "region",
          region,
          color: regionColor(region),
          weight: 1,
          shared: 1,
          dominantPos: posMeta.dominantPos,
          dominantPosShare: posMeta.dominantPosShare,
          posGroup: posMeta.posGroup,
          posGroupLabel: posMeta.posGroupLabel,
          posAmbiguous: posMeta.posAmbiguous,
        });
      });
      rows.forEach((row) => {
        const regionId = `region::${row.region_norm}`;
        const tokenId = `token::${row.alter_token}`;
        if (!nodes.has(regionId)) {
          const posMeta = regionPosMeta(row.region_norm);
          nodes.set(regionId, {
            id: regionId,
            label: row.region_norm,
            type: "region",
            region: row.region_norm,
            color: regionColor(row.region_norm),
            weight: 1,
            shared: 1,
            dominantPos: posMeta.dominantPos,
            dominantPosShare: posMeta.dominantPosShare,
            posGroup: posMeta.posGroup,
            posGroupLabel: posMeta.posGroupLabel,
            posAmbiguous: posMeta.posAmbiguous,
          });
        }
        const regions = tokenRegions.get(tokenId) || new Set();
        regions.add(row.region_norm);
        tokenRegions.set(tokenId, regions);
        tokenWeights.set(tokenId, Number(tokenWeights.get(tokenId) || 0) + Number(row.edge_weight || 0));
        if (!nodes.has(tokenId)) {
          nodes.set(tokenId, {
            id: tokenId,
            label: row.alter_token,
            type: "token",
            region: "",
            color: "#7b8794",
            weight: 0,
            shared: 1,
            isRegionRef: row.alter_is_region_reference,
            dominantPos: row.dominant_pos || "(missing)",
            dominantPosShare: Number(row.dominant_pos_share || 0),
            posGroup: row.pos_group || "missing",
            posGroupLabel: row.pos_group_label || posGroupLabel(row.pos_group || "missing"),
            posAmbiguous: Boolean(row.pos_is_ambiguous),
          });
        } else {
          const node = nodes.get(tokenId);
          node.posAmbiguous = Boolean(node.posAmbiguous || row.pos_is_ambiguous);
        }
        edges.push({
          id: `${regionId}--${tokenId}`,
          source: regionId,
          target: tokenId,
          dataScope: row.data_scope,
          tokenProfile: row.token_profile,
          networkWindow: row.network_window,
          centerMode: row.center_mode,
          periodSetId: row.period_set_id,
          periodId: row.period_id,
          region: row.region_norm,
          alterToken: row.alter_token,
          color: regionColor(row.region_norm),
          weight: Number(row.edge_weight || 0),
          pmi: Number(row.pmi_mean || 0),
          rank: Number(row.neighbor_rank || 0),
        });
      });
      tokenRegions.forEach((regions, tokenId) => {
        const node = nodes.get(tokenId);
        node.shared = regions.size;
        node.regions = Array.from(regions);
        node.weight = tokenWeights.get(tokenId) || 0;
      });
      return {nodes: Array.from(nodes.values()), edges};
    }
    function hashText(text) {
      let hash = 0;
      for (let i = 0; i < text.length; i += 1) hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
      return Math.abs(hash);
    }
    function layoutGraph(graph, width, height) {
      const cx = width / 2;
      const cy = height / 2;
      const selectedRegions = graph.nodes.filter((node) => node.type === "region");
      const regionRadius = Math.min(width, height) * 0.32;
      const horizontalSpread = selectedRegions.length >= 3 ? 1.18 : 1.06;
      if (selectedRegions.length === 1) {
        selectedRegions[0].x = cx;
        selectedRegions[0].y = cy - regionRadius;
        selectedRegions[0].fixed = true;
      } else if (selectedRegions.length === 2) {
        const offset = Math.min(width * 0.34, Math.max(170, regionRadius * horizontalSpread));
        selectedRegions[0].x = cx - offset;
        selectedRegions[0].y = cy;
        selectedRegions[0].fixed = true;
        selectedRegions[1].x = cx + offset;
        selectedRegions[1].y = cy;
        selectedRegions[1].fixed = true;
      } else {
        selectedRegions.forEach((node, idx) => {
          const angle = -Math.PI / 2 + (Math.PI * 2 * idx / Math.max(1, selectedRegions.length));
          node.x = cx + Math.cos(angle) * regionRadius * horizontalSpread;
          node.y = cy + Math.sin(angle) * regionRadius;
          node.fixed = true;
        });
      }
      const regionNodeMap = new Map(selectedRegions.map((node) => [node.region, node]));
      graph.nodes.filter((node) => node.type === "token").forEach((node) => {
        const hash = hashText(node.id);
        const regions = (node.regions || []).map((region) => regionNodeMap.get(region)).filter((item) => item);
        let baseX = cx;
        let baseY = cy;
        if (regions.length) {
          const avgX = regions.reduce((sum, item) => sum + item.x, 0) / regions.length;
          const avgY = regions.reduce((sum, item) => sum + item.y, 0) / regions.length;
          if (regions.length === 1) {
            const dx = avgX - cx;
            const dy = avgY - cy;
            const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
            const clusterRadius = regionRadius * 0.66;
            baseX = cx + dx / distance * clusterRadius;
            baseY = cy + dy / distance * clusterRadius;
          } else {
            const centerPull = regions.length === 2 ? 0.24 : 0.34;
            baseX = avgX * (1 - centerPull) + cx * centerPull;
            baseY = avgY * (1 - centerPull) + cy * centerPull;
          }
        }
        const angle = (hash % 360) / 360 * Math.PI * 2;
        const radius = 18 + (hash % 48);
        node.x = baseX + Math.cos(angle) * radius;
        node.y = baseY + Math.sin(angle) * radius;
        node.anchorX = baseX;
        node.anchorY = baseY;
        node.anchorStrength = regions.length === 1 ? 0.022 : regions.length === 2 ? 0.012 : 0.007;
        node.vx = 0;
        node.vy = 0;
      });
      const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
      for (let tick = 0; tick < 190; tick += 1) {
        graph.edges.forEach((edge) => {
          const source = nodeMap.get(edge.source);
          const target = nodeMap.get(edge.target);
          if (!source || !target) return;
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
          const desired = 105 + Math.max(0, 4 - Math.log1p(edge.weight)) * 8;
          const strength = 0.018;
          const force = (distance - desired) * strength;
          if (!target.fixed) {
            target.vx -= dx / distance * force;
            target.vy -= dy / distance * force;
          }
          if (!source.fixed) {
            source.vx += dx / distance * force;
            source.vy += dy / distance * force;
          }
        });
        for (let i = 0; i < graph.nodes.length; i += 1) {
          for (let j = i + 1; j < graph.nodes.length; j += 1) {
            const a = graph.nodes[i];
            const b = graph.nodes[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist2 = Math.max(16, dx * dx + dy * dy);
            const dist = Math.sqrt(dist2);
            const force = (a.type === "region" || b.type === "region" ? 2600 : 1250) / dist2;
            if (!a.fixed) {
              a.vx -= dx / dist * force;
              a.vy -= dy / dist * force;
            }
            if (!b.fixed) {
              b.vx += dx / dist * force;
              b.vy += dy / dist * force;
            }
            const minDistance = collisionRadius(a) + collisionRadius(b) + (a.type === "region" || b.type === "region" ? 10 : 4);
            if (dist < minDistance) {
              const push = (minDistance - dist) * 0.045;
              if (!a.fixed) {
                a.vx -= dx / dist * push;
                a.vy -= dy / dist * push;
              }
              if (!b.fixed) {
                b.vx += dx / dist * push;
                b.vy += dy / dist * push;
              }
            }
          }
        }
        graph.nodes.forEach((node) => {
          if (node.fixed) return;
          node.vx += (Number(node.anchorX || cx) - node.x) * Number(node.anchorStrength || 0.004);
          node.vy += (Number(node.anchorY || cy) - node.y) * Number(node.anchorStrength || 0.004);
          if (Number(node.shared || 1) > 1) {
            node.vx += (cx - node.x) * 0.001;
            node.vy += (cy - node.y) * 0.001;
          }
          if (Number(node.shared || 1) === 1) {
            const anchorDx = Number(node.anchorX || cx) - cx;
            const anchorDy = Number(node.anchorY || cy) - cy;
            const nodeDx = node.x - cx;
            const nodeDy = node.y - cy;
            if (anchorDx * nodeDx + anchorDy * nodeDy < 0) {
              const distance = Math.max(1, Math.sqrt(anchorDx * anchorDx + anchorDy * anchorDy));
              node.vx += anchorDx / distance * 0.7;
              node.vy += anchorDy / distance * 0.7;
            }
          }
          node.vx *= 0.82;
          node.vy *= 0.82;
          const margin = nodeRadius(node) + 8;
          node.x = Math.max(margin, Math.min(width - margin, node.x + node.vx));
          node.y = Math.max(margin, Math.min(height - margin, node.y + node.vy));
        });
      }
      for (let settle = 0; settle < 35; settle += 1) {
        for (let i = 0; i < graph.nodes.length; i += 1) {
          for (let j = i + 1; j < graph.nodes.length; j += 1) {
            const a = graph.nodes[i];
            const b = graph.nodes[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 0.01) {
              const angle = (hashText(`${a.id}|${b.id}`) % 360) / 360 * Math.PI * 2;
              dx = Math.cos(angle);
              dy = Math.sin(angle);
              dist = 1;
            }
            const minDistance = collisionRadius(a) + collisionRadius(b) + (a.type === "region" || b.type === "region" ? 10 : 4);
            if (dist >= minDistance) continue;
            const correction = (minDistance - dist) * 0.22;
            const ux = dx / dist;
            const uy = dy / dist;
            if (!a.fixed && !b.fixed) {
              a.x -= ux * correction * 0.5;
              a.y -= uy * correction * 0.5;
              b.x += ux * correction * 0.5;
              b.y += uy * correction * 0.5;
            } else if (!a.fixed) {
              a.x -= ux * correction;
              a.y -= uy * correction;
            } else if (!b.fixed) {
              b.x += ux * correction;
              b.y += uy * correction;
            }
            [a, b].forEach((node) => {
              if (node.fixed) return;
              const margin = collisionRadius(node) + 8;
              node.x = Math.max(margin, Math.min(width - margin, node.x));
              node.y = Math.max(margin, Math.min(height - margin, node.y));
            });
          }
        }
      }
      return graph;
    }
    function nodeRadius(node) {
      if (node.type === "region") return 16;
      return Math.min(19, 5.5 + Math.log1p(Number(node.weight || 0)) * 1.65 + Math.max(0, node.shared - 1) * 1.45);
    }
    function collisionRadius(node) {
      const radius = nodeRadius(node);
      if (node.type === "region" || !state.shapeByPos) return radius;
      if (["noun_entity", "state_attitude", "function_quantity_other"].includes(node.posGroup)) return radius * 1.2;
      return radius;
    }
    function labelFontSize(node) {
      const length = String(node.label || "").length;
      if (node.type === "region") return length > 18 ? 8.5 : length > 11 ? 9.5 : 10.5;
      return length > 5 ? 9.5 : 11;
    }
    function tokenColor(node) {
      if (node.isRegionRef) return "#ad5d20";
      if (node.shared >= 4) return mixRegionColors(node.regions || [], 0.48);
      if (node.shared === 3) return mixRegionColors(node.regions || [], 0.34);
      if (node.shared === 2) return mixRegionColors(node.regions || [], 0.18);
      return mixRegionColors(node.regions || [], 0.04);
    }
    function isActiveNode(node) {
      if (state.highlightedEdge) {
        const edge = currentGraph.edges.find((item) => edgeContextKeyFromEdge(item) === state.highlightedEdge);
        if (!edge) return true;
        return edge.source === node.id || edge.target === node.id;
      }
      if (state.highlightedNode) {
        if (node.id === state.highlightedNode) return true;
        return currentGraph.edges.some((edge) =>
          (edge.source === state.highlightedNode && edge.target === node.id) ||
          (edge.target === state.highlightedNode && edge.source === node.id)
        );
      }
      if (!state.nodeSearch) return true;
      if (isSearchHit(node)) return true;
      return currentGraph.edges.some((edge) =>
        (edge.source === node.id && isSearchHit(nodeById(edge.target))) ||
        (edge.target === node.id && isSearchHit(nodeById(edge.source)))
      );
    }
    function isActiveEdge(edge) {
      if (state.highlightedEdge) return edgeContextKeyFromEdge(edge) === state.highlightedEdge;
      if (state.highlightedNode) return edge.source === state.highlightedNode || edge.target === state.highlightedNode;
      if (!state.nodeSearch) return true;
      return isSearchHit(nodeById(edge.source)) || isSearchHit(nodeById(edge.target));
    }
    function nodeById(id) {
      return currentGraph.nodes.find((node) => node.id === id) || null;
    }
    function isSearchHit(node) {
      if (!node || !state.nodeSearch) return false;
      const values = [node.label, node.region, (node.regions || []).join(" ")];
      return values.some((value) => String(value || "").toLowerCase().includes(state.nodeSearch));
    }
    let currentGraph = {nodes: [], edges: []};
    let graphEventsBound = false;
    function nodeShapeMarkup(node, radius, cls, fill, title) {
      const titleMarkup = `<title>${esc(title)}</title>`;
      const shapeClass = `node-shape ${cls}`;
      if (!state.shapeByPos) {
        return `<circle class="${shapeClass}" cx="${node.x}" cy="${node.y}" r="${radius}" fill="${fill}">${titleMarkup}</circle>`;
      }
      if (node.posGroup === "noun_entity") {
        const side = radius * 1.7;
        const half = side / 2;
        return `<rect class="${shapeClass}" x="${node.x - half}" y="${node.y - half}" width="${side}" height="${side}" rx="2.5" fill="${fill}">${titleMarkup}</rect>`;
      }
      if (node.posGroup === "state_attitude") {
        const r = radius * 1.12;
        const points = `${node.x},${node.y - r} ${node.x + r},${node.y} ${node.x},${node.y + r} ${node.x - r},${node.y}`;
        return `<polygon class="${shapeClass}" points="${points}" fill="${fill}">${titleMarkup}</polygon>`;
      }
      if (node.posGroup === "function_quantity_other") {
        const r = radius * 1.12;
        const points = `${node.x},${node.y - r} ${node.x + r * 0.98},${node.y + r * 0.78} ${node.x - r * 0.98},${node.y + r * 0.78}`;
        return `<polygon class="${shapeClass}" points="${points}" fill="${fill}">${titleMarkup}</polygon>`;
      }
      return `<circle class="${shapeClass}" cx="${node.x}" cy="${node.y}" r="${radius}" fill="${fill}">${titleMarkup}</circle>`;
    }
    function drawGraph(graph) {
      currentGraph = graph;
      const svg = byId("graph");
      const rect = svg.getBoundingClientRect();
      const width = Math.max(760, rect.width || 900);
      const height = Math.max(560, rect.height || 680);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
      const edges = graph.edges.map((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) return "";
        const edgeKey = edgeContextKeyFromEdge(edge);
        const dim = isActiveEdge(edge) ? "" : " dim";
        const searchHit = state.nodeSearch && (isSearchHit(source) || isSearchHit(target)) ? " search-hit" : "";
        const selected = state.highlightedEdge === edgeKey ? " selected-edge" : "";
        const widthValue = Math.max(1.0, Math.min(7, Math.log1p(edge.weight) * 1.1));
        return `<line class="edge${dim}${searchHit}${selected}" data-edge-key="${esc(edgeKey)}" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="${edge.color}" stroke-width="${widthValue}"><title>${esc(edge.region)} - ${esc(target.label)} | weight ${fmt(edge.weight, 0)} | PMI ${fmt(edge.pmi, 2)}</title></line>`;
      }).join("");
      const nodes = graph.nodes.map((node) => {
        const active = isActiveNode(node);
        const dim = active ? "" : " dim";
        const searchHit = isSearchHit(node) ? " search-hit" : "";
        const radius = nodeRadius(node);
        const fill = node.type === "region" ? node.color : tokenColor(node);
        const ambiguousCls = node.type === "token" && state.markAmbiguousPos && node.posAmbiguous ? " ambiguous-pos" : "";
        const cls = (node.type === "region" ? "region-node" : "token-node") + ambiguousCls;
        const labelCls = node.type === "region" ? "region-label" : "token-label";
        const fontSize = labelFontSize(node);
        const title = node.type === "region"
          ? `${node.label} | POS ${node.dominantPos || "(missing)"} (${fmt(Number(node.dominantPosShare || 0) * 100, 1)}%) | ${node.posGroupLabel || posGroupLabel(node.posGroup || "missing")}`
          : `${node.label} | POS ${node.dominantPos} (${fmt(node.dominantPosShare * 100, 1)}%) | ${node.posGroupLabel}${node.posAmbiguous ? " | ambiguous POS" : ""} | shared regions ${node.shared} | weight ${fmt(node.weight, 0)} | ${esc((node.regions || []).join(", "))}`;
        const shape = nodeShapeMarkup(node, radius, cls, fill, title);
        return `<g class="node${dim}${searchHit}" data-node-id="${esc(node.id)}">
          ${shape}
          <text class="node-label ${labelCls}${dim}${searchHit}" x="${node.x}" y="${node.y}" text-anchor="middle" font-size="${fontSize}">${esc(node.label)}</text>
        </g>`;
      }).join("");
      const transform = `translate(${state.zoom.x} ${state.zoom.y}) scale(${state.zoom.scale})`;
      svg.innerHTML = `<rect id="graph-bg" x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect><g id="graph-viewport" transform="${transform}">${edges}${nodes}</g>`;
      svg.querySelectorAll("[data-edge-key]").forEach((item) => {
        item.addEventListener("click", (event) => {
          event.stopPropagation();
          const edgeKey = item.getAttribute("data-edge-key");
          state.highlightedEdge = state.highlightedEdge === edgeKey ? "" : edgeKey;
          if (state.highlightedEdge) state.highlightedNode = "";
          drawGraph(currentGraph);
          renderEdgeContexts();
        });
      });
      svg.querySelectorAll("[data-node-id]").forEach((item) => {
        item.addEventListener("click", (event) => {
          event.stopPropagation();
          const nodeId = item.getAttribute("data-node-id");
          state.highlightedNode = state.highlightedNode === nodeId ? "" : nodeId;
          if (state.highlightedNode) state.highlightedEdge = "";
          drawGraph(currentGraph);
          renderEdgeContexts();
        });
      });
      if (!graphEventsBound) setupGraphInteractions(svg);
    }
    function graphPoint(svg, event) {
      const rect = svg.getBoundingClientRect();
      const box = svg.viewBox.baseVal;
      return {
        x: box.x + (event.clientX - rect.left) / Math.max(1, rect.width) * box.width,
        y: box.y + (event.clientY - rect.top) / Math.max(1, rect.height) * box.height,
      };
    }
    function graphCenter() {
      const box = byId("graph").viewBox.baseVal;
      return {x: box.x + box.width / 2, y: box.y + box.height / 2};
    }
    function applyZoom(factor, center = null) {
      const point = center || graphCenter();
      const oldScale = state.zoom.scale;
      const nextScale = Math.max(0.35, Math.min(5, oldScale * factor));
      if (nextScale === oldScale) return;
      state.zoom.x = point.x - (point.x - state.zoom.x) * (nextScale / oldScale);
      state.zoom.y = point.y - (point.y - state.zoom.y) * (nextScale / oldScale);
      state.zoom.scale = nextScale;
      drawGraph(currentGraph);
    }
    function resetZoom() {
      state.zoom = {scale: 1, x: 0, y: 0};
      drawGraph(currentGraph);
    }
    function safeFilenamePart(value) {
      return String(value || "")
        .trim()
        .replace(/[\\/:*?"<>|]+/g, "_")
        .replace(/\s+/g, "_")
        .slice(0, 80) || "value";
    }
    function captureGraphPng() {
      const svg = byId("graph");
      if (!svg || !currentGraph.nodes.length) return;
      const box = svg.viewBox.baseVal;
      const width = Math.max(1, Math.round(box.width || svg.clientWidth || 900));
      const height = Math.max(1, Math.round(box.height || svg.clientHeight || 680));
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const bg = clone.querySelector("#graph-bg");
      if (bg) bg.setAttribute("fill", "#fbfbf8");
      const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
      style.textContent = `
        .edge { stroke-opacity: 0.48; }
        .edge.search-hit { stroke-opacity: 0.88; }
        .edge.dim, .node.dim, .node-label.dim { opacity: 0.16; }
        .region-node { stroke: #263238; stroke-width: 1.2px; }
        .token-node { stroke: #ffffff; stroke-width: 1.2px; }
        .token-node.ambiguous-pos { stroke: #111827; stroke-width: 2.6px; stroke-dasharray: 4 2; }
        .node-label {
          pointer-events: none;
          paint-order: stroke;
          stroke: #fbfbf8;
          stroke-width: 3px;
          stroke-linejoin: round;
          fill: #1f2933;
          font-weight: 700;
          dominant-baseline: middle;
          font-family: "Malgun Gothic", "Microsoft YaHei", Arial, sans-serif;
        }
        .region-label { fill: #ffffff; stroke: rgba(0, 0, 0, 0.38); stroke-width: 3px; }
        .node.search-hit .node-shape { stroke: #f2b705; stroke-width: 4px; }
        .node-label.search-hit { fill: #111827; stroke: #f8d463; stroke-width: 4px; }
      `;
      clone.insertBefore(style, clone.firstChild);
      const source = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([source], {type: "image/svg+xml;charset=utf-8"});
      const url = URL.createObjectURL(blob);
      const image = new Image();
      image.onload = () => {
        const scale = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#fbfbf8";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        const link = document.createElement("a");
        const parts = [
          "multi_core_ego",
          safeFilenamePart(state.profile),
          `w${safeFilenamePart(state.window)}`,
          safeFilenamePart(state.centerMode),
          safeFilenamePart(state.period),
          `top${safeFilenamePart(state.topN)}`,
        ];
        link.download = `${parts.join("_")}.png`;
        link.href = canvas.toDataURL("image/png");
        document.body.appendChild(link);
        link.click();
        link.remove();
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        alert("Graph capture failed in this browser.");
      };
      image.src = url;
    }
    function setupGraphInteractions(svg) {
      graphEventsBound = true;
      svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
        applyZoom(factor, graphPoint(svg, event));
      }, {passive: false});
      svg.addEventListener("mousedown", (event) => {
        if (event.target.closest("[data-node-id]") || event.target.closest("[data-edge-key]")) return;
        state.pan.active = true;
        state.pan.moved = false;
        state.pan.startX = event.clientX;
        state.pan.startY = event.clientY;
        state.pan.originX = state.zoom.x;
        state.pan.originY = state.zoom.y;
        svg.classList.add("panning");
      });
      window.addEventListener("mousemove", (event) => {
        if (!state.pan.active) return;
        const rect = svg.getBoundingClientRect();
        const box = svg.viewBox.baseVal;
        const dx = (event.clientX - state.pan.startX) / Math.max(1, rect.width) * box.width;
        const dy = (event.clientY - state.pan.startY) / Math.max(1, rect.height) * box.height;
        state.pan.moved = Math.abs(dx) + Math.abs(dy) > 2;
        state.zoom.x = state.pan.originX + dx;
        state.zoom.y = state.pan.originY + dy;
        drawGraph(currentGraph);
      });
      window.addEventListener("mouseup", () => {
        if (!state.pan.active) return;
        state.pan.active = false;
        svg.classList.remove("panning");
      });
      svg.addEventListener("click", (event) => {
        if (event.target.closest("[data-node-id]") || event.target.closest("[data-edge-key]")) return;
        if (state.pan.moved) {
          state.pan.moved = false;
          return;
        }
        if (!state.highlightedNode && !state.highlightedEdge) return;
        state.highlightedNode = "";
        state.highlightedEdge = "";
        drawGraph(currentGraph);
        renderEdgeContexts();
      });
    }
    function sharedTokenRows(rows) {
      const byToken = new Map();
      rows.forEach((row) => {
        const item = byToken.get(row.alter_token) || {
          token: row.alter_token,
          regions: new Set(),
          weight: 0,
          dominant_pos: row.dominant_pos || "(missing)",
          pos_group_label: row.pos_group_label || posGroupLabel(row.pos_group || "missing"),
          dominant_pos_share: Number(row.dominant_pos_share || 0),
          pos_is_ambiguous: Boolean(row.pos_is_ambiguous),
        };
        item.regions.add(row.region_norm);
        item.weight += Number(row.edge_weight || 0);
        item.pos_is_ambiguous = Boolean(item.pos_is_ambiguous || row.pos_is_ambiguous);
        byToken.set(row.alter_token, item);
      });
      return Array.from(byToken.values())
        .map((item) => ({
          token: item.token,
          region_count: item.regions.size,
          regions: Array.from(item.regions).join(", "),
          weight: item.weight,
          dominant_pos: item.dominant_pos,
          pos_group_label: item.pos_group_label,
          dominant_pos_share: item.dominant_pos_share,
          pos_is_ambiguous: item.pos_is_ambiguous,
        }))
        .sort((a, b) => Number(b.region_count) - Number(a.region_count) || Number(b.weight) - Number(a.weight) || a.token.localeCompare(b.token))
        .slice(0, 25);
    }
    function rawSortValue(row, column) {
      if (column.sortValue) return column.sortValue(row);
      return row[column.key];
    }
    function compareValues(left, right) {
      const leftEmpty = left === null || left === undefined || left === "";
      const rightEmpty = right === null || right === undefined || right === "";
      if (leftEmpty && rightEmpty) return 0;
      if (leftEmpty) return 1;
      if (rightEmpty) return -1;
      const leftNumber = Number(String(left).replaceAll(",", ""));
      const rightNumber = Number(String(right).replaceAll(",", ""));
      if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber;
      return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
    }
    function sortedRows(rows, columns, tableId) {
      const active = tableSortState[tableId];
      if (!active) return rows;
      const column = columns.find((item) => item.key === active.key);
      if (!column) return rows;
      return rows.slice().sort((left, right) => {
        const result = compareValues(rawSortValue(left, column), rawSortValue(right, column));
        return active.dir === "desc" ? -result : result;
      });
    }
    function sortIndicator(tableId, key) {
      const active = tableSortState[tableId];
      if (!active || active.key !== key) return "";
      return `<span class="sort-indicator">${active.dir === "asc" ? "&uarr;" : "&darr;"}</span>`;
    }
    function table(rows, columns, tableId) {
      if (!rows.length) return '<div class="empty">No rows for this selection.</div>';
      const displayRows = sortedRows(rows, columns, tableId);
      const head = '<tr>' + columns.map((col) => `<th class="sortable" data-table-id="${esc(tableId)}" data-sort-key="${esc(col.key)}">${esc(col.label)}${sortIndicator(tableId, col.key)}</th>`).join("") + '</tr>';
      const body = displayRows.map((row) => '<tr>' + columns.map((col) => `<td>${esc(col.format ? col.format(row[col.key], row) : row[col.key])}</td>`).join("") + '</tr>').join("");
      return `<div class="table-wrap"><table>${head}${body}</table></div>`;
    }
    function markEscapedTerm(html, term, cls) {
      const safeTerm = esc(term || "");
      if (!safeTerm) return html;
      return html.split(safeTerm).join(`<mark class="${cls}">${safeTerm}</mark>`);
    }
    function markedContextText(row, edge) {
      let html = esc(row.context_text || "");
      const focusTokens = String(row.focus_tokens || edge?.region || "")
        .split(";")
        .map((value) => value.trim())
        .filter((value) => value)
        .sort((a, b) => b.length - a.length);
      focusTokens.forEach((token) => {
        html = markEscapedTerm(html, token, "focus-mark");
      });
      html = markEscapedTerm(html, edge?.alterToken || row.alter_token, "neighbor-mark");
      return html;
    }
    function renderEdgeContexts() {
      const target = byId("edgeContexts");
      if (!target) return;
      if (DATA.meta.public_release) {
        target.innerHTML = '<h2>Selected edge contexts</h2><div class="empty">Edge-level context excerpts are omitted from the public edition because source texts are not redistributed.</div>';
        return;
      }
      if (!state.highlightedEdge) {
        target.innerHTML = '<h2>Selected edge contexts</h2><div class="empty">Click an edge in the graph to show keyword-centered contexts.</div>';
        return;
      }
      const edge = currentGraph.edges.find((item) => edgeContextKeyFromEdge(item) === state.highlightedEdge) || null;
      const rows = (EDGE_CONTEXT_LOOKUP.get(state.highlightedEdge) || []).slice();
      if (!edge) {
        target.innerHTML = '<h2>Selected edge contexts</h2><div class="empty">Selected edge is no longer visible.</div>';
        return;
      }
      const availableContextCount = rows.length ? Number(rows[0].available_context_count || 0) : 0;
      const availableArticleCount = rows.length ? Number(rows[0].available_article_count || 0) : 0;
      const header = `<h2>Selected edge contexts</h2>
        <div class="small">${esc(edge.region)} - ${esc(edge.alterToken)} | weight ${fmt(edge.weight, 0)} | PMI ${fmt(edge.pmi, 2)} | matched contexts ${fmt(availableContextCount, 0)} | articles ${fmt(availableArticleCount, 0)}</div>`;
      if (!rows.length) {
        target.innerHTML = header + '<div class="empty">No embedded context rows for this edge. The edge can still be valid; only representative context rows are embedded in this dashboard.</div>';
        return;
      }
      const cards = rows
        .sort((a, b) => Number(a.context_rank || 0) - Number(b.context_rank || 0))
        .map((row) => {
          const meta = [
            `#${fmt(row.context_rank, 0)}`,
            row.date || "",
            row.article_id ? `article ${row.article_id}` : "",
            row.issue_page ? `page ${row.issue_page}` : "",
            row.min_token_distance ? `distance ${fmt(row.min_token_distance, 0)}` : "",
            row.focus_tokens ? `focus ${row.focus_tokens}` : "",
          ].filter((value) => String(value || "").trim());
          return `<div class="context-card">
            <div class="context-meta">${meta.map((value) => `<span>${esc(value)}</span>`).join("")}</div>
            <div class="context-text">${markedContextText(row, edge)}</div>
          </div>`;
        })
        .join("");
      target.innerHTML = header + `<div class="context-list">${cards}</div>`;
    }
    function renderStats(rows, graph) {
      const sharedRows = sharedTokenRows(rows);
      const stats = [
        [UI.entity_plural_title, state.selectedRegions.length],
        ["Token nodes", graph.nodes.filter((node) => node.type === "token").length],
        ["Edges", graph.edges.length],
        ["Shared tokens", sharedRows.filter((row) => Number(row.region_count) >= 2).length],
        ["Ambiguous POS", graph.nodes.filter((node) => node.type === "token" && node.posAmbiguous).length],
      ];
      byId("stats").innerHTML = stats.map(([label, value]) => `<div class="metric"><div class="value">${fmt(value, 0)}</div><div class="label">${esc(label)}</div></div>`).join("");
      byId("sharedTokens").innerHTML = `<h2>Visible neighbor tokens by connected ${esc(UI.entity_plural)}</h2>` + table(sharedRows, [
        {key: "token", label: "Token"},
        {key: "region_count", label: UI.entity_plural_title, format: (value) => fmt(value, 0)},
        {key: "weight", label: "Weight", format: (value) => fmt(value, 0)},
        {key: "dominant_pos", label: "POS"},
        {key: "dominant_pos_share", label: "POS share", format: (value) => `${fmt(Number(value || 0) * 100, 1)}%`},
        {key: "regions", label: `Connected ${UI.entity_plural}`},
      ], "sharedTokens");
      const edgeRows = rows
        .slice()
        .sort((a, b) => Number(b.edge_weight || 0) - Number(a.edge_weight || 0))
        .slice(0, 80);
      byId("edgeTable").innerHTML = `<h2>Top visible edges</h2>` + table(edgeRows, [
        {key: "region_norm", label: UI.entity_singular_title},
        {key: "alter_token", label: "Neighbor"},
        {key: "dominant_pos", label: "POS"},
        {key: "pos_group_label", label: "POS group"},
        {key: "dominant_pos_share", label: "POS share", format: (value) => `${fmt(Number(value || 0) * 100, 1)}%`},
        {key: "edge_weight", label: "Weight", format: (value) => fmt(value, 0)},
        {key: "pmi_mean", label: "PMI", format: (value) => fmt(value, 2)},
      ], "edgeTable");
    }
    function legendShapeIcon(group) {
      const stroke = "#4b5563";
      const fill = "#eef2f1";
      if (group === "action_process") {
        return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5.1" fill="${fill}" stroke="${stroke}" stroke-width="1.4"></circle></svg>`;
      }
      if (group === "noun_entity") {
        return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.2" y="2.2" width="9.6" height="9.6" rx="1.5" fill="${fill}" stroke="${stroke}" stroke-width="1.4"></rect></svg>`;
      }
      if (group === "state_attitude") {
        return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><polygon points="7,1.8 12.2,7 7,12.2 1.8,7" fill="${fill}" stroke="${stroke}" stroke-width="1.4"></polygon></svg>`;
      }
      if (group === "function_quantity_other") {
        return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><polygon points="7,1.8 12,11.4 2,11.4" fill="${fill}" stroke="${stroke}" stroke-width="1.4"></polygon></svg>`;
      }
      return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5.1" fill="${fill}" stroke="${stroke}" stroke-width="1.4"></circle></svg>`;
    }
    function renderLegend() {
      const regions = state.selectedRegions;
      const regionItems = regions.map((region) => `<span class="legend-item"><span class="swatch" style="background:${regionColor(region)}"></span>${esc(region)}</span>`);
      const selectedPos = new Set(state.selectedPosGroups || []);
      const shapeItems = state.shapeByPos
        ? (DATA.meta.pos_group_order || []).filter((group) => group !== "missing" && selectedPos.has(group) && availablePosGroups().includes(group))
          .map((group) => `<span class="legend-item"><span class="legend-shape">${legendShapeIcon(group)}</span>${esc(posGroupLabel(group))}</span>`)
        : [];
      byId("legend").innerHTML = regionItems.concat(shapeItems).join("");
    }
    function renderAll(refreshOptions = true) {
      if (refreshOptions) updateOptions();
      else updateStateFromControls();
      const rows = selectedRows();
      const graph = layoutGraph(buildGraph(rows), byId("graph").clientWidth || 900, byId("graph").clientHeight || 680);
      drawGraph(graph);
      renderLegend();
      renderStats(rows, graph);
      renderEdgeContexts();
      byId("selectionNote").textContent = `${state.selectedRegions.length} selected / ${availableRegions().length} available`;
    }
    function init() {
      renderTopNOptions();
      byId("includeRegionRefs").checked = Boolean(DATA.meta.default_include_region_reference_neighbors);
      byId("markAmbiguousPos").checked = true;
      byId("shapeByPos").checked = false;
      fields.forEach((id) => {
        const el = byId(id);
        if (!el) return;
        el.addEventListener("change", () => {
          state.highlightedNode = "";
          state.highlightedEdge = "";
          renderAll();
        });
      });
      byId("topN").addEventListener("change", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("minShared").addEventListener("input", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("maxShared").addEventListener("input", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("includeRegionRefs").addEventListener("change", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("markAmbiguousPos").addEventListener("change", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("shapeByPos").addEventListener("change", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      byId("nodeSearch").addEventListener("input", () => {
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll(false);
      });
      document.addEventListener("click", (event) => {
        const header = event.target.closest("th[data-table-id][data-sort-key]");
        if (!header) return;
        const tableId = header.dataset.tableId;
        const key = header.dataset.sortKey;
        const active = tableSortState[tableId];
        tableSortState[tableId] = active && active.key === key && active.dir === "asc"
          ? {key, dir: "desc"}
          : {key, dir: "asc"};
        renderStats(selectedRows(), currentGraph);
      });
      byId("zoomIn").addEventListener("click", () => applyZoom(1.2));
      byId("zoomOut").addEventListener("click", () => applyZoom(1 / 1.2));
      byId("zoomReset").addEventListener("click", resetZoom);
      byId("captureGraph").addEventListener("click", captureGraphPng);
      byId("selectMajor").addEventListener("click", () => {
        const regions = availableRegions();
        state.selectedRegions = DATA.meta.default_selected_regions.filter((region) => regions.includes(region));
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll();
      });
      byId("selectAll").addEventListener("click", () => {
        state.selectedRegions = availableRegions();
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll();
      });
      byId("clearRegions").addEventListener("click", () => {
        state.selectedRegions = [];
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderAll();
      });
      byId("selectDefaultPos").addEventListener("click", () => {
        const groups = availablePosGroups();
        state.selectedPosGroups = (DATA.meta.default_selected_pos_groups || []).filter((group) => groups.includes(group));
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderPosControls(false);
        renderAll(false);
      });
      byId("selectAllPos").addEventListener("click", () => {
        state.selectedPosGroups = availablePosGroups();
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderPosControls(false);
        renderAll(false);
      });
      byId("clearPos").addEventListener("click", () => {
        state.selectedPosGroups = [];
        state.highlightedNode = "";
        state.highlightedEdge = "";
        renderPosControls(false);
        renderAll(false);
      });
      window.addEventListener("resize", () => renderAll(false));
      renderAll();
    }
    init();
  </script>
</body>
</html>
"""
    rendered = inject_controls_collapse(
        template.replace("__TITLE__", safe_title)
        .replace("__DATA__", payload_json)
        .replace("__DATA_SCOPE_CONTROL__", data_scope_control)
        .replace("__ENTITY_SINGULAR__", entity_singular)
        .replace("__ENTITY_PLURAL__", entity_plural)
        .replace("__CHECKBOX_LABEL__", checkbox_label)
        .replace("__SEARCH_PLACEHOLDER__", search_placeholder)
        .replace("__GRAPH_HINT__", html_escape(graph_hint))
    )
    if public_release:
        start = rendered.index("    function markEscapedTerm")
        end = rendered.index("    function renderStats", start)
        public_context_renderer = """    function renderEdgeContexts() {
      const target = byId(\"edgeContexts\");
      if (!target) return;
      target.innerHTML = '<h2>Selected edge contexts</h2><div class=\"empty\">Edge-level context excerpts are omitted from the public edition because source texts are not redistributed.</div>';
    }
"""
        rendered = rendered[:start] + public_context_renderer + rendered[end:]
    return rendered


def write_html(payload: Dict[str, object], args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else output_dir / DEFAULT_OUTPUT_HTML.name
    )
    if not output_path.is_absolute():
        output_path = output_dir / output_path
    payload_json = json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_template(args.title, payload_json), encoding="utf-8")
    log(f"Wrote HTML: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = read_csv(input_path)
    require_columns(df, input_path)
    df = normalize_dataframe(df)
    df = filter_rows(df, args)
    df = attach_pos_metadata(df, args)
    if args.prepared_output_csv:
        prepared_path = Path(args.prepared_output_csv).expanduser().resolve()
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(prepared_path, index=False, encoding="utf-8")
        log(f"Wrote prepared aggregate CSV: {prepared_path}")
    payload = build_payload(df, args)
    output_path = write_html(payload, args)
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_path.with_suffix(".summary.json")
    )
    write_json(
        {
            "input_csv": input_path.name if args.public_release else str(input_path),
            "output_html": output_path.name if args.public_release else str(output_path),
            "rows": len(df),
            "regions": selected_regions(args, df),
            "generated_at": payload["meta"]["generated_at"],
        },
        summary_path,
    )


if __name__ == "__main__":
    main()
