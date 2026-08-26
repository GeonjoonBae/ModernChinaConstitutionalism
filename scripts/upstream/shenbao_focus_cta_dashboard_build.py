#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build focus-norm CTA dashboards from focus-anchor annotation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from shenbao_context_filter_utils import apply_context_filter_to_df, load_context_filter
from shenbao_html_controls import inject_controls_collapse
from shenbao_token_filter_utils import (
    DEFAULT_TOKEN_FILTER,
    apply_token_filter_to_df,
    load_token_filter,
)


ROOT = Path(__file__).resolve().parent
SHENBAO = ROOT / "shenbao"
DEFAULT_ANNOTATION_CSV = (
    SHENBAO
    / "shenbao_interpretation"
    / "focus_anchor_annotation"
    / "focus_anchor_annotation_candidates.csv"
)
DEFAULT_TOKENS_ROOT = SHENBAO / "shenbao_network" / "applied_tokens"
DEFAULT_PERIODS_PARQUET = DEFAULT_TOKENS_ROOT / "strict" / "periods" / "periods.parquet"
DEFAULT_FILTER_ROOT = SHENBAO / "shenbao_filters"
DEFAULT_NETWORK_ROOT = SHENBAO / "shenbao_network" / "network_applied"
DEFAULT_OUTPUT_DIR = SHENBAO / "shenbao_interpretation" / "focus_anchor_dashboard"

PROFILE_DIRS = {
    "regex-only": "regex_only",
    "regex_only": "regex_only",
    "strict": "strict",
    "full": "full",
}
PROFILE_LABELS = {
    "regex_only": "regex-only",
    "regex-only": "regex-only",
    "strict": "strict",
    "full": "full",
}
FOCUS_LABELS = {
    "xianzheng": "憲政",
    "lixian": "立憲",
    "xianfa": "憲法",
    "zhixian": "制憲",
}
FOCUS_ORDER = ["xianzheng", "lixian", "xianfa", "zhixian"]
DATA_SCOPE = "body"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def split_values(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[|,;\s]+", text) if part.strip()]


def parse_csv_list(value: str) -> List[str]:
    return split_values(value)


def parse_int_list(value: str) -> List[int]:
    return [int(part) for part in split_values(value)]


def normalize_profile(value: str) -> str:
    if value not in PROFILE_DIRS:
        raise ValueError(f"Unknown token profile: {value}")
    return PROFILE_LABELS.get(value, value)


def compact_profile(value: str) -> str:
    return "ro" if value == "regex-only" else ("st" if value == "strict" else "fu")


def esc(value: object) -> str:
    return html.escape("" if value is None or pd.isna(value) else str(value), quote=True)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if isinstance(value, float) else False:
        return None
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build focus-norm CTA, neighbor-overlap, and multi-ego HTML dashboards."
    )
    parser.add_argument("--annotation-csv", default=str(DEFAULT_ANNOTATION_CSV))
    parser.add_argument("--tokens-root", default=str(DEFAULT_TOKENS_ROOT))
    parser.add_argument("--periods-parquet", default=str(DEFAULT_PERIODS_PARQUET))
    parser.add_argument("--period-set-id", default="long_period_manual")
    parser.add_argument("--context-filter", default="filter_context_pre_zhixian")
    parser.add_argument("--context-filter-root", default=str(DEFAULT_FILTER_ROOT))
    parser.add_argument("--token-filter", default=str(DEFAULT_TOKEN_FILTER))
    parser.add_argument("--network-root", default=str(DEFAULT_NETWORK_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profiles", default="regex-only,strict,full")
    parser.add_argument("--windows", default="5,10,20")
    parser.add_argument(
        "--include-mode",
        choices=["active", "max", "all"],
        default="active",
        help="Annotation candidate inclusion tier for mentions. Default: active.",
    )
    parser.add_argument(
        "--include-review-flags",
        default="auto_high,auto_medium",
        help="Review flags to include. Use all to keep every review flag allowed by --include-mode.",
    )
    parser.add_argument("--include-global", type=parse_bool, default=True)
    parser.add_argument("--top-tokens-per-focus", type=int, default=30)
    parser.add_argument("--top-neighbors", type=int, default=80)
    parser.add_argument("--overlap-topn", type=int, default=80)
    parser.add_argument("--max-network-rows-per-file", type=int, default=0, help="Development limit; 0 means no limit.")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--write-html", type=parse_bool, default=True)
    return parser.parse_args()


def load_periods(path: Path, period_set_id: str) -> pd.DataFrame:
    periods = pd.read_parquet(path)
    periods = periods[periods["period_set_id"].astype(str).eq(period_set_id)].copy()
    if periods.empty:
        raise ValueError(f"No periods found for period_set_id={period_set_id}: {path}")
    periods["start_date"] = pd.to_datetime(periods["start_date"])
    periods["end_date"] = pd.to_datetime(periods["end_date"])
    periods = periods.sort_values("sort_order", kind="mergesort").reset_index(drop=True)
    return periods


def load_annotation(args: argparse.Namespace, profiles: Sequence[str]) -> pd.DataFrame:
    path = Path(args.annotation_csv).expanduser().resolve()
    raise_csv_field_limit()
    usecols = [
        "profile",
        "token",
        "focus_norms",
        "focus_roots",
        "active_include",
        "max_include",
        "review_flag",
        "review_priority",
        "confidence_score",
        "confidence_level",
        "positive_signals",
        "negative_signals",
        "anchor_group_candidate",
        "anchor_subgroup_candidate",
        "general_term_policy",
        "dominant_pos",
        "dict_lv1",
        "dict_lv2",
        "dict_ner_like_type",
    ]
    ann = pd.read_csv(path, usecols=lambda col: col in usecols, encoding="utf-8-sig")
    ann["profile"] = ann["profile"].map(normalize_profile)
    ann = ann[ann["profile"].isin(profiles)].copy()
    ann = ann[ann["focus_norms"].notna() & ann["token"].notna()].copy()

    if args.include_mode == "active":
        ann = ann[ann["active_include"].fillna(False).astype(bool)].copy()
    elif args.include_mode == "max":
        ann = ann[ann["max_include"].fillna(False).astype(bool)].copy()

    flags = [flag for flag in parse_csv_list(args.include_review_flags)]
    if flags and "all" not in {flag.lower() for flag in flags}:
        ann = ann[ann["review_flag"].astype(str).isin(flags)].copy()

    rows: List[Dict[str, object]] = []
    for record in ann.to_dict("records"):
        focus_norms = split_values(record.get("focus_norms"))
        focus_roots = split_values(record.get("focus_roots"))
        for focus_norm in focus_norms:
            if focus_norm not in FOCUS_LABELS:
                continue
            row = dict(record)
            row["focus_norm"] = focus_norm
            row["focus_label_zh"] = FOCUS_LABELS[focus_norm]
            row["focus_roots_list"] = "|".join(focus_roots)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No focus annotation candidates remained after filtering.")
    out = out.drop_duplicates(["profile", "token", "focus_norm"], keep="first")
    out["token"] = out["token"].astype(str)
    return out


def assign_periods(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out["period_set_id"] = pd.NA
    out["period_id"] = pd.NA
    out["period_label"] = pd.NA
    out["period_sort_order"] = pd.NA
    out["period_start_date"] = pd.NA
    out["period_end_date"] = pd.NA
    for period in periods.to_dict("records"):
        mask = out["date_dt"].between(period["start_date"], period["end_date"], inclusive="both")
        if not mask.any():
            continue
        out.loc[mask, "period_set_id"] = period["period_set_id"]
        out.loc[mask, "period_id"] = period["period_id"]
        out.loc[mask, "period_label"] = period.get("label", period["period_id"])
        out.loc[mask, "period_sort_order"] = int(period["sort_order"])
        out.loc[mask, "period_start_date"] = period["start_date"].date().isoformat()
        out.loc[mask, "period_end_date"] = period["end_date"].date().isoformat()
    out = out[out["period_id"].notna()].copy()
    return out.drop(columns=["date_dt"])


def load_mentions(args: argparse.Namespace, annotation: pd.DataFrame, periods: pd.DataFrame, profiles: Sequence[str]) -> pd.DataFrame:
    tokens_root = Path(args.tokens_root).expanduser().resolve()
    filter_info = load_context_filter(args.context_filter, Path(args.context_filter_root).expanduser().resolve())
    token_filter_rows = load_token_filter(args.token_filter)
    mention_parts: List[pd.DataFrame] = []
    ann_cols = [
        "profile",
        "token",
        "focus_norm",
        "focus_label_zh",
        "focus_roots",
        "focus_roots_list",
        "confidence_score",
        "confidence_level",
        "review_flag",
        "review_priority",
        "anchor_group_candidate",
        "anchor_subgroup_candidate",
        "positive_signals",
        "negative_signals",
        "general_term_policy",
    ]
    for profile in profiles:
        profile_dir = PROFILE_DIRS[profile]
        token_path = tokens_root / profile_dir / "tokens.parquet"
        if not token_path.exists():
            log(f"Skip missing token parquet: {token_path}")
            continue
        profile_ann = annotation[annotation["profile"].eq(profile)][ann_cols].copy()
        if profile_ann.empty:
            continue
        token_set = set(profile_ann["token"].astype(str))
        cols = [
            "token_uid",
            "context_uid",
            "article_uid",
            "article_id",
            "date",
            "token_order_in_context",
            "token",
            "pos",
            "token_source",
            "pos_source",
            "dict_lv1",
            "dict_lv2",
            "dict_ner_like_type",
        ]
        log(f"Loading mentions for profile={profile}: {token_path}")
        try:
            df = pd.read_parquet(token_path, columns=cols)
        except Exception:
            df = pd.read_parquet(token_path)
            df = df[[col for col in cols if col in df.columns]].copy()
        df, summary = apply_context_filter_to_df(df, filter_info)
        df, token_filter_summary = apply_token_filter_to_df(df, token_filter_rows)
        summary.update(token_filter_summary)
        log(
            f"Context filter profile={profile}: rows {summary['rows_before_context_filter']:,} -> {summary['rows_after_context_filter']:,}"
        )
        if token_filter_rows:
            log(
                f"Token filter profile={profile}: excluded token rows "
                f"{token_filter_summary['excluded_token_rows']:,}"
            )
        df = df[df["token"].astype(str).isin(token_set)].copy()
        if df.empty:
            continue
        df["profile"] = profile
        df["token"] = df["token"].astype(str)
        merged = df.merge(profile_ann, on=["profile", "token"], how="inner")
        merged["data_scope"] = DATA_SCOPE
        merged["token_profile"] = merged["profile"]
        merged["profile_folder"] = profile_dir
        merged = assign_periods(merged, periods)
        mention_parts.append(merged)
    if not mention_parts:
        raise ValueError("No focus mentions were found in applied token files.")
    mentions = pd.concat(mention_parts, ignore_index=True)
    keep_cols = [
        "data_scope",
        "token_profile",
        "profile_folder",
        "token_uid",
        "context_uid",
        "article_uid",
        "article_id",
        "date",
        "token_order_in_context",
        "token",
        "pos",
        "token_source",
        "pos_source",
        "dict_lv1",
        "dict_lv2",
        "dict_ner_like_type",
        "focus_norm",
        "focus_label_zh",
        "focus_roots",
        "focus_roots_list",
        "confidence_score",
        "confidence_level",
        "review_flag",
        "review_priority",
        "anchor_group_candidate",
        "anchor_subgroup_candidate",
        "positive_signals",
        "negative_signals",
        "general_term_policy",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
    ]
    return mentions[[col for col in keep_cols if col in mentions.columns]].copy()


def add_global_mentions(mentions: pd.DataFrame, include_global: bool) -> pd.DataFrame:
    if not include_global:
        return mentions
    global_rows = mentions.copy()
    global_rows["period_set_id"] = "global"
    global_rows["period_id"] = "global"
    global_rows["period_label"] = "global"
    global_rows["period_sort_order"] = -1
    global_rows["period_start_date"] = pd.NA
    global_rows["period_end_date"] = pd.NA
    return pd.concat([global_rows, mentions], ignore_index=True)


def top_tokens_string(group: pd.DataFrame, topn: int) -> str:
    counts = group["token"].value_counts().head(topn)
    return "; ".join(f"{token}:{int(count)}" for token, count in counts.items())


def build_counts(mentions_with_global: pd.DataFrame, topn: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "data_scope",
        "token_profile",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
        "focus_norm",
        "focus_label_zh",
    ]
    counts = mentions_with_global.groupby(group_cols, dropna=False).agg(
        mention_count=("token_uid", "count"),
        distinct_context_count=("context_uid", "nunique"),
        distinct_article_count=("article_uid", "nunique"),
        distinct_token_count=("token", "nunique"),
        mean_confidence_score=("confidence_score", "mean"),
    ).reset_index()
    top = (
        mentions_with_global.groupby(group_cols, dropna=False)
        .apply(lambda g: top_tokens_string(g, topn), include_groups=False)
        .reset_index(name="top_tokens")
    )
    counts = counts.merge(top, on=group_cols, how="left")
    period_totals = counts.groupby(
        ["data_scope", "token_profile", "period_set_id", "period_id"], dropna=False
    )["mention_count"].sum().reset_index(name="period_focus_mention_count")
    counts = counts.merge(period_totals, on=["data_scope", "token_profile", "period_set_id", "period_id"], how="left")
    counts["focus_share_in_period"] = counts["mention_count"] / counts["period_focus_mention_count"].where(
        counts["period_focus_mention_count"].ne(0)
    )
    counts["_order"] = counts["focus_norm"].map({v: i for i, v in enumerate(FOCUS_ORDER)}).fillna(99)
    counts = counts.sort_values(["token_profile", "period_set_id", "period_sort_order", "_order"], kind="mergesort").drop(
        columns=["_order"]
    )

    token_cols = group_cols + [
        "token",
        "confidence_score",
        "confidence_level",
        "review_flag",
        "anchor_group_candidate",
        "anchor_subgroup_candidate",
        "positive_signals",
        "negative_signals",
    ]
    token_counts = mentions_with_global.groupby(token_cols, dropna=False).agg(
        mention_count=("token_uid", "count"),
        distinct_context_count=("context_uid", "nunique"),
        distinct_article_count=("article_uid", "nunique"),
    ).reset_index()
    token_counts["rank_within_focus_period"] = token_counts.groupby(group_cols, dropna=False)["mention_count"].rank(
        method="first", ascending=False
    ).astype(int)
    token_counts = token_counts.sort_values(
        ["token_profile", "period_set_id", "period_sort_order", "focus_norm", "rank_within_focus_period"],
        kind="mergesort",
    )
    return counts, token_counts


def build_representative_contexts(mentions_with_global: pd.DataFrame, per_focus_period: int = 5) -> pd.DataFrame:
    group_cols = [
        "data_scope",
        "token_profile",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "focus_norm",
        "focus_label_zh",
        "context_uid",
        "article_uid",
        "article_id",
        "date",
    ]
    context_rows = mentions_with_global.groupby(group_cols, dropna=False).agg(
        mention_count=("token_uid", "count"),
        distinct_token_count=("token", "nunique"),
        tokens=("token", lambda values: "; ".join(pd.Series(values).astype(str).value_counts().head(12).index)),
        max_confidence_score=("confidence_score", "max"),
    ).reset_index()
    context_rows = context_rows.sort_values(
        [
            "token_profile",
            "period_set_id",
            "period_sort_order",
            "focus_norm",
            "mention_count",
            "max_confidence_score",
        ],
        ascending=[True, True, True, True, False, False],
        kind="mergesort",
    )
    rank_cols = ["data_scope", "token_profile", "period_set_id", "period_id", "focus_norm"]
    context_rows["rank_within_focus_period"] = context_rows.groupby(rank_cols, dropna=False).cumcount() + 1
    return context_rows[context_rows["rank_within_focus_period"].le(per_focus_period)].copy()


def parse_network_file_meta(path: Path) -> Optional[Dict[str, object]]:
    name = path.name
    path_text = str(path)
    if "filtered_pre_zhixian_context" not in path_text:
        return None
    if "stopv5always" not in name:
        return None
    profile_match = re.search(r"applied_(regex-only|strict|full)", name)
    window_match = re.search(r"_w(\d+)", name)
    joint_match = re.search(r"joint(\d+)up", name)
    if not profile_match or not window_match or not joint_match:
        return None
    profile = normalize_profile(profile_match.group(1))
    window = int(window_match.group(1))
    threshold = int(joint_match.group(1))
    if "global_all" in name or "stopfiltered_global" in str(path.parent):
        period_set_id = "global"
        period_id = "global"
        period_label = "global"
        period_sort_order = -1
    else:
        period_match = re.search(r"(long_period_manual_p\d{3})", name) or re.search(
            r"stopfiltered_(long_period_manual_p\d{3})", str(path.parent)
        )
        if not period_match:
            return None
        period_id = period_match.group(1)
        period_set_id = "long_period_manual"
        period_label = period_id
        period_sort_order = int(period_id.rsplit("_p", 1)[1])
    return {
        "path": path,
        "profile": profile,
        "network_window": window,
        "source_joint_threshold": threshold,
        "period_set_id": period_set_id,
        "period_id": period_id,
        "period_label": period_label,
        "period_sort_order": period_sort_order,
    }


def select_network_files(network_root: Path, profiles: Sequence[str], windows: Sequence[int]) -> List[Dict[str, object]]:
    records: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for path in network_root.rglob("*.csv"):
        meta = parse_network_file_meta(path)
        if not meta:
            continue
        key = (str(meta["profile"]), int(meta["network_window"]), str(meta["period_id"]))
        if meta["profile"] not in profiles or meta["network_window"] not in windows:
            continue
        # Keep the lowest joint threshold for broader neighbor coverage.
        prev = records.get(key)
        if prev is None or int(meta["source_joint_threshold"]) < int(prev["source_joint_threshold"]):
            records[key] = meta
    return sorted(records.values(), key=lambda r: (str(r["profile"]), int(r["network_window"]), int(r["period_sort_order"])))


def network_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    cols = [
        "center_token",
        "neighbor_token",
        "direction",
        "joint_count",
        "raw_joint_event_count",
        "distinct_context_count",
        "distinct_article_count",
        "pmi",
        "ppmi",
        "avg_distance",
    ]
    for chunk in pd.read_csv(path, usecols=lambda col: col in cols, encoding="utf-8-sig", chunksize=chunksize):
        yield chunk


def build_network_edges(
    args: argparse.Namespace,
    annotation: pd.DataFrame,
    periods: pd.DataFrame,
    profiles: Sequence[str],
    windows: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    network_root = Path(args.network_root).expanduser().resolve()
    files = select_network_files(network_root, profiles, windows)
    if not files:
        log("No network files found for focus dashboard.")
        return pd.DataFrame(), pd.DataFrame()
    period_meta = {
        row["period_id"]: {
            "period_label": row.get("label", row["period_id"]),
            "period_start_date": row["start_date"].date().isoformat(),
            "period_end_date": row["end_date"].date().isoformat(),
        }
        for row in periods.to_dict("records")
    }
    ann_by_profile: Dict[str, pd.DataFrame] = {profile: annotation[annotation["profile"].eq(profile)].copy() for profile in profiles}
    token_sets: Dict[str, Set[str]] = {
        profile: set(df["token"].astype(str)) for profile, df in ann_by_profile.items() if not df.empty
    }
    token_edges: List[Dict[str, object]] = []
    for meta in files:
        profile = str(meta["profile"])
        token_set = token_sets.get(profile, set())
        if not token_set:
            continue
        path = Path(meta["path"])
        log(f"Scanning network: {path.name}")
        row_limit = int(args.max_network_rows_per_file or 0)
        seen_rows = 0
        for chunk in network_chunks(path, args.chunksize):
            if row_limit and seen_rows >= row_limit:
                break
            seen_rows += len(chunk)
            chunk["center_token"] = chunk["center_token"].astype(str)
            chunk["neighbor_token"] = chunk["neighbor_token"].astype(str)
            mask_center = chunk["center_token"].isin(token_set)
            mask_neighbor = chunk["neighbor_token"].isin(token_set)
            if not (mask_center.any() or mask_neighbor.any()):
                continue
            relevant = chunk[mask_center | mask_neighbor].copy()
            for side, source_col, alter_col in [
                ("center", "center_token", "neighbor_token"),
                ("neighbor", "neighbor_token", "center_token"),
            ]:
                side_df = relevant[relevant[source_col].isin(token_set)].copy()
                if side_df.empty:
                    continue
                side_df = side_df.merge(
                    ann_by_profile[profile],
                    left_on=source_col,
                    right_on="token",
                    how="inner",
                    suffixes=("", "_ann"),
                )
                for rec in side_df.to_dict("records"):
                    focus_token = str(rec[source_col])
                    alter_token = str(rec[alter_col])
                    token_edges.append(
                        {
                            "data_scope": DATA_SCOPE,
                            "token_profile": profile,
                            "network_window": int(meta["network_window"]),
                            "period_set_id": meta["period_set_id"],
                            "period_id": meta["period_id"],
                            "period_label": period_meta.get(meta["period_id"], {}).get("period_label", meta["period_label"]),
                            "period_sort_order": meta["period_sort_order"],
                            "period_start_date": period_meta.get(meta["period_id"], {}).get("period_start_date", pd.NA),
                            "period_end_date": period_meta.get(meta["period_id"], {}).get("period_end_date", pd.NA),
                            "focus_norm": rec["focus_norm"],
                            "focus_label_zh": rec["focus_label_zh"],
                            "focus_token": focus_token,
                            "alter_token": alter_token,
                            "focus_side": side,
                            "alter_is_focus_reference": alter_token in token_set,
                            "joint_count_undirected": float(rec.get("joint_count", 0) or 0),
                            "raw_joint_event_count_undirected": float(rec.get("raw_joint_event_count", 0) or 0),
                            "distinct_context_count_undirected": float(rec.get("distinct_context_count", 0) or 0),
                            "distinct_article_count_undirected": float(rec.get("distinct_article_count", 0) or 0),
                            "pmi_mean": float(rec.get("pmi", 0) or 0),
                            "ppmi_mean": float(rec.get("ppmi", 0) or 0),
                            "avg_distance_mean": float(rec.get("avg_distance", 0) or 0),
                            "source_file": str(path),
                            "source_file_name": path.name,
                            "source_joint_threshold": int(meta["source_joint_threshold"]),
                        }
                    )
    if not token_edges:
        return pd.DataFrame(), pd.DataFrame()
    token_df = pd.DataFrame(token_edges)
    group_cols = [
        "data_scope",
        "token_profile",
        "network_window",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
        "focus_norm",
        "focus_label_zh",
        "alter_token",
        "alter_is_focus_reference",
    ]
    focus_edges = token_df.groupby(group_cols, dropna=False).agg(
        joint_count_undirected_sum=("joint_count_undirected", "sum"),
        raw_joint_event_count_undirected_sum=("raw_joint_event_count_undirected", "sum"),
        distinct_context_count_undirected_sum=("distinct_context_count_undirected", "sum"),
        distinct_article_count_undirected_sum=("distinct_article_count_undirected", "sum"),
        pmi_mean=("pmi_mean", "mean"),
        ppmi_mean=("ppmi_mean", "mean"),
        avg_distance_mean=("avg_distance_mean", "mean"),
        distinct_focus_token_count=("focus_token", "nunique"),
        top_focus_tokens=("focus_token", lambda values: "; ".join(pd.Series(values).astype(str).value_counts().head(10).index)),
        source_file_count=("source_file_name", "nunique"),
        source_files=("source_file_name", lambda values: "; ".join(sorted(set(map(str, values)))[:4])),
    ).reset_index()
    den_cols = ["data_scope", "token_profile", "network_window", "period_set_id", "period_id", "focus_norm"]
    den = focus_edges.groupby(den_cols, dropna=False)["joint_count_undirected_sum"].sum().reset_index(
        name="focus_joint_total"
    )
    focus_edges = focus_edges.merge(den, on=den_cols, how="left")
    focus_edges["neighbor_share_within_focus"] = focus_edges["joint_count_undirected_sum"] / focus_edges[
        "focus_joint_total"
    ].where(focus_edges["focus_joint_total"].ne(0))
    focus_edges["rank_within_focus"] = focus_edges.groupby(den_cols, dropna=False)["joint_count_undirected_sum"].rank(
        method="first", ascending=False
    ).astype(int)
    focus_edges = focus_edges.sort_values(
        ["token_profile", "network_window", "period_sort_order", "focus_norm", "rank_within_focus"],
        kind="mergesort",
    )
    return focus_edges, token_df


def weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    sa = {k for k, v in a.items() if v > 0}
    sb = {k for k, v in b.items() if v > 0}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def build_overlap(focus_edges: pd.DataFrame, topn: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if focus_edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: List[Dict[str, object]] = []
    matrix_rows: List[Dict[str, object]] = []
    group_cols = ["data_scope", "token_profile", "network_window", "period_set_id", "period_id", "period_label"]
    for key, group in focus_edges.groupby(group_cols, dropna=False):
        vectors: Dict[str, Dict[str, float]] = {}
        labels: Dict[str, str] = {}
        for focus_norm, fg in group.groupby("focus_norm", dropna=False):
            top = fg.sort_values("rank_within_focus", kind="mergesort").head(topn)
            vectors[str(focus_norm)] = {
                str(row["alter_token"]): float(row["joint_count_undirected_sum"]) for row in top.to_dict("records")
            }
            labels[str(focus_norm)] = str(top["focus_label_zh"].iloc[0]) if not top.empty else FOCUS_LABELS.get(str(focus_norm), str(focus_norm))
        for a in FOCUS_ORDER:
            for b in FOCUS_ORDER:
                if a not in vectors or b not in vectors:
                    continue
                metrics = {
                    "weighted_jaccard": weighted_jaccard(vectors[a], vectors[b]),
                    "jaccard": jaccard(vectors[a], vectors[b]),
                    "cosine": cosine(vectors[a], vectors[b]),
                }
                shared = set(vectors[a]) & set(vectors[b])
                shared_top = sorted(shared, key=lambda tok: min(vectors[a][tok], vectors[b][tok]), reverse=True)[:20]
                row_base = dict(zip(group_cols, key))
                row_base.update(
                    {
                        "focus_norm_a": a,
                        "focus_label_a": labels.get(a, FOCUS_LABELS.get(a, a)),
                        "focus_norm_b": b,
                        "focus_label_b": labels.get(b, FOCUS_LABELS.get(b, b)),
                        "shared_neighbor_count": len(shared),
                        "shared_neighbors": "; ".join(shared_top),
                    }
                )
                row_base.update(metrics)
                rows.append(row_base)
                for metric, value in metrics.items():
                    matrix_row = dict(row_base)
                    matrix_row["metric"] = metric
                    matrix_row["value"] = value
                    matrix_rows.append(matrix_row)
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


def records(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records", force_ascii=False))


def render_base_html(title: str, body: str, script: str) -> str:
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    header {{ padding: 22px 28px 14px; background: #ffffff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 26px 0 12px; font-size: 19px; }}
    main {{ padding: 18px 28px 36px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 12px 0 18px; }}
    label {{ display: grid; gap: 4px; font-size: 12px; color: #475569; }}
    select, input {{ min-width: 140px; padding: 6px 8px; border: 1px solid #b8c2d1; border-radius: 4px; background: #fff; }}
    .panel {{ background: #fff; border: 1px solid #d8dee8; border-radius: 6px; padding: 14px; margin-bottom: 18px; }}
    .chart {{ width: 100%; height: 430px; }}
    .chart-tall {{ width: 100%; height: 600px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; background: #fff; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; position: sticky; top: 0; z-index: 1; }}
    .table-wrap {{ max-height: 460px; overflow: auto; border: 1px solid #d8dee8; border-radius: 4px; }}
    .note {{ color: #64748b; font-size: 12px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .summary div {{ background: #fff; border: 1px solid #d8dee8; border-radius: 6px; padding: 10px; }}
    .summary strong {{ display: block; font-size: 20px; margin-top: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <div class="note">Generated {esc(datetime.now().isoformat(timespec="seconds"))}</div>
</header>
<main>
{body}
</main>
<script>
{script}
</script>
</body>
</html>
"""
    return inject_controls_collapse(html)


def render_cta_dashboard(payload: Dict[str, object]) -> str:
    body = """
<section class="summary" id="summary"></section>
<section class="panel">
  <div class="controls">
    <label>Profile<select id="profile"></select></label>
    <label>Count Metric<select id="metric">
      <option value="distinct_context_count">distinct_context_count</option>
      <option value="mention_count">mention_count</option>
      <option value="distinct_article_count">distinct_article_count</option>
      <option value="distinct_token_count">distinct_token_count</option>
      <option value="focus_share_in_period">focus_share_in_period</option>
    </select></label>
  </div>
  <div id="period-chart" class="chart"></div>
</section>
<section class="panel">
  <h2>Top Tokens By Focus And Period</h2>
  <div class="controls">
    <label>Period<select id="period"></select></label>
    <label>Focus<select id="focus"></select></label>
  </div>
  <div class="table-wrap"><table id="token-table"></table></div>
</section>
<section class="panel">
  <h2>Representative Context Index</h2>
  <div class="table-wrap"><table id="context-table"></table></div>
</section>
"""
    script = f"""
const PAYLOAD = {json.dumps(payload, ensure_ascii=False)};
const focusOrder = {json.dumps(FOCUS_ORDER, ensure_ascii=False)};
const focusLabels = {json.dumps(FOCUS_LABELS, ensure_ascii=False)};
function uniq(arr) {{ return Array.from(new Set(arr.filter(x => x !== null && x !== undefined))); }}
function byId(id) {{ return document.getElementById(id); }}
function fillSelect(el, values, labels={{}}) {{
  el.innerHTML = values.map(v => `<option value="${{v}}">${{labels[v] || v}}</option>`).join('');
}}
function fmt(x) {{ if (x === null || x === undefined) return ''; if (typeof x === 'number') return Number.isInteger(x) ? String(x) : x.toFixed(4); return String(x); }}
function table(el, rows, cols) {{
  if (!rows.length) {{ el.innerHTML = '<tbody><tr><td>No rows</td></tr></tbody>'; return; }}
  el.innerHTML = '<thead><tr>' + cols.map(c=>`<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c=>`<td>${{fmt(r[c])}}</td>`).join('') + '</tr>').join('') + '</tbody>';
}}
function init() {{
  const counts = PAYLOAD.focus_period_counts;
  const tokens = PAYLOAD.focus_token_period_counts;
  const contexts = PAYLOAD.focus_representative_contexts;
  fillSelect(byId('profile'), uniq(counts.map(r=>r.token_profile)));
  fillSelect(byId('period'), uniq(tokens.map(r=>r.period_id)).sort((a,b)=> (a==='global'?-1:0) - (b==='global'?-1:0) || a.localeCompare(b)));
  fillSelect(byId('focus'), focusOrder, focusLabels);
  ['profile','metric','period','focus'].forEach(id => byId(id).addEventListener('change', draw));
  const s = PAYLOAD.summary.rows;
  byId('summary').innerHTML = [
    ['mentions', s.focus_mentions],
    ['period counts', s.focus_period_counts],
    ['token counts', s.focus_token_period_counts],
    ['network edges', s.focus_network_ego_edges]
  ].map(([k,v])=>`<div>${{k}}<strong>${{v}}</strong></div>`).join('');
  draw();
}}
function draw() {{
  const profile = byId('profile').value;
  const metric = byId('metric').value;
  const period = byId('period').value;
  const focus = byId('focus').value;
  const rows = PAYLOAD.focus_period_counts.filter(r => r.token_profile === profile && r.period_set_id !== 'global');
  const traces = focusOrder.map(f => {{
    const fr = rows.filter(r => r.focus_norm === f).sort((a,b)=>(a.period_sort_order||0)-(b.period_sort_order||0));
    return {{type:'scatter', mode:'lines+markers', name:focusLabels[f], x:fr.map(r=>r.period_label), y:fr.map(r=>r[metric] || 0),
      customdata: fr.map(r=>[r.period_id, r.top_tokens]), hovertemplate:'%{{fullData.name}}<br>%{{x}}<br>'+metric+': %{{y}}<br>%{{customdata[1]}}<extra></extra>'}};
  }});
  Plotly.react('period-chart', traces, {{margin:{{l:60,r:20,t:20,b:80}}, xaxis:{{title:'period'}}, yaxis:{{title:metric}}, legend:{{orientation:'h'}}}}, {{responsive:true}});
  const tokenRows = PAYLOAD.focus_token_period_counts
    .filter(r => r.token_profile === profile && r.period_id === period && r.focus_norm === focus)
    .sort((a,b)=>(a.rank_within_focus_period||999)-(b.rank_within_focus_period||999)).slice(0, 80);
  table(byId('token-table'), tokenRows, ['rank_within_focus_period','token','mention_count','distinct_context_count','distinct_article_count','confidence_score','review_flag','anchor_group_candidate','positive_signals']);
  const ctxRows = PAYLOAD.focus_representative_contexts
    .filter(r => r.token_profile === profile && r.period_id === period && r.focus_norm === focus)
    .sort((a,b)=>(a.rank_within_focus_period||999)-(b.rank_within_focus_period||999));
  table(byId('context-table'), ctxRows, ['rank_within_focus_period','date','article_id','context_uid','mention_count','tokens']);
}}
init();
"""
    return render_base_html("Focus CTA Dashboard", body, script)


def render_overlap_dashboard(payload: Dict[str, object]) -> str:
    body = """
<section class="panel">
  <div class="controls">
    <label>Profile<select id="profile"></select></label>
    <label>Window<select id="window"></select></label>
    <label>Period<select id="period"></select></label>
    <label>Metric<select id="metric">
      <option value="weighted_jaccard">weighted_jaccard</option>
      <option value="jaccard">jaccard</option>
      <option value="cosine">cosine</option>
    </select></label>
  </div>
  <div id="heatmap" class="chart"></div>
</section>
<section class="panel">
  <h2>Pairwise Overlap</h2>
  <div class="table-wrap"><table id="pair-table"></table></div>
</section>
"""
    script = f"""
const PAYLOAD = {json.dumps(payload, ensure_ascii=False)};
const focusOrder = {json.dumps(FOCUS_ORDER, ensure_ascii=False)};
const focusLabels = {json.dumps(FOCUS_LABELS, ensure_ascii=False)};
function uniq(arr) {{ return Array.from(new Set(arr.filter(x => x !== null && x !== undefined))); }}
function byId(id) {{ return document.getElementById(id); }}
function fillSelect(el, values, labels={{}}) {{ el.innerHTML = values.map(v => `<option value="${{v}}">${{labels[v] || v}}</option>`).join(''); }}
function fmt(x) {{ if (x === null || x === undefined) return ''; if (typeof x === 'number') return Number.isInteger(x) ? String(x) : x.toFixed(4); return String(x); }}
function table(el, rows, cols) {{ if (!rows.length) {{ el.innerHTML='<tbody><tr><td>No rows</td></tr></tbody>'; return; }} el.innerHTML='<thead><tr>'+cols.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${{fmt(r[c])}}</td>`).join('')+'</tr>').join('')+'</tbody>'; }}
function init() {{
 const rows = PAYLOAD.focus_network_neighbor_overlap;
 fillSelect(byId('profile'), uniq(rows.map(r=>r.token_profile)));
 fillSelect(byId('window'), uniq(rows.map(r=>r.network_window)).sort((a,b)=>a-b));
 fillSelect(byId('period'), uniq(rows.map(r=>r.period_id)).sort((a,b)=> (a==='global'?-1:0) - (b==='global'?-1:0) || a.localeCompare(b)));
 ['profile','window','period','metric'].forEach(id=>byId(id).addEventListener('change', draw));
 draw();
}}
function draw() {{
 const profile=byId('profile').value, win=+byId('window').value, period=byId('period').value, metric=byId('metric').value;
 const rows=PAYLOAD.focus_network_neighbor_overlap.filter(r=>r.token_profile===profile && +r.network_window===win && r.period_id===period);
 const z = focusOrder.map(a => focusOrder.map(b => {{
   const r = rows.find(x=>x.focus_norm_a===a && x.focus_norm_b===b);
   return r ? r[metric] : null;
 }}));
 Plotly.react('heatmap', [{{type:'heatmap', x:focusOrder.map(f=>focusLabels[f]), y:focusOrder.map(f=>focusLabels[f]), z:z, colorscale:'Blues', zmin:0, zmax:1, hoverongaps:false}}], {{margin:{{l:80,r:20,t:20,b:70}}, xaxis:{{side:'top'}}, yaxis:{{autorange:'reversed'}}}}, {{responsive:true}});
 table(byId('pair-table'), rows.filter(r=>r.focus_norm_a!==r.focus_norm_b).sort((a,b)=>(b[metric]||0)-(a[metric]||0)), ['focus_label_a','focus_label_b','weighted_jaccard','jaccard','cosine','shared_neighbor_count','shared_neighbors']);
}}
init();
"""
    return render_base_html("Focus Network Neighbor Overlap", body, script)


def render_multi_ego_dashboard(payload: Dict[str, object]) -> str:
    body = """
<section class="panel">
  <div class="controls">
    <label>Profile<select id="profile"></select></label>
    <label>Window<select id="window"></select></label>
    <label>Period<select id="period"></select></label>
    <label>Top N<input id="topn" type="number" min="5" max="200" value="40"></label>
    <label>Include Focus Neighbors<select id="includeFocus"><option value="false">false</option><option value="true">true</option></select></label>
  </div>
  <div id="ego" class="chart-tall"></div>
</section>
<section class="panel">
  <h2>Top Neighbors</h2>
  <div class="table-wrap"><table id="neighbor-table"></table></div>
</section>
"""
    script = f"""
const PAYLOAD = {json.dumps(payload, ensure_ascii=False)};
const focusOrder = {json.dumps(FOCUS_ORDER, ensure_ascii=False)};
const focusLabels = {json.dumps(FOCUS_LABELS, ensure_ascii=False)};
const colors = {{xianzheng:'#2563eb', lixian:'#dc2626', xianfa:'#16a34a', zhixian:'#9333ea'}};
function uniq(arr) {{ return Array.from(new Set(arr.filter(x => x !== null && x !== undefined))); }}
function byId(id) {{ return document.getElementById(id); }}
function fillSelect(el, values, labels={{}}) {{ el.innerHTML = values.map(v => `<option value="${{v}}">${{labels[v] || v}}</option>`).join(''); }}
function fmt(x) {{ if (x === null || x === undefined) return ''; if (typeof x === 'number') return Number.isInteger(x) ? String(x) : x.toFixed(4); return String(x); }}
function table(el, rows, cols) {{ if (!rows.length) {{ el.innerHTML='<tbody><tr><td>No rows</td></tr></tbody>'; return; }} el.innerHTML='<thead><tr>'+cols.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${{fmt(r[c])}}</td>`).join('')+'</tr>').join('')+'</tbody>'; }}
function init() {{
 const rows=PAYLOAD.focus_network_ego_edges;
 fillSelect(byId('profile'), uniq(rows.map(r=>r.token_profile)));
 fillSelect(byId('window'), uniq(rows.map(r=>r.network_window)).sort((a,b)=>a-b));
 fillSelect(byId('period'), uniq(rows.map(r=>r.period_id)).sort((a,b)=> (a==='global'?-1:0) - (b==='global'?-1:0) || a.localeCompare(b)));
 ['profile','window','period','topn','includeFocus'].forEach(id=>byId(id).addEventListener('change', draw));
 draw();
}}
function draw() {{
 const profile=byId('profile').value, win=+byId('window').value, period=byId('period').value, topn=+byId('topn').value, includeFocus=byId('includeFocus').value==='true';
 let rows=PAYLOAD.focus_network_ego_edges.filter(r=>r.token_profile===profile && +r.network_window===win && r.period_id===period);
 if (!includeFocus) rows = rows.filter(r=>!r.alter_is_focus_reference);
 const selected=[];
 focusOrder.forEach(f => selected.push(...rows.filter(r=>r.focus_norm===f).sort((a,b)=>(a.rank_within_focus||999)-(b.rank_within_focus||999)).slice(0, topn)));
 const nodes = new Map();
 focusOrder.forEach((f,i)=>nodes.set('focus:'+f, {{id:'focus:'+f, label:focusLabels[f], x:-1, y:i*2, color:colors[f], size:24, type:'focus'}}));
 selected.forEach(r=>{{ const key='n:'+r.alter_token; if(!nodes.has(key)) nodes.set(key, {{id:key, label:r.alter_token, x:1, y:0, color:'#64748b', size:8, type:'neighbor', weight:0}}); nodes.get(key).weight += r.joint_count_undirected_sum || 0; }});
 const neighbors = Array.from(nodes.values()).filter(n=>n.type==='neighbor').sort((a,b)=>b.weight-a.weight);
 neighbors.forEach((n,i)=>{{ n.y = (i % Math.max(1, Math.ceil(neighbors.length/3))) * 0.45; n.x = 0.8 + Math.floor(i / Math.max(1, Math.ceil(neighbors.length/3))) * 0.55; n.size = Math.max(6, Math.min(18, 5 + Math.sqrt(n.weight))); }});
 const edgeX=[], edgeY=[];
 selected.forEach(r=>{{ const a=nodes.get('focus:'+r.focus_norm), b=nodes.get('n:'+r.alter_token); if(a&&b){{edgeX.push(a.x,b.x,null); edgeY.push(a.y,b.y,null);}} }});
 const nodeList=Array.from(nodes.values());
 const traces=[
   {{type:'scatter', mode:'lines', x:edgeX, y:edgeY, line:{{color:'rgba(100,116,139,.25)', width:1}}, hoverinfo:'skip', showlegend:false}},
   {{type:'scatter', mode:'markers+text', x:nodeList.map(n=>n.x), y:nodeList.map(n=>n.y), text:nodeList.map(n=>n.label), textposition:'top center', marker:{{size:nodeList.map(n=>n.size), color:nodeList.map(n=>n.color), line:{{color:'#fff', width:1}}}}, hovertemplate:'%{{text}}<extra></extra>', showlegend:false}}
 ];
 Plotly.react('ego', traces, {{margin:{{l:20,r:20,t:20,b:20}}, xaxis:{{visible:false}}, yaxis:{{visible:false}}, plot_bgcolor:'#fff'}}, {{responsive:true}});
 table(byId('neighbor-table'), selected.sort((a,b)=>(a.focus_norm.localeCompare(b.focus_norm)) || (a.rank_within_focus-b.rank_within_focus)).slice(0, 500), ['focus_label_zh','alter_token','rank_within_focus','joint_count_undirected_sum','neighbor_share_within_focus','ppmi_mean','top_focus_tokens','alter_is_focus_reference']);
}}
init();
"""
    return render_base_html("Focus Multi-Ego Network Dashboard", body, script)


def main() -> None:
    args = parse_args()
    profiles = [normalize_profile(value) for value in parse_csv_list(args.profiles)]
    windows = parse_int_list(args.windows)
    output_dir = Path(args.output_dir).expanduser().resolve()
    html_dir = output_dir / "html"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    log("Loading periods")
    periods = load_periods(Path(args.periods_parquet).expanduser().resolve(), args.period_set_id)
    log("Loading focus annotation")
    annotation = load_annotation(args, profiles)
    annotation.to_csv(output_dir / "focus_annotation_seed_tokens.csv", index=False, encoding="utf-8")

    log("Building focus mentions")
    mentions = load_mentions(args, annotation, periods, profiles)
    mentions.to_parquet(output_dir / "focus_context_mentions.parquet", index=False)
    mentions_with_global = add_global_mentions(mentions, args.include_global)
    period_counts, token_counts = build_counts(mentions_with_global, args.top_tokens_per_focus)
    reps = build_representative_contexts(mentions_with_global)
    period_counts.to_csv(output_dir / "focus_period_counts.csv", index=False, encoding="utf-8")
    token_counts.to_csv(output_dir / "focus_token_period_counts.csv", index=False, encoding="utf-8")
    reps.to_csv(output_dir / "focus_representative_contexts.csv", index=False, encoding="utf-8")

    log("Building focus network summaries")
    focus_edges, token_edges = build_network_edges(args, annotation, periods, profiles, windows)
    if not focus_edges.empty:
        focus_edges.to_csv(output_dir / "focus_network_ego_edges.csv", index=False, encoding="utf-8")
        token_edges.to_csv(output_dir / "focus_network_token_ego_edges.csv", index=False, encoding="utf-8")
        top_neighbors = focus_edges[focus_edges["rank_within_focus"].le(args.top_neighbors)].copy()
        top_neighbors.to_csv(output_dir / "focus_network_top_neighbors.csv", index=False, encoding="utf-8")
        overlap, overlap_matrix = build_overlap(focus_edges, args.overlap_topn)
        overlap.to_csv(output_dir / "focus_network_neighbor_overlap.csv", index=False, encoding="utf-8")
        overlap_matrix.to_csv(output_dir / "focus_network_neighbor_overlap_matrix.csv", index=False, encoding="utf-8")
    else:
        top_neighbors = pd.DataFrame()
        overlap = pd.DataFrame()
        overlap_matrix = pd.DataFrame()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_csv": str(Path(args.annotation_csv).expanduser().resolve()),
        "tokens_root": str(Path(args.tokens_root).expanduser().resolve()),
        "network_root": str(Path(args.network_root).expanduser().resolve()),
        "period_set_id": args.period_set_id,
        "profiles": profiles,
        "windows": windows,
        "include_mode": args.include_mode,
        "include_review_flags": args.include_review_flags,
        "rows": {
            "annotation_seed_tokens": int(len(annotation)),
            "focus_mentions": int(len(mentions)),
            "focus_period_counts": int(len(period_counts)),
            "focus_token_period_counts": int(len(token_counts)),
            "focus_representative_contexts": int(len(reps)),
            "focus_network_ego_edges": int(len(focus_edges)),
            "focus_network_token_ego_edges": int(len(token_edges)),
            "focus_network_neighbor_overlap": int(len(overlap)),
        },
    }
    write_json(output_dir / "focus_dashboard_summary.json", summary)

    if args.write_html:
        log("Rendering HTML dashboards")
        payload = {
            "summary": summary,
            "focus_period_counts": records(period_counts),
            "focus_token_period_counts": records(token_counts[token_counts["rank_within_focus_period"].le(120)]),
            "focus_representative_contexts": records(reps),
            "focus_network_ego_edges": records(top_neighbors),
            "focus_network_neighbor_overlap": records(overlap),
        }
        (html_dir / "focus_cta_dashboard.html").write_text(render_cta_dashboard(payload), encoding="utf-8", newline="\n")
        (html_dir / "focus_network_neighbor_overlap_dashboard.html").write_text(
            render_overlap_dashboard(payload), encoding="utf-8", newline="\n"
        )
        (html_dir / "focus_multi_ego_network_dashboard.html").write_text(
            render_multi_ego_dashboard(payload), encoding="utf-8", newline="\n"
        )
    log(f"Wrote focus dashboard outputs: {output_dir}")


if __name__ == "__main__":
    main()
