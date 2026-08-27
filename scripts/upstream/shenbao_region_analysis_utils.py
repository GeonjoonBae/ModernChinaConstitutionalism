#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
SHENBAO_DIR = ROOT / "shenbao"
DEFAULT_FOREIGN_REGION_DIR = SHENBAO_DIR / "shenbao_interpretation" / "foreign_regions"
DEFAULT_MENTIONS_PARQUET = DEFAULT_FOREIGN_REGION_DIR / "context_region_mentions.parquet"
DEFAULT_PERIODS_PARQUET = (
    SHENBAO_DIR / "shenbao_network" / "applied_tokens" / "full" / "periods" / "periods.parquet"
)

PROFILE_TO_CODE = {
    "regex-only": "ro",
    "regex_only": "ro",
    "strict": "st",
    "full": "fu",
}
CODE_TO_PROFILE = {
    "ro": "regex-only",
    "st": "strict",
    "fu": "full",
}
PERIOD_KEY_COLUMNS = [
    "period_set_id",
    "period_id",
    "period_label",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
]


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def log(message: str) -> None:
    print(message, flush=True)


def parse_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            parts.extend(parse_list(item))
        return parts
    return [part.strip() for part in re.split(r"[,\s]+", str(value).strip()) if part.strip()]


def parse_int_list(value: object) -> List[int]:
    values = [int(part) for part in parse_list(value)]
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected positive integer values.")
    return values


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false: {value}")


def normalize_profile(value: object) -> str:
    text = str(value).strip()
    if text == "regex_only":
        return "regex-only"
    return text


def normalize_profiles(values: Sequence[str]) -> List[str]:
    return [normalize_profile(value) for value in values]


def profile_code(value: str) -> str:
    normalized = normalize_profile(value)
    if normalized not in PROFILE_TO_CODE:
        raise ValueError(f"Unsupported token profile: {value}")
    return PROFILE_TO_CODE[normalized]


def profile_from_filename(name: str) -> Optional[str]:
    match = re.search(r"_(ro|st|fu)_", name)
    if not match:
        return None
    return CODE_TO_PROFILE[match.group(1)]


def slug_profile(value: str) -> str:
    return normalize_profile(value).replace("-", "_")


def sanitize_slug(value: object) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "value"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    raise_csv_field_limit()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log(f"Wrote {path} ({len(df):,} rows)")


def write_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    log(f"Wrote {path}")


def normalize_period_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in PERIOD_KEY_COLUMNS:
        if column not in out.columns:
            continue
        if column == "period_sort_order":
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = numeric.round().astype("Int64").astype("string")
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
        else:
            out[column] = out[column].astype("string")
        out[column] = out[column].fillna("").replace({"<NA>": "", "nan": "", "NaT": "", "None": ""})
    return out


def load_periods(path: Path = DEFAULT_PERIODS_PARQUET, period_set_id: str = "long_period_manual") -> pd.DataFrame:
    periods = pd.read_parquet(path)
    required = ["period_set_id", "period_id", "label", "sort_order", "start_date", "end_date"]
    missing = [column for column in required if column not in periods.columns]
    if missing:
        raise ValueError(f"Periods parquet is missing columns: {missing}")
    periods = periods[periods["period_set_id"].astype(str).eq(period_set_id)].copy()
    periods = periods.rename(
        columns={
            "label": "period_label",
            "sort_order": "period_sort_order",
            "start_date": "period_start_date",
            "end_date": "period_end_date",
        }
    )
    periods = periods[
        [
            "period_set_id",
            "period_id",
            "period_label",
            "period_sort_order",
            "period_start_date",
            "period_end_date",
        ]
    ]
    periods = normalize_period_key_columns(periods)
    return periods.sort_values(["period_sort_order", "period_id"], kind="mergesort")


def period_id_from_unit_suffix(name: str, periods: pd.DataFrame, period_set_id: str) -> Optional[Dict[str, object]]:
    match = re.search(r"_up(\d+)\.kv$", name)
    if not match:
        return None
    sort_order = int(match.group(1))
    row = periods.loc[periods["period_sort_order"].astype("int64").eq(sort_order)]
    if row.empty:
        return {
            "period_set_id": period_set_id,
            "period_id": f"{period_set_id}_p{sort_order:03d}",
            "period_label": "",
            "period_sort_order": sort_order,
            "period_start_date": "",
            "period_end_date": "",
        }
    item = row.iloc[0]
    return {
        "period_set_id": period_set_id,
        "period_id": item["period_id"],
        "period_label": item["period_label"],
        "period_sort_order": item["period_sort_order"],
        "period_start_date": item["period_start_date"],
        "period_end_date": item["period_end_date"],
    }


def load_mentions(
    path: Path,
    data_scopes: Sequence[str],
    profiles: Sequence[str],
    analysis_buckets: Sequence[str],
    reference_statuses: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mentions parquet not found: {path}")
    mentions = pd.read_parquet(path)
    mentions["token_profile"] = mentions["token_profile"].map(normalize_profile)
    if data_scopes:
        mentions = mentions[mentions["data_scope"].astype(str).isin(set(data_scopes))].copy()
    if profiles:
        mentions = mentions[mentions["token_profile"].isin(set(normalize_profiles(profiles)))].copy()
    if analysis_buckets:
        mentions = mentions[mentions["analysis_bucket"].astype(str).isin(set(analysis_buckets))].copy()
    if reference_statuses:
        mentions = mentions[
            mentions["foreign_reference_status"].astype(str).isin(set(reference_statuses))
        ].copy()
    return normalize_period_key_columns(mentions)


def add_global_mentions_from_period_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    if mentions.empty or "period_set_id" not in mentions.columns:
        return mentions
    if mentions["period_set_id"].astype(str).eq("global").any():
        return mentions
    global_rows = mentions.copy()
    global_rows["period_set_id"] = "global"
    global_rows["period_id"] = "global"
    global_rows["period_label"] = "global"
    global_rows["period_sort_order"] = -1
    global_rows["period_start_date"] = ""
    global_rows["period_end_date"] = ""
    return normalize_period_key_columns(pd.concat([mentions, global_rows], ignore_index=True))


def make_context_weights(mentions: pd.DataFrame, weighting_mode: str) -> pd.DataFrame:
    if mentions.empty:
        return pd.DataFrame()
    mentions = normalize_period_key_columns(mentions)
    group_cols = [
        "data_scope",
        "token_profile",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
        "region_norm",
        "macro_region",
        "context_uid",
    ]
    optional = ["article_uid", "article_id", "date"]
    group_cols.extend([column for column in optional if column in mentions.columns])
    weights = mentions.groupby(group_cols, dropna=False).agg(
        mention_count_in_context=("token_uid", "count"),
        distinct_token_count_in_context=("token", "nunique"),
        region_tokens_in_context=("token", lambda values: "|".join(sorted(set(map(str, values))))),
    ).reset_index()
    if weighting_mode == "context_once":
        weights["region_weight"] = 1.0
    elif weighting_mode == "mention_count":
        weights["region_weight"] = weights["mention_count_in_context"].astype(float)
    else:
        raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")
    return weights


def component_columns(df: pd.DataFrame, prefix: str = "svd_") -> List[str]:
    return [column for column in df.columns if re.fullmatch(rf"{re.escape(prefix)}\d+", str(column))]


def ensure_global_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_set_id"] = "global"
    df["period_id"] = "global"
    df["period_label"] = "global"
    df["period_sort_order"] = -1
    df["period_start_date"] = ""
    df["period_end_date"] = ""
    return normalize_period_key_columns(df)


def assign_periods_by_date(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    frames = []
    for _, period in periods.iterrows():
        start = pd.to_datetime(period["period_start_date"], errors="coerce")
        end = pd.to_datetime(period["period_end_date"], errors="coerce")
        mask = out["date_dt"].ge(start) & out["date_dt"].le(end)
        if not mask.any():
            continue
        chunk = out.loc[mask].copy()
        for column in [
            "period_set_id",
            "period_id",
            "period_label",
            "period_sort_order",
            "period_start_date",
            "period_end_date",
        ]:
            chunk[column] = period[column]
        frames.append(chunk.drop(columns=["date_dt"]))
    if not frames:
        return normalize_period_key_columns(out.drop(columns=["date_dt"]))
    return normalize_period_key_columns(pd.concat(frames, ignore_index=True))


def weighted_mean(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    weight_col: str,
    output_col: str,
    denominator: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    temp = df[list(group_cols) + [value_col, weight_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce").fillna(0.0)
    temp[weight_col] = pd.to_numeric(temp[weight_col], errors="coerce").fillna(0.0)
    temp["_weighted_value"] = temp[value_col] * temp[weight_col]
    numerator = temp.groupby(list(group_cols), dropna=False)["_weighted_value"].sum().reset_index()
    if denominator is None:
        denominator = temp.groupby(list(group_cols), dropna=False)[weight_col].sum().reset_index(name="_denominator")
    out = numerator.merge(denominator, on=list(group_cols), how="left")
    out[output_col] = out["_weighted_value"] / out["_denominator"].where(out["_denominator"].ne(0))
    return out.drop(columns=["_weighted_value", "_denominator"])


def top_values(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    label_col: str,
    n: int,
    fmt: str = "{label}:{value}",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(group_cols) + [label_col])
    temp = df.sort_values(list(group_cols) + [value_col], ascending=[True] * len(group_cols) + [False])
    temp["_rank"] = temp.groupby(list(group_cols), dropna=False).cumcount() + 1
    temp = temp[temp["_rank"].le(n)].copy()
    temp["_item"] = [
        fmt.format(label=row[label_col], value=row[value_col], rank=row["_rank"])
        for _, row in temp.iterrows()
    ]
    return temp.groupby(list(group_cols), dropna=False)["_item"].agg("; ".join).reset_index()


def parse_window_from_name(name: str) -> Optional[int]:
    match = re.search(r"_w(\d+)_", name)
    if not match:
        return None
    return int(match.group(1))


def has_all_substrings(name: str, substrings: Sequence[str]) -> bool:
    return all(item in name for item in substrings if item)


def has_no_substrings(name: str, substrings: Sequence[str]) -> bool:
    return not any(item in name for item in substrings if item)
