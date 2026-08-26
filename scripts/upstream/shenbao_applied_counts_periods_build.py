#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
SHENBAO_DIR = ROOT_DIR / "shenbao"
DEFAULT_APPLIED_TOKENS_ROOT = SHENBAO_DIR / "shenbao_network" / "applied_tokens"
DEFAULT_PROFILES = ("regex_only", "strict", "full")


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
        description=(
            "Rebuild counts and period metric tables for applied token parquet files "
            "(regex_only, strict, full) under shenbao_network/applied_tokens."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(DEFAULT_PROFILES),
        default=list(DEFAULT_PROFILES),
        help="Applied token profiles to process. Default: regex_only strict full.",
    )
    parser.add_argument(
        "--applied-tokens-root",
        default=str(DEFAULT_APPLIED_TOKENS_ROOT),
        help="Root directory containing applied token profile folders.",
    )
    parser.add_argument(
        "--reference-counts-daily-csv",
        default="",
        help="Optional reference daily_counts.csv. Empty uses counts/daily_counts.csv under the selected applied-token reference profile.",
    )
    parser.add_argument(
        "--reference-periods-parquet",
        default="",
        help="Optional reference periods.parquet. Empty uses periods/periods.parquet under the selected applied-token reference profile.",
    )
    parser.add_argument(
        "--reference-period-sets-parquet",
        default="",
        help="Optional reference period_sets.parquet. Empty uses periods/period_sets.parquet under the selected applied-token reference profile.",
    )
    parser.add_argument(
        "--reference-burst-overlays-parquet",
        default="",
        help="Optional reference burst_overlays.parquet. Empty uses periods/burst_overlays.parquet under the selected applied-token reference profile.",
    )
    parser.add_argument(
        "--pause-every-contexts",
        type=int,
        default=10000,
        help="Pause briefly after processing this many contexts. Default: 10000.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.05,
        help="Sleep duration for throttling between context batches. Default: 0.05.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV: {path}")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_reference_profile_dir(
    applied_tokens_root: Path,
    preferred_profiles: Sequence[str],
) -> Path:
    candidate_profiles: List[str] = []
    for profile in preferred_profiles:
        if profile not in candidate_profiles:
            candidate_profiles.append(profile)
    for profile in DEFAULT_PROFILES:
        if profile not in candidate_profiles:
            candidate_profiles.append(profile)

    for profile in candidate_profiles:
        profile_dir = applied_tokens_root / profile
        if (
            (profile_dir / "counts" / "daily_counts.csv").exists()
            and (profile_dir / "periods" / "periods.parquet").exists()
            and (profile_dir / "periods" / "period_sets.parquet").exists()
            and (profile_dir / "periods" / "burst_overlays.parquet").exists()
        ):
            return profile_dir

    raise FileNotFoundError(
        "No applied-token reference set found under "
        f"{applied_tokens_root}. Expected counts/daily_counts.csv and periods/*.parquet."
    )


def compute_adjacency_event_count_by_date(
    tokens_df: pd.DataFrame,
    pause_every_contexts: int,
    pause_seconds: float,
) -> Counter:
    ordered = tokens_df[
        ["context_uid", "date", "token_order_in_context", "is_punctuation", "is_boundary"]
    ].sort_values(["context_uid", "token_order_in_context"], kind="mergesort")

    context_ids = ordered["context_uid"].to_numpy()
    dates = ordered["date"].to_numpy()
    is_punctuation = ordered["is_punctuation"].to_numpy(dtype=bool)
    is_boundary = ordered["is_boundary"].to_numpy(dtype=bool)

    if len(context_ids) == 0:
        return Counter()

    group_starts = np.r_[0, np.flatnonzero(context_ids[1:] != context_ids[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(context_ids)]

    event_count_by_date: Counter = Counter()

    for group_idx, (start, end) in enumerate(zip(group_starts, group_ends), start=1):
        punct = is_punctuation[start:end]
        boundary = is_boundary[start:end]
        n = end - start

        left_hits = np.zeros(n, dtype=np.int16)
        right_hits = np.zeros(n, dtype=np.int16)

        seen_nonpunct_left = False
        for i in range(n):
            if boundary[i]:
                seen_nonpunct_left = False
            if not punct[i]:
                left_hits[i] = 1 if seen_nonpunct_left else 0
                seen_nonpunct_left = True

        seen_nonpunct_right = False
        for i in range(n - 1, -1, -1):
            if boundary[i]:
                seen_nonpunct_right = False
            if not punct[i]:
                right_hits[i] = 1 if seen_nonpunct_right else 0
                seen_nonpunct_right = True

        event_count = int(left_hits.sum() + right_hits.sum())
        event_count_by_date[dates[start]] += event_count

        if pause_every_contexts > 0 and pause_seconds > 0 and group_idx % pause_every_contexts == 0:
            time.sleep(pause_seconds)

    return event_count_by_date


def build_daily_monthly_cumulative_counts(
    tokens_df: pd.DataFrame,
    calendar_dates: Sequence[str],
    pause_every_contexts: int,
    pause_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    article_count_by_date = (
        tokens_df[["date", "article_uid"]]
        .drop_duplicates()
        .groupby("date")
        .size()
        .to_dict()
    )
    context_count_by_date = (
        tokens_df[["date", "context_uid"]]
        .drop_duplicates()
        .groupby("date")
        .size()
        .to_dict()
    )
    token_count_by_date = tokens_df.groupby("date").size().to_dict()
    event_count_by_date = compute_adjacency_event_count_by_date(
        tokens_df=tokens_df,
        pause_every_contexts=pause_every_contexts,
        pause_seconds=pause_seconds,
    )

    daily_rows: List[dict] = []
    for ds in calendar_dates:
        daily_rows.append(
            {
                "date": ds,
                "article_count": int(article_count_by_date.get(ds, 0)),
                "context_count": int(context_count_by_date.get(ds, 0)),
                "token_count": int(token_count_by_date.get(ds, 0)),
                "adjacency_event_count": int(event_count_by_date.get(ds, 0)),
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    daily_df["year_month"] = daily_df["date"].str[:7]
    monthly_df = (
        daily_df.groupby("year_month", as_index=False)[
            ["article_count", "context_count", "token_count", "adjacency_event_count"]
        ]
        .sum()
    )

    cumulative_df = daily_df[
        ["date", "article_count", "context_count", "token_count", "adjacency_event_count"]
    ].copy()
    cumulative_df["cumulative_article_count"] = cumulative_df["article_count"].cumsum()
    cumulative_df["cumulative_context_count"] = cumulative_df["context_count"].cumsum()
    cumulative_df["cumulative_token_count"] = cumulative_df["token_count"].cumsum()
    cumulative_df["cumulative_adjacency_event_count"] = cumulative_df[
        "adjacency_event_count"
    ].cumsum()
    cumulative_df = cumulative_df[
        [
            "date",
            "cumulative_article_count",
            "cumulative_context_count",
            "cumulative_token_count",
            "cumulative_adjacency_event_count",
        ]
    ]

    return daily_df.drop(columns=["year_month"]), monthly_df, cumulative_df


def rebuild_periods_table(raw_periods_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    metric_index = daily_df.set_index("date")[
        ["article_count", "context_count", "token_count", "adjacency_event_count"]
    ]

    rebuilt_rows: List[dict] = []
    for row in raw_periods_df.to_dict("records"):
        start_date = row["start_date"]
        end_date = row["end_date"]
        metric_slice = metric_index.loc[start_date:end_date]
        rebuilt = dict(row)
        rebuilt["article_count"] = int(metric_slice["article_count"].sum())
        rebuilt["context_count"] = int(metric_slice["context_count"].sum())
        rebuilt["token_count"] = int(metric_slice["token_count"].sum())
        rebuilt["adjacency_event_count"] = int(metric_slice["adjacency_event_count"].sum())
        rebuilt_rows.append(rebuilt)
    return pd.DataFrame(rebuilt_rows)


def write_metadata(
    path: Path,
    profile: str,
    tokens_path: Path,
    row_count: int,
    daily_df: pd.DataFrame,
    periods_df: pd.DataFrame,
    pause_every_contexts: int,
    pause_seconds: float,
) -> None:
    payload = {
        "profile": profile,
        "tokens_parquet": str(tokens_path),
        "token_row_count": int(row_count),
        "daily_row_count": int(len(daily_df)),
        "period_row_count": int(len(periods_df)),
        "token_count_total": int(daily_df["token_count"].sum()),
        "adjacency_event_count_total": int(daily_df["adjacency_event_count"].sum()),
        "pause_every_contexts": int(pause_every_contexts),
        "pause_seconds": float(pause_seconds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_if_different(src: Path, dst: Path) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve() if dst.exists() else dst.resolve(strict=False)
    if src_resolved == dst_resolved:
        return
    shutil.copy2(src, dst)


def process_profile(
    profile: str,
    applied_tokens_root: Path,
    calendar_dates: Sequence[str],
    raw_periods_df: pd.DataFrame,
    raw_period_sets_parquet: Path,
    raw_burst_overlays_parquet: Path,
    pause_every_contexts: int,
    pause_seconds: float,
) -> None:
    profile_dir = applied_tokens_root / profile
    tokens_path = profile_dir / "tokens.parquet"
    ensure_file(tokens_path, f"{profile} tokens.parquet")

    counts_dir = profile_dir / "counts"
    periods_dir = profile_dir / "periods"

    tokens_df = pd.read_parquet(
        tokens_path,
        columns=[
            "date",
            "article_uid",
            "context_uid",
            "token_order_in_context",
            "is_punctuation",
            "is_boundary",
        ],
    )

    daily_df, monthly_df, cumulative_df = build_daily_monthly_cumulative_counts(
        tokens_df=tokens_df,
        calendar_dates=calendar_dates,
        pause_every_contexts=pause_every_contexts,
        pause_seconds=pause_seconds,
    )

    write_csv(counts_dir / "daily_counts.csv", daily_df)
    write_csv(counts_dir / "monthly_counts.csv", monthly_df)
    write_csv(counts_dir / "cumulative_counts.csv", cumulative_df)

    rebuilt_periods_df = rebuild_periods_table(raw_periods_df=raw_periods_df, daily_df=daily_df)
    periods_dir.mkdir(parents=True, exist_ok=True)
    rebuilt_periods_df.to_parquet(periods_dir / "periods.parquet", index=False)

    copy_if_different(raw_period_sets_parquet, periods_dir / "period_sets.parquet")
    copy_if_different(raw_burst_overlays_parquet, periods_dir / "burst_overlays.parquet")

    write_metadata(
        path=profile_dir / "counts_periods_build_metadata.json",
        profile=profile,
        tokens_path=tokens_path,
        row_count=len(tokens_df),
        daily_df=daily_df,
        periods_df=rebuilt_periods_df,
        pause_every_contexts=pause_every_contexts,
        pause_seconds=pause_seconds,
    )


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()

    applied_tokens_root = Path(args.applied_tokens_root).expanduser().resolve()
    reference_profile_dir = find_reference_profile_dir(
        applied_tokens_root=applied_tokens_root,
        preferred_profiles=args.profiles,
    )
    reference_counts_daily_csv = (
        Path(args.reference_counts_daily_csv).expanduser().resolve()
        if args.reference_counts_daily_csv
        else reference_profile_dir / "counts" / "daily_counts.csv"
    )
    reference_periods_parquet = (
        Path(args.reference_periods_parquet).expanduser().resolve()
        if args.reference_periods_parquet
        else reference_profile_dir / "periods" / "periods.parquet"
    )
    reference_period_sets_parquet = (
        Path(args.reference_period_sets_parquet).expanduser().resolve()
        if args.reference_period_sets_parquet
        else reference_profile_dir / "periods" / "period_sets.parquet"
    )
    reference_burst_overlays_parquet = (
        Path(args.reference_burst_overlays_parquet).expanduser().resolve()
        if args.reference_burst_overlays_parquet
        else reference_profile_dir / "periods" / "burst_overlays.parquet"
    )

    ensure_file(reference_counts_daily_csv, "reference daily_counts.csv")
    ensure_file(reference_periods_parquet, "reference periods.parquet")
    ensure_file(reference_period_sets_parquet, "reference period_sets.parquet")
    ensure_file(reference_burst_overlays_parquet, "reference burst_overlays.parquet")

    raw_daily_df = read_csv_rows(reference_counts_daily_csv)
    raw_periods_df = pd.read_parquet(reference_periods_parquet)
    calendar_dates = raw_daily_df["date"].tolist()

    for profile in args.profiles:
        process_profile(
            profile=profile,
            applied_tokens_root=applied_tokens_root,
            calendar_dates=calendar_dates,
            raw_periods_df=raw_periods_df,
            raw_period_sets_parquet=reference_period_sets_parquet,
            raw_burst_overlays_parquet=reference_burst_overlays_parquet,
            pause_every_contexts=args.pause_every_contexts,
            pause_seconds=args.pause_seconds,
        )
        print(f"done: {profile}")


if __name__ == "__main__":
    main()
