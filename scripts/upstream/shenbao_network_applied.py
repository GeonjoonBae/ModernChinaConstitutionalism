#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq

from shenbao_context_filter_utils import (
    DEFAULT_CONTEXT_FILTER_NAME,
    DEFAULT_FILTER_ROOT,
    context_filter_stem_part,
    load_context_filter,
)


ROOT = Path(__file__).resolve().parent
SHENBAO_DIR = ROOT / "shenbao"
DEFAULT_APPLIED_TOKENS_DIR = SHENBAO_DIR / "shenbao_network" / "applied_tokens"
DEFAULT_TOKENS_PARQUET = DEFAULT_APPLIED_TOKENS_DIR / "regex_only" / "tokens.parquet"
DEFAULT_PERIODS_PARQUET = DEFAULT_APPLIED_TOKENS_DIR / "regex_only" / "periods" / "periods.parquet"
DEFAULT_APPLIED_OUTPUT_DIR = SHENBAO_DIR / "shenbao_network" / "network_applied"
DEFAULT_KEYWORDS = ["\u61b2\u6cd5", "\u7acb\u61b2", "\u61b2\u653f", "\u5236\u61b2"]
DEFAULT_BOUNDARY_POS = ["EXCLAMATIONCATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY"]
BATCH_SIZE = 100000


@dataclass(frozen=True)
class PeriodInfo:
    period_set_id: str
    period_id: str
    sort_order: int
    start_date: str
    end_date: str


@dataclass
class TokenRecord:
    token: str
    pos: str
    token_order: int
    is_punctuation: bool
    is_boundary: bool
    is_excluded: bool


@dataclass(frozen=True)
class ContextMeta:
    context_uid: int
    article_uid: int
    date: str


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_list(value: str, treat_none_as_empty: bool = False) -> List[str]:
    normalized = value.strip()
    if not normalized:
        return []
    if treat_none_as_empty and normalized.lower() == "none":
        return []
    return [part.strip() for part in normalized.split(",") if part.strip()]


def sanitize_slug(value: str) -> str:
    out: List[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("_")
    return text or "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate keyword-centered PMI/PPMI from applied or custom tokens.parquet and export CSV."
    )
    parser.add_argument("--tokens-parquet", default=str(DEFAULT_TOKENS_PARQUET))
    parser.add_argument("--applied-token-profile", choices=["strict", "full", "regex-only"])
    parser.add_argument("--periods-parquet", default=str(DEFAULT_PERIODS_PARQUET))
    parser.add_argument("--output-csv")
    parser.add_argument("--dataset-label")
    parser.add_argument(
        "--context-filter",
        default=DEFAULT_CONTEXT_FILTER_NAME,
        help="Context filter name/CSV path. Use 'none' to disable. Default: filter_context_pre_zhixian.",
    )
    parser.add_argument(
        "--context-filter-root",
        default=str(DEFAULT_FILTER_ROOT),
        help=f"Filter root directory. Default: {DEFAULT_FILTER_ROOT}",
    )
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS))
    parser.add_argument("--keyword-match", choices=["token-exact", "token-contains"], default="token-contains")
    parser.add_argument("--center-mode", choices=["keyword-only", "all-tokens"], default="keyword-only")
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--window-type", choices=["symmetric", "split-lr"], default="split-lr")
    parser.add_argument("--distance-weight", choices=["none", "inverse", "linear-decay"], default="none")
    parser.add_argument("--count-mode", choices=["raw-freq", "context-freq", "article-freq"], default="raw-freq")
    parser.add_argument("--period-set-id", default="global_all")
    parser.add_argument("--period-id")
    parser.add_argument(
        "--boundary-pos",
        default=",".join(DEFAULT_BOUNDARY_POS),
        help='Comma-separated POS list or "none".',
    )
    parser.add_argument("--skip-nonboundary-punctuation", type=parse_bool, default=True)
    parser.add_argument("--min-joint-count", type=float, default=3)
    parser.add_argument("--min-center-count", type=float, default=5)
    parser.add_argument("--min-neighbor-count", type=float, default=5)
    parser.add_argument("--min-article-count", type=int, default=1)
    parser.add_argument("--min-context-count", type=int, default=1)
    parser.add_argument("--min-token-length", type=int, default=1)
    parser.add_argument("--exclude-numeric-only", type=parse_bool, default=False)
    parser.add_argument("--exclude-latin-only", type=parse_bool, default=False)
    parser.add_argument("--exclude-pos", default="")
    parser.add_argument(
        "--sort-by",
        choices=["raw_joint_event_count", "ppmi", "pmi", "joint_count"],
        default="raw_joint_event_count",
    )
    parser.add_argument("--topn", type=int)
    parser.add_argument("--rows-per-output-file", type=int, default=500000)
    parser.add_argument("--write-chunk-size", type=int, default=50000)
    parser.add_argument("--write-pause-seconds", type=float, default=0.1)
    return parser.parse_args()


def resolve_dataset_label_base(value: Optional[str]) -> str:
    if value and value.strip():
        return value.strip()
    entered = input("Enter dataset-label: ").strip()
    if not entered:
        raise ValueError("dataset-label is required.")
    return entered


def resolve_applied_token_profile_selection(
    value: Optional[str],
    tokens_parquet_arg: str,
) -> Tuple[Optional[str], Optional[str]]:
    if value:
        return value, value
    requested_tokens_path = Path(tokens_parquet_arg).expanduser().resolve()
    if requested_tokens_path != DEFAULT_TOKENS_PARQUET.resolve():
        return None, None
    entered = input(
        "Enter applied-token-profile [strict/full/regex-only/custom] (Enter=regex-only): "
    ).strip().lower()
    if not entered:
        return "regex-only", "regex-only"
    if entered not in {"strict", "full", "regex-only"}:
        raise ValueError("applied-token-profile must be one of: strict, full, regex-only, custom tokens-parquet")
    return entered, entered


def build_effective_dataset_label(dataset_label_base: str, profile_label: Optional[str]) -> str:
    if profile_label in {"strict", "full", "regex-only"}:
        return f"{dataset_label_base}_applied_{profile_label}"
    return dataset_label_base


def resolve_tokens_parquet_path(args: argparse.Namespace) -> Path:
    if args.applied_token_profile:
        profile_dir = {
            "strict": "strict",
            "full": "full",
            "regex-only": "regex_only",
        }[args.applied_token_profile]
        return (
            DEFAULT_APPLIED_TOKENS_DIR
            / profile_dir
            / "tokens.parquet"
        ).resolve()
    return Path(args.tokens_parquet).expanduser().resolve()


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output_csv:
        return Path(args.output_csv).expanduser().resolve()
    output_root = DEFAULT_APPLIED_OUTPUT_DIR
    period_scope = args.period_id if args.period_id else args.period_set_id
    filename = (
        f"network_{sanitize_slug(args.dataset_label)}_{sanitize_slug(args.center_mode)}"
        f"_w{args.window_size}_{sanitize_slug(args.window_type)}_{sanitize_slug(args.distance_weight)}"
        f"_{sanitize_slug(args.count_mode)}_{sanitize_slug(period_scope)}"
        f"{context_filter_stem_part(args.context_filter_info)}.csv"
    )
    return (output_root / filename).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_periods(periods_path: Path, period_set_id: str, period_id: Optional[str]) -> List[PeriodInfo]:
    periods_df = pd.read_parquet(periods_path, columns=["period_set_id", "period_id", "sort_order", "start_date", "end_date"])
    subset = periods_df[periods_df["period_set_id"] == period_set_id].copy()
    if period_id:
        subset = subset[subset["period_id"] == period_id].copy()
    if subset.empty:
        raise ValueError(f"No periods matched period_set_id={period_set_id}, period_id={period_id or 'ALL'}.")
    subset = subset.sort_values(["sort_order", "period_id"])
    periods: List[PeriodInfo] = []
    for row in subset.itertuples(index=False):
        periods.append(
            PeriodInfo(
                period_set_id=str(row.period_set_id),
                period_id=str(row.period_id),
                sort_order=int(row.sort_order),
                start_date=str(row.start_date),
                end_date=str(row.end_date),
            )
        )
    return periods


def build_date_to_periods(periods: Sequence[PeriodInfo]) -> Dict[str, List[PeriodInfo]]:
    mapping: DefaultDict[str, List[PeriodInfo]] = defaultdict(list)
    for period in periods:
        cur = datetime.strptime(period.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(period.end_date, "%Y-%m-%d").date()
        while cur <= end:
            mapping[cur.isoformat()].append(period)
            cur += timedelta(days=1)
    return dict(mapping)


def matches_keyword(token: str, keywords: Sequence[str], keyword_match: str) -> bool:
    if keyword_match == "token-exact":
        return token in keywords
    return any(keyword in token for keyword in keywords)


def is_numeric_only(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", token))


def is_latin_only(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+", token))


def compute_weight(distance: int, window_size: int, distance_weight: str) -> float:
    if distance_weight == "none":
        return 1.0
    if distance_weight == "inverse":
        return 1.0 / float(distance)
    if distance_weight == "linear-decay":
        return float(window_size - distance + 1) / float(window_size)
    raise ValueError(f"Unsupported distance-weight: {distance_weight}")


def format_elapsed(seconds: float) -> str:
    rounded_seconds = int(seconds)
    minutes, remain_seconds = divmod(rounded_seconds, 60)
    return f"{minutes}m {remain_seconds}s ({seconds:,.1f}s)"


def build_pos_pair_json(counter: Counter) -> str:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ordered = {key: int(value) for key, value in items}
    return json.dumps(ordered, ensure_ascii=False)


def print_config(args: argparse.Namespace, output_path: Path, periods: Sequence[PeriodInfo]) -> None:
    config_rows = [
        ("dataset-label", args.dataset_label),
        ("applied-token-profile", args.applied_token_profile_label or "custom"),
        ("tokens-parquet", str(args.resolved_tokens_parquet)),
        ("periods-parquet", str(Path(args.periods_parquet).resolve())),
        ("output-csv", str(output_path)),
        ("context-filter", args.context_filter_info.name if args.context_filter_info else "none"),
        ("context-filter-csv", str(args.context_filter_info.path) if args.context_filter_info else ""),
        (
            "context-filter-excluded-uids",
            len(args.context_filter_info.excluded_context_uids) if args.context_filter_info else 0,
        ),
        ("keywords", ",".join(args.keywords_list)),
        ("keyword-match", args.keyword_match),
        ("center-mode", args.center_mode),
        ("window-size", args.window_size),
        ("window-type", args.window_type),
        ("distance-weight", args.distance_weight),
        ("count-mode", args.count_mode),
        ("period-set-id", args.period_set_id),
        ("period-id", args.period_id or "ALL"),
        ("selected-period-count", len(periods)),
        ("boundary-pos", ",".join(args.boundary_pos_list) if args.boundary_pos_list else "none"),
        ("skip-nonboundary-punctuation", args.skip_nonboundary_punctuation),
        ("min-joint-count", args.min_joint_count),
        ("min-center-count", args.min_center_count),
        ("min-neighbor-count", args.min_neighbor_count),
        ("min-article-count", args.min_article_count),
        ("min-context-count", args.min_context_count),
        ("min-token-length", args.min_token_length),
        ("exclude-numeric-only", args.exclude_numeric_only),
        ("exclude-latin-only", args.exclude_latin_only),
        ("exclude-pos", ",".join(args.exclude_pos_list) if args.exclude_pos_list else "none"),
        ("sort-by", args.sort_by),
        ("topn", args.topn if args.topn is not None else "ALL"),
        ("rows-per-output-file", args.rows_per_output_file),
        ("write-chunk-size", args.write_chunk_size),
        ("write-pause-seconds", args.write_pause_seconds),
        ("punctuation-nodes", "always excluded"),
    ]
    print("command line options", flush=True)
    for key, value in config_rows:
        print(f"  - {key}: {value}", flush=True)


def make_token_record(row: dict, boundary_pos: set, exclude_pos: set, args: argparse.Namespace) -> TokenRecord:
    token = str(row["token"])
    pos = str(row["pos"])
    is_punctuation = bool(row["is_punctuation"])
    is_boundary = pos in boundary_pos
    excluded = False
    if not is_punctuation:
        if pos in exclude_pos:
            excluded = True
        elif args.exclude_numeric_only and is_numeric_only(token):
            excluded = True
        elif args.exclude_latin_only and is_latin_only(token):
            excluded = True
        elif len(token) < args.min_token_length:
            excluded = True
    return TokenRecord(
        token=token,
        pos=pos,
        token_order=int(row["token_order_in_context"]),
        is_punctuation=is_punctuation,
        is_boundary=is_boundary,
        is_excluded=excluded,
    )


def iter_contexts(tokens_path: Path, boundary_pos: set, exclude_pos: set, args: argparse.Namespace) -> Iterable[Tuple[ContextMeta, List[TokenRecord]]]:
    parquet = pq.ParquetFile(tokens_path)
    columns = [
        "context_uid",
        "article_uid",
        "date",
        "token_order_in_context",
        "token",
        "pos",
        "is_punctuation",
    ]
    excluded_context_uids = (
        args.context_filter_info.excluded_context_uids
        if args.context_filter_info
        else set()
    )

    def is_excluded_context(meta: ContextMeta) -> bool:
        return str(meta.context_uid) in excluded_context_uids

    current_meta: Optional[ContextMeta] = None
    current_tokens: List[TokenRecord] = []
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        data = batch.to_pydict()
        size = len(data["context_uid"])
        for idx in range(size):
            context_uid = int(data["context_uid"][idx])
            article_uid = int(data["article_uid"][idx])
            token_date = str(data["date"][idx])
            if current_meta is None:
                current_meta = ContextMeta(context_uid=context_uid, article_uid=article_uid, date=token_date)
            elif context_uid != current_meta.context_uid:
                current_tokens.sort(key=lambda token: token.token_order)
                if not is_excluded_context(current_meta):
                    yield current_meta, current_tokens
                current_meta = ContextMeta(context_uid=context_uid, article_uid=article_uid, date=token_date)
                current_tokens = []
            row = {column: data[column][idx] for column in columns[3:]}
            current_tokens.append(make_token_record(row, boundary_pos, exclude_pos, args))
    if current_meta is not None:
        current_tokens.sort(key=lambda token: token.token_order)
        if not is_excluded_context(current_meta):
            yield current_meta, current_tokens


def find_neighbors(
    tokens: Sequence[TokenRecord],
    center_index: int,
    step: int,
    window_size: int,
    skip_nonboundary_punctuation: bool,
) -> List[Tuple[int, int]]:
    neighbors: List[Tuple[int, int]] = []
    cursor = center_index + step
    logical_distance = 0
    while 0 <= cursor < len(tokens) and logical_distance < window_size:
        token = tokens[cursor]
        if token.is_boundary:
            break
        if token.is_punctuation:
            if skip_nonboundary_punctuation:
                cursor += step
                continue
            break
        if token.is_excluded:
            cursor += step
            continue
        logical_distance += 1
        neighbors.append((cursor, logical_distance))
        cursor += step
    return neighbors


def update_weighted_counts(
    pair_weights: Dict[Tuple[str, str, str, str, str], float],
    center_weights: Dict[Tuple[str, str, str, str], float],
    neighbor_weights: Dict[Tuple[str, str, str, str], float],
    event_totals: Dict[Tuple[str, str, str], float],
    pair_key: Tuple[str, str, str, str, str],
    weight: float,
) -> None:
    period_set_id, period_id, direction, center_token, neighbor_token = pair_key
    pair_weights[pair_key] += weight
    center_weights[(period_set_id, period_id, direction, center_token)] += weight
    neighbor_weights[(period_set_id, period_id, direction, neighbor_token)] += weight
    event_totals[(period_set_id, period_id, direction)] += weight


def process_context(
    meta: ContextMeta,
    tokens: Sequence[TokenRecord],
    periods_for_date: Sequence[PeriodInfo],
    args: argparse.Namespace,
    pair_weights: Dict[Tuple[str, str, str, str, str], float],
    center_weights: Dict[Tuple[str, str, str, str], float],
    neighbor_weights: Dict[Tuple[str, str, str, str], float],
    event_totals: Dict[Tuple[str, str, str], float],
    article_best: Dict[Tuple[str, str, int, str, str, str], float],
    raw_pair_counts: Counter,
    raw_distance_sums: Dict[Tuple[str, str, str, str, str], float],
    raw_center_counts: Counter,
    raw_neighbor_counts: Counter,
    raw_event_totals: Counter,
    pair_context_counts: Counter,
    article_seen_pairs: set,
    pair_pos_counters: DefaultDict[Tuple[str, str, str, str, str], Counter],
) -> int:
    context_best: Dict[Tuple[str, str, str, str, str], float] = {}
    context_seen_pairs: set = set()
    raw_event_counter = 0
    for center_index, center_token in enumerate(tokens):
        if center_token.is_punctuation or center_token.is_excluded:
            continue
        if args.center_mode == "keyword-only" and not matches_keyword(
            center_token.token,
            args.keywords_list,
            args.keyword_match,
        ):
            continue
        left_neighbors = find_neighbors(
            tokens,
            center_index,
            step=-1,
            window_size=args.window_size,
            skip_nonboundary_punctuation=args.skip_nonboundary_punctuation,
        )
        right_neighbors = find_neighbors(
            tokens,
            center_index,
            step=1,
            window_size=args.window_size,
            skip_nonboundary_punctuation=args.skip_nonboundary_punctuation,
        )
        event_specs: List[Tuple[str, int, int, str]] = []
        if args.window_type == "split-lr":
            for neighbor_index, distance in left_neighbors:
                surface_pos_pair = f"{tokens[neighbor_index].pos}|{center_token.pos}"
                event_specs.append(("L", neighbor_index, distance, surface_pos_pair))
            for neighbor_index, distance in right_neighbors:
                surface_pos_pair = f"{center_token.pos}|{tokens[neighbor_index].pos}"
                event_specs.append(("R", neighbor_index, distance, surface_pos_pair))
        else:
            for neighbor_index, distance in left_neighbors:
                surface_pos_pair = f"{tokens[neighbor_index].pos}|{center_token.pos}"
                event_specs.append(("S", neighbor_index, distance, surface_pos_pair))
            for neighbor_index, distance in right_neighbors:
                surface_pos_pair = f"{center_token.pos}|{tokens[neighbor_index].pos}"
                event_specs.append(("S", neighbor_index, distance, surface_pos_pair))

        for direction, neighbor_index, distance, surface_pos_pair in event_specs:
            neighbor_token = tokens[neighbor_index]
            weight = compute_weight(distance, args.window_size, args.distance_weight)
            for period in periods_for_date:
                pair_key = (
                    period.period_set_id,
                    period.period_id,
                    direction,
                    center_token.token,
                    neighbor_token.token,
                )
                raw_pair_counts[pair_key] += 1
                raw_distance_sums[pair_key] += float(distance)
                raw_center_counts[(period.period_set_id, period.period_id, direction, center_token.token)] += 1
                raw_neighbor_counts[(period.period_set_id, period.period_id, direction, neighbor_token.token)] += 1
                raw_event_totals[(period.period_set_id, period.period_id, direction)] += 1
                context_seen_pairs.add(pair_key)
                article_seen_pairs.add(pair_key)
                pair_pos_counters[pair_key][surface_pos_pair] += 1
                if args.count_mode == "raw-freq":
                    update_weighted_counts(
                        pair_weights,
                        center_weights,
                        neighbor_weights,
                        event_totals,
                        pair_key,
                        weight,
                    )
                elif args.count_mode == "context-freq":
                    prev = context_best.get(pair_key)
                    if prev is None or weight > prev:
                        context_best[pair_key] = weight
                else:
                    article_key = (
                        period.period_set_id,
                        period.period_id,
                        meta.article_uid,
                        direction,
                        center_token.token,
                        neighbor_token.token,
                    )
                    prev = article_best.get(article_key)
                    if prev is None or weight > prev:
                        article_best[article_key] = weight
            raw_event_counter += 1

    if args.count_mode == "context-freq":
        for pair_key, weight in context_best.items():
            update_weighted_counts(pair_weights, center_weights, neighbor_weights, event_totals, pair_key, weight)
    for pair_key in context_seen_pairs:
        pair_context_counts[pair_key] += 1
    return raw_event_counter


def flush_article_freq(
    article_best: Dict[Tuple[str, str, int, str, str, str], float],
    pair_weights: Dict[Tuple[str, str, str, str, str], float],
    center_weights: Dict[Tuple[str, str, str, str], float],
    neighbor_weights: Dict[Tuple[str, str, str, str], float],
    event_totals: Dict[Tuple[str, str, str], float],
) -> None:
    for article_key, weight in article_best.items():
        period_set_id, period_id, _, direction, center_token, neighbor_token = article_key
        pair_key = (period_set_id, period_id, direction, center_token, neighbor_token)
        update_weighted_counts(pair_weights, center_weights, neighbor_weights, event_totals, pair_key, weight)


def flush_article_counts(
    pair_article_counts: Counter,
    article_seen_pairs: set,
) -> None:
    for pair_key in article_seen_pairs:
        pair_article_counts[pair_key] += 1


def build_spool_db_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.spool.sqlite")


def spool_rows_to_sqlite(
    spool_path: Path,
    periods: Sequence[PeriodInfo],
    args: argparse.Namespace,
    pair_weights: Dict[Tuple[str, str, str, str, str], float],
    center_weights: Dict[Tuple[str, str, str, str], float],
    neighbor_weights: Dict[Tuple[str, str, str, str], float],
    event_totals: Dict[Tuple[str, str, str], float],
    raw_pair_counts: Counter,
    raw_distance_sums: Dict[Tuple[str, str, str, str, str], float],
    raw_center_counts: Counter,
    raw_neighbor_counts: Counter,
    raw_event_totals: Counter,
    pair_context_counts: Counter,
    pair_article_counts: Counter,
    pair_pos_counters: DefaultDict[Tuple[str, str, str, str, str], Counter],
) -> int:
    if spool_path.exists():
        spool_path.unlink()
    conn = sqlite3.connect(spool_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE rows (
            period_set_id TEXT NOT NULL,
            period_id TEXT NOT NULL,
            period_sort_order INTEGER NOT NULL,
            center_token TEXT NOT NULL,
            neighbor_token TEXT NOT NULL,
            direction TEXT NOT NULL,
            joint_count REAL NOT NULL,
            raw_joint_event_count INTEGER NOT NULL,
            center_marginal_count REAL NOT NULL,
            neighbor_marginal_count REAL NOT NULL,
            event_total REAL NOT NULL,
            raw_event_total INTEGER NOT NULL,
            center_raw_count INTEGER NOT NULL,
            neighbor_raw_count INTEGER NOT NULL,
            pmi REAL NOT NULL,
            ppmi REAL NOT NULL,
            distinct_context_count INTEGER NOT NULL,
            distinct_article_count INTEGER NOT NULL,
            avg_distance REAL NOT NULL,
            pos_pair_json TEXT NOT NULL
        )
        """
    )
    insert_sql = """
        INSERT INTO rows (
            period_set_id,
            period_id,
            period_sort_order,
            center_token,
            neighbor_token,
            direction,
            joint_count,
            raw_joint_event_count,
            center_marginal_count,
            neighbor_marginal_count,
            event_total,
            raw_event_total,
            center_raw_count,
            neighbor_raw_count,
            pmi,
            ppmi,
            distinct_context_count,
            distinct_article_count,
            avg_distance,
            pos_pair_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    period_order = {(period.period_set_id, period.period_id): period.sort_order for period in periods}
    batch: List[Tuple[object, ...]] = []
    inserted_rows = 0
    try:
        for pair_key, joint_count in pair_weights.items():
            period_set_id, period_id, direction, center_token, neighbor_token = pair_key
            center_count = center_weights[(period_set_id, period_id, direction, center_token)]
            neighbor_count = neighbor_weights[(period_set_id, period_id, direction, neighbor_token)]
            event_total = event_totals[(period_set_id, period_id, direction)]
            raw_joint_event_count = raw_pair_counts[pair_key]
            raw_event_total = raw_event_totals[(period_set_id, period_id, direction)]
            center_raw_count = raw_center_counts[(period_set_id, period_id, direction, center_token)]
            neighbor_raw_count = raw_neighbor_counts[(period_set_id, period_id, direction, neighbor_token)]
            distinct_context_count = pair_context_counts[pair_key]
            distinct_article_count = pair_article_counts[pair_key]
            if joint_count < args.min_joint_count:
                continue
            if center_count < args.min_center_count:
                continue
            if neighbor_count < args.min_neighbor_count:
                continue
            if distinct_article_count < args.min_article_count:
                continue
            if distinct_context_count < args.min_context_count:
                continue
            pmi = math.log2((joint_count * event_total) / (center_count * neighbor_count))
            batch.append(
                (
                    period_set_id,
                    period_id,
                    period_order[(period_set_id, period_id)],
                    center_token,
                    neighbor_token,
                    direction,
                    joint_count,
                    raw_joint_event_count,
                    center_count,
                    neighbor_count,
                    event_total,
                    raw_event_total,
                    center_raw_count,
                    neighbor_raw_count,
                    pmi,
                    max(pmi, 0.0),
                    distinct_context_count,
                    distinct_article_count,
                    raw_distance_sums[pair_key] / float(raw_joint_event_count),
                    build_pos_pair_json(pair_pos_counters[pair_key]),
                )
            )
            inserted_rows += 1
            if len(batch) >= 10000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()
                if inserted_rows % 100000 == 0:
                    log(f"Spooled final rows: {inserted_rows:,}")
        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
    finally:
        conn.close()
    return inserted_rows


def build_ranked_rows_query(sort_by: str) -> str:
    metric = "raw_joint_event_count"
    if sort_by == "joint_count":
        metric = "joint_count"
    elif sort_by == "pmi":
        metric = "pmi"
    elif sort_by == "ppmi":
        metric = "ppmi"
    return f"""
        WITH ranked AS (
            SELECT
                period_set_id,
                period_id,
                period_sort_order,
                center_token,
                neighbor_token,
                direction,
                joint_count,
                raw_joint_event_count,
                center_marginal_count,
                neighbor_marginal_count,
                event_total,
                raw_event_total,
                center_raw_count,
                neighbor_raw_count,
                pmi,
                ppmi,
                distinct_context_count,
                distinct_article_count,
                avg_distance,
                pos_pair_json,
                ROW_NUMBER() OVER (
                    ORDER BY
                        joint_count DESC,
                        raw_joint_event_count DESC,
                        period_sort_order ASC,
                        period_id ASC,
                        center_token ASC,
                        direction ASC,
                        neighbor_token ASC
                ) AS rank_by_count,
                ROW_NUMBER() OVER (
                    ORDER BY
                        pmi DESC,
                        joint_count DESC,
                        raw_joint_event_count DESC,
                        period_sort_order ASC,
                        period_id ASC,
                        center_token ASC,
                        direction ASC,
                        neighbor_token ASC
                ) AS rank_by_pmi
            FROM rows
        )
        SELECT
            period_set_id,
            period_id,
            center_token,
            neighbor_token,
            direction,
            joint_count,
            raw_joint_event_count,
            center_marginal_count,
            neighbor_marginal_count,
            event_total,
            raw_event_total,
            center_raw_count,
            neighbor_raw_count,
            pmi,
            ppmi,
            distinct_context_count,
            distinct_article_count,
            avg_distance,
            pos_pair_json,
            rank_by_count,
            rank_by_pmi
        FROM ranked
        ORDER BY
            {metric} DESC,
            period_sort_order ASC,
            period_id ASC,
            center_token ASC,
            direction ASC,
            neighbor_token ASC
        LIMIT ? OFFSET ?
    """


def write_csv_from_sqlite(
    spool_path: Path,
    output_path: Path,
    sort_by: str,
    topn: Optional[int],
    rows_per_output_file: int,
    chunk_size: int,
    pause_seconds: float,
) -> Tuple[List[Path], int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "period_set_id",
        "period_id",
        "center_token",
        "neighbor_token",
        "direction",
        "joint_count",
        "raw_joint_event_count",
        "center_marginal_count",
        "neighbor_marginal_count",
        "event_total",
        "raw_event_total",
        "center_raw_count",
        "neighbor_raw_count",
        "pmi",
        "ppmi",
        "distinct_context_count",
        "distinct_article_count",
        "avg_distance",
        "pos_pair_json",
        "rank_by_count",
        "rank_by_pmi",
    ]
    conn = sqlite3.connect(spool_path)
    try:
        total_available_rows = int(conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        total_rows = total_available_rows
        if topn is not None:
            total_rows = min(total_available_rows, topn)
        cleanup_legacy_base_output(output_path)
        if total_rows == 0:
            cleanup_stale_output_parts(output_path, 0)
            marker_path = write_empty_output_marker(output_path)
            log(f"No final CSV rows. Wrote empty-result marker: {marker_path.name}")
            return [], 0
        remove_empty_output_marker(output_path)
        effective_rows_per_output_file = max(1, rows_per_output_file)
        effective_chunk_size = max(1, chunk_size)
        completed_paths, completed_rows = load_completed_output_parts(
            output_path,
            effective_rows_per_output_file,
            total_rows,
        )
        if completed_rows > 0:
            log(
                f"Resuming CSV write from row {completed_rows + 1:,}; "
                f"completed parts: {len(completed_paths):,}"
            )
        if completed_rows >= total_rows:
            log("All output parts are already complete. Skipping CSV write.")
            cleanup_stale_output_parts(output_path, len(completed_paths))
            return completed_paths, total_rows

        query = build_ranked_rows_query(sort_by)
        cursor = conn.execute(query, (total_rows - completed_rows, completed_rows))
        written_paths: List[Path] = []
        current_part_index = len(completed_paths)
        current_part_rows = 0
        rows_written_total = completed_rows
        handle = None
        writer = None
        current_output_path: Optional[Path] = None
        try:
            for row in cursor:
                if writer is None or current_part_rows >= effective_rows_per_output_file:
                    if handle is not None:
                        finalize_output_part(handle, current_output_path, current_part_index, current_part_rows)
                        handle = None
                        writer = None
                        current_output_path = None
                        if pause_seconds > 0.0:
                            time.sleep(pause_seconds)
                    current_part_index += 1
                    current_part_rows = 0
                    current_output_path = build_output_part_path(output_path, current_part_index)
                    current_marker_path = build_output_part_marker_path(current_output_path)
                    if current_output_path.exists():
                        log(f"Overwriting incomplete CSV part {current_part_index}: {current_output_path.name}")
                    if current_marker_path.exists():
                        current_marker_path.unlink()
                    written_paths.append(current_output_path)
                    log(f"Opening CSV part {current_part_index}: {current_output_path.name}")
                    handle = current_output_path.open("w", encoding="utf-8", newline="")
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(dict(zip(fieldnames, row)))
                current_part_rows += 1
                rows_written_total += 1
                if rows_written_total % effective_chunk_size == 0 or rows_written_total == total_rows:
                    handle.flush()
                    log(
                        f"Wrote CSV rows: {rows_written_total:,}/{total_rows:,} "
                        f"(part {current_part_index}, part rows: {current_part_rows:,})"
                    )
                    if pause_seconds > 0.0 and rows_written_total < total_rows:
                        time.sleep(pause_seconds)
        finally:
            if handle is not None:
                finalize_output_part(handle, current_output_path, current_part_index, current_part_rows)
        final_paths = completed_paths + written_paths
        cleanup_stale_output_parts(output_path, len(final_paths))
        return final_paths, total_rows
    finally:
        conn.close()


def build_output_part_path(output_path: Path, part_index: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.part{part_index:04d}{output_path.suffix}")


def build_output_part_marker_path(part_path: Path) -> Path:
    return part_path.with_name(f"{part_path.name}.done.json")


def build_empty_output_marker_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.empty.json")


def write_empty_output_marker(output_path: Path) -> Path:
    marker_path = build_empty_output_marker_path(output_path)
    marker = {
        "status": "empty",
        "rows_written": 0,
        "file_name": output_path.name,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return marker_path


def remove_empty_output_marker(output_path: Path) -> None:
    marker_path = build_empty_output_marker_path(output_path)
    if marker_path.exists():
        marker_path.unlink()
        log(f"Removed stale empty output marker: {marker_path.name}")


def cleanup_legacy_base_output(output_path: Path) -> None:
    legacy_marker_path = build_output_part_marker_path(output_path)
    if output_path.exists():
        output_path.unlink()
        log(f"Removed legacy base CSV file: {output_path.name}")
    if legacy_marker_path.exists():
        legacy_marker_path.unlink()
        log(f"Removed legacy base output marker: {legacy_marker_path.name}")


def load_completed_output_parts(
    output_path: Path,
    rows_per_output_file: int,
    total_rows: int,
) -> Tuple[List[Path], int]:
    completed_paths: List[Path] = []
    completed_rows = 0
    effective_rows_per_output_file = max(1, rows_per_output_file)
    part_index = 1
    while completed_rows < total_rows:
        part_path = build_output_part_path(output_path, part_index)
        marker_path = build_output_part_marker_path(part_path)
        if not part_path.exists() or not marker_path.exists():
            break
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker_rows = int(marker["rows_written"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            log(f"Ignoring invalid output marker: {marker_path.name}")
            break
        expected_rows = min(effective_rows_per_output_file, total_rows - completed_rows)
        if marker_rows != expected_rows:
            log(
                f"Ignoring mismatched output marker: {marker_path.name} "
                f"(expected {expected_rows:,}, found {marker_rows:,})"
            )
            break
        completed_paths.append(part_path)
        completed_rows += marker_rows
        part_index += 1
    return completed_paths, completed_rows


def finalize_output_part(
    handle,
    part_path: Path,
    part_index: int,
    part_rows: int,
) -> None:
    handle.flush()
    handle.close()
    marker_path = build_output_part_marker_path(part_path)
    marker = {
        "part_index": part_index,
        "rows_written": part_rows,
        "file_name": part_path.name,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cleanup_stale_output_parts(output_path: Path, keep_part_count: int) -> None:
    part_index = max(1, keep_part_count + 1)
    while True:
        part_path = build_output_part_path(output_path, part_index)
        marker_path = build_output_part_marker_path(part_path)
        if not part_path.exists() and not marker_path.exists():
            break
        if part_path.exists():
            part_path.unlink()
            log(f"Removed stale CSV part: {part_path.name}")
        if marker_path.exists():
            marker_path.unlink()
            log(f"Removed stale output marker: {marker_path.name}")
        part_index += 1


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1.")

    (
        args.applied_token_profile,
        args.applied_token_profile_label,
    ) = resolve_applied_token_profile_selection(
        args.applied_token_profile,
        args.tokens_parquet,
    )
    args.dataset_label_base = resolve_dataset_label_base(args.dataset_label)
    args.dataset_label = build_effective_dataset_label(
        args.dataset_label_base,
        args.applied_token_profile_label,
    )
    args.keywords_list = parse_csv_list(args.keywords)
    if not args.keywords_list:
        raise ValueError("--keywords must contain at least one keyword.")
    args.boundary_pos_list = parse_csv_list(args.boundary_pos, treat_none_as_empty=True)
    args.exclude_pos_list = parse_csv_list(args.exclude_pos, treat_none_as_empty=True)
    args.context_filter_info = load_context_filter(
        args.context_filter,
        Path(args.context_filter_root).expanduser().resolve(),
    )

    tokens_path = resolve_tokens_parquet_path(args)
    args.resolved_tokens_parquet = tokens_path
    periods_path = Path(args.periods_parquet).expanduser().resolve()
    require_file(tokens_path, "tokens parquet")
    require_file(periods_path, "periods parquet")

    periods = load_periods(periods_path, args.period_set_id, args.period_id)
    output_path = build_output_path(args)
    print_config(args, output_path, periods)

    date_to_periods = build_date_to_periods(periods)
    boundary_pos = set(args.boundary_pos_list)
    exclude_pos = set(args.exclude_pos_list)

    pair_weights: Dict[Tuple[str, str, str, str, str], float] = defaultdict(float)
    center_weights: Dict[Tuple[str, str, str, str], float] = defaultdict(float)
    neighbor_weights: Dict[Tuple[str, str, str, str], float] = defaultdict(float)
    event_totals: Dict[Tuple[str, str, str], float] = defaultdict(float)
    article_best: Dict[Tuple[str, str, int, str, str, str], float] = {}

    raw_pair_counts: Counter = Counter()
    raw_distance_sums: Dict[Tuple[str, str, str, str, str], float] = defaultdict(float)
    raw_center_counts: Counter = Counter()
    raw_neighbor_counts: Counter = Counter()
    raw_event_totals: Counter = Counter()
    pair_context_counts: Counter = Counter()
    pair_article_counts: Counter = Counter()
    pair_pos_counters: DefaultDict[Tuple[str, str, str, str, str], Counter] = defaultdict(Counter)

    processed_contexts = 0
    selected_contexts = 0
    raw_event_rows = 0
    started = datetime.now()
    log("Starting context scan.")
    current_article_uid: Optional[int] = None
    article_seen_pairs: set = set()
    for meta, tokens in iter_contexts(tokens_path, boundary_pos, exclude_pos, args):
        processed_contexts += 1
        if current_article_uid is None:
            current_article_uid = meta.article_uid
        elif meta.article_uid != current_article_uid:
            flush_article_counts(pair_article_counts, article_seen_pairs)
            article_seen_pairs.clear()
            if args.count_mode == "article-freq":
                flush_article_freq(article_best, pair_weights, center_weights, neighbor_weights, event_totals)
                article_best.clear()
            current_article_uid = meta.article_uid
        periods_for_date = date_to_periods.get(meta.date)
        if periods_for_date:
            selected_contexts += 1
            raw_event_rows += process_context(
                meta,
                tokens,
                periods_for_date,
                args,
                pair_weights,
                center_weights,
                neighbor_weights,
                event_totals,
                article_best,
                raw_pair_counts,
                raw_distance_sums,
                raw_center_counts,
                raw_neighbor_counts,
                raw_event_totals,
                pair_context_counts,
                article_seen_pairs,
                pair_pos_counters,
            )
        if processed_contexts % 5000 == 0:
            log(
                f"Processed contexts: {processed_contexts:,}; selected contexts: {selected_contexts:,}; "
                f"raw events: {raw_event_rows:,}"
            )

    if current_article_uid is not None:
        flush_article_counts(pair_article_counts, article_seen_pairs)
        if args.count_mode == "article-freq":
            flush_article_freq(article_best, pair_weights, center_weights, neighbor_weights, event_totals)

    spool_path = build_spool_db_path(output_path)
    log("Spooling final rows to SQLite.")
    final_row_count = spool_rows_to_sqlite(
        spool_path,
        periods,
        args,
        pair_weights,
        center_weights,
        neighbor_weights,
        event_totals,
        raw_pair_counts,
        raw_distance_sums,
        raw_center_counts,
        raw_neighbor_counts,
        raw_event_totals,
        pair_context_counts,
        pair_article_counts,
        pair_pos_counters,
    )
    output_row_count = final_row_count if args.topn is None else min(final_row_count, args.topn)
    estimated_output_files = max(1, math.ceil(output_row_count / max(1, args.rows_per_output_file)))
    log(f"Writing CSV rows: {output_row_count:,} across {estimated_output_files:,} output file(s).")
    written_paths, written_row_count = write_csv_from_sqlite(
        spool_path,
        output_path,
        args.sort_by,
        args.topn,
        args.rows_per_output_file,
        args.write_chunk_size,
        args.write_pause_seconds,
    )
    if spool_path.exists():
        spool_path.unlink()
    elapsed = (datetime.now() - started).total_seconds()
    log(f"Processed contexts total: {processed_contexts:,}")
    log(f"Selected contexts total: {selected_contexts:,}")
    log(f"Raw event rows total: {raw_event_rows:,}")
    log(f"Final CSV rows: {written_row_count:,}")
    if len(written_paths) == 1:
        log(f"Output: {written_paths[0]}")
    elif written_paths:
        log(f"Output files: {len(written_paths):,} parts under {output_path.parent}")
        log(f"Output base: {output_path.name}")
    else:
        log(f"Output: {output_path}")
    log(f"Done in {format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
