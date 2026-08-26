#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute pair-conditioned keyness from Shenbao undirected network edges."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
SHENBAO = ROOT / "shenbao"
DEFAULT_INPUT_DIR = (
    SHENBAO
    / "shenbao_network"
    / "network_applied"
    / "for_gephi"
    / "dynamic"
    / "long_period_manual"
)
DEFAULT_GLOBAL_INPUT_DIR = (
    SHENBAO
    / "shenbao_network"
    / "network_applied"
    / "stopfiltered_global_filtered_pre_zhixian_context"
    / "undirected"
)
DEFAULT_PERIOD_OVERRIDE_DIR = (
    SHENBAO
    / "shenbao_network"
    / "network_applied"
    / "stopfiltered_long_period_manual_p001_filtered_pre_zhixian_context"
    / "undirected"
)
DEFAULT_PERIOD_KEYNESS_CSV = (
    SHENBAO
    / "shenbao_interpretation"
    / "period_keyness"
    / "period_keyness_long_period_manual_regex-only_strict_full_top100.csv"
)
DEFAULT_OUTPUT_DIR = SHENBAO / "shenbao_interpretation" / "pair_conditioned_keyness"

CORE_ORDER = ["lixian", "xianzheng", "xianfa", "zhixian"]
CORE_TOKENS = {
    "lixian": "\u7acb\u61b2",
    "xianzheng": "\u61b2\u653f",
    "xianfa": "\u61b2\u6cd5",
    "zhixian": "\u5236\u61b2",
}
PERIOD_DATES = {
    "long_period_manual_p000": ("1872-08-20", "1905-07-11", 0),
    "long_period_manual_p001": ("1905-07-12", "1913-02-22", 1),
    "long_period_manual_p002": ("1913-02-23", "1923-11-17", 2),
    "long_period_manual_p003": ("1923-11-18", "1937-07-25", 3),
    "long_period_manual_p004": ("1937-07-26", "1945-11-23", 4),
    "long_period_manual_p005": ("1945-11-24", "1949-05-26", 5),
}
DEFAULT_PAIRS = list(combinations(CORE_ORDER, 2))
DEFAULT_PROFILES = ["regex-only", "strict", "full"]
DEFAULT_WINDOWS = [1, 5, 10, 20]
DEFAULT_NEIGHBOR_SCOPES = ["all", "top20", "top50", "top100"]
METRICS_FOR_ROBUSTNESS = ["log_odds_z", "log_likelihood", "log_ratio", "chi_square", "tfidf"]

TOKEN_COUNT_FIELDNAMES = [
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "period_label",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "pair_id",
    "focus_a_norm",
    "focus_a_label",
    "focus_b_norm",
    "focus_b_label",
    "candidate_scope",
    "neighbor_scope",
    "topn",
    "pair_count_mode",
    "token",
    "pair_count",
    "edge_a",
    "edge_b",
    "norm_a",
    "norm_b",
    "shared_strength",
    "rank_a",
    "rank_b",
    "a_strength",
    "b_strength",
    "a_neighbor_count",
    "b_neighbor_count",
    "source_file",
]

KEYNESS_FIELDNAMES = [
    "token",
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "period_label",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "pair_id",
    "focus_a_norm",
    "focus_a_label",
    "focus_b_norm",
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
    "log_odds_delta",
    "log_odds_z",
    "log_likelihood",
    "log_ratio",
    "chi_square",
    "tfidf",
    "direction",
    "rank_abs_z",
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
    "a_strength",
    "b_strength",
    "a_neighbor_count",
    "b_neighbor_count",
    "period_keyness_log_odds_z",
    "period_keyness_rank_positive",
    "period_keyness_rank_negative",
    "period_keyness_count",
    "period_keyness_rate",
    "dominant_pos",
    "dict_lv1",
    "dict_lv2",
    "dict_ner_like_type",
    "is_core_exact",
    "is_core_contain",
    "matched_core_keyword",
    "source_file",
]

ROBUST_FIELDNAMES = [
    "token",
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "period_label",
    "period_sort_order",
    "period_start_date",
    "period_end_date",
    "pair_id",
    "focus_a_norm",
    "focus_a_label",
    "focus_b_norm",
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
]


def log(message: str) -> None:
    print(message, flush=True)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    raise_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if fieldnames and fieldnames[0].startswith("\ufeff"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(parse_list(item))
        return out
    return [part.strip() for part in re.split(r"[,\s]+", str(value).strip()) if part.strip()]


def parse_int_list(value: object) -> List[int]:
    return [int(part) for part in parse_list(value)]


def parse_float(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def normalize_profile(value: object) -> str:
    text = str(value).strip()
    return "regex-only" if text == "regex_only" else text


def sanitize_slug(value: object) -> str:
    out: List[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "value"


def json_ready(value: object) -> object:
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


def canonical_pair(token_a: str, token_b: str) -> Tuple[str, str]:
    return (token_a, token_b) if token_a <= token_b else (token_b, token_a)


def should_use_row(row: Dict[str, str]) -> bool:
    direction = row.get("direction", "").strip()
    return not direction or direction in {"R", "U"} or row.get("Type", "").strip().lower() == "undirected"


def token_for_core(value: str) -> str:
    value = value.strip()
    return CORE_TOKENS.get(value, value)


def core_norm_for_token(token: str) -> str:
    for norm, core_token in CORE_TOKENS.items():
        if token == core_token:
            return norm
    return token


def parse_pairs(values: Sequence[str]) -> List[Tuple[str, str]]:
    if not values:
        return [(CORE_TOKENS[a], CORE_TOKENS[b]) for a, b in DEFAULT_PAIRS]
    pairs: List[Tuple[str, str]] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                left, right = part.split(":", 1)
            elif "-" in part:
                left, right = part.split("-", 1)
            else:
                raise ValueError(f"Pair must use ':' or '-': {part}")
            pairs.append((token_for_core(left), token_for_core(right)))
    return pairs


def parse_neighbor_scopes(values: Sequence[str]) -> List[Tuple[str, int]]:
    scopes = values or DEFAULT_NEIGHBOR_SCOPES
    parsed: List[Tuple[str, int]] = []
    for scope in scopes:
        text = str(scope).strip().lower()
        if text in {"all", "top0", "0"}:
            parsed.append(("all", 0))
        elif text.startswith("top"):
            parsed.append((text, int(text[3:])))
        else:
            parsed.append((f"top{int(text)}", int(text)))
    return parsed


def parse_pair_count_modes(values: Sequence[str]) -> List[str]:
    modes = values or ["sum", "min"]
    out: List[str] = []
    for mode in modes:
        text = str(mode).strip().lower()
        if text == "raw_min":
            text = "min"
        if text not in {"sum", "min"}:
            raise ValueError(f"Unsupported pair count mode: {mode}")
        if text not in out:
            out.append(text)
    return out


def parse_source_metadata(input_csv: Path) -> Dict[str, str]:
    name = input_csv.name
    profile_match = re.search(r"_(strict|full|regex-only)_(?:all-tokens_)?w\d+_", name)
    window_match = re.search(r"_w(\d+)_", name)
    return {
        "source_file": str(input_csv),
        "token_profile": profile_match.group(1) if profile_match else "",
        "network_window": window_match.group(1) if window_match else "",
    }


def discover_period_override_csvs(
    input_dir: Path,
    profiles: Sequence[str],
    windows: Sequence[int],
    period_set_id: str,
) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        return []
    out: List[Path] = []
    for profile in profiles:
        for window in windows:
            pattern = f"*_{profile}_w{window}_{period_set_id}_p*_filtered_*stopv5always.csv"
            matches = sorted(input_dir.glob(pattern))
            if not matches:
                log(f"WARNING: no period override CSV for profile={profile} window={window}: {pattern}")
                continue
            out.append(matches[-1])
    return out


def discover_input_csvs(
    input_dir: Path,
    profiles: Sequence[str],
    windows: Sequence[int],
    period_set_id: str,
) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    out: List[Path] = []
    for profile in profiles:
        for window in windows:
            pattern = f"*{period_set_id}_{profile}_all-tokens_w{window}_undirected*_edges.csv"
            matches = sorted(input_dir.glob(pattern))
            if not matches:
                log(f"WARNING: no edge CSV for profile={profile} window={window}: {pattern}")
                continue
            out.append(matches[-1])
    return out


def discover_global_input_csvs(
    input_dir: Path,
    profiles: Sequence[str],
    windows: Sequence[int],
) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    out: List[Path] = []
    for profile in profiles:
        for window in windows:
            pattern = f"*_{profile}_all-tokens_w{window}_*global_all*_stopv5alwaysfiltered.csv"
            matches = sorted(input_dir.glob(pattern))
            if not matches:
                log(f"WARNING: no global edge CSV for profile={profile} window={window}: {pattern}")
                continue
            out.append(matches[-1])
    return out


def unique_paths(paths: Sequence[Path]) -> List[Path]:
    seen: Set[str] = set()
    out: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def load_period_edges(
    input_csv: Path,
    weight_col: str,
) -> Tuple[Dict[Tuple[str, str], Dict[Tuple[str, str], float]], Dict[Tuple[str, str], Dict[str, object]], Dict[str, str]]:
    rows, fieldnames = read_csv_dicts(input_csv)
    source_col = "Source" if "Source" in fieldnames else "center_token"
    target_col = "Target" if "Target" in fieldnames else "neighbor_token"
    required = {source_col, target_col, "period_set_id", "period_id", weight_col}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Missing required column(s) in {input_csv}: {', '.join(sorted(missing))}")

    metadata = parse_source_metadata(input_csv)
    period_edges: DefaultDict[Tuple[str, str], DefaultDict[Tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    period_meta: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        if not should_use_row(row):
            continue
        token_a = row.get(source_col, "").strip()
        token_b = row.get(target_col, "").strip()
        if not token_a or not token_b or token_a == token_b:
            continue
        weight = parse_float(row.get(weight_col))
        if weight <= 0:
            continue
        period_key = (row["period_set_id"].strip(), row["period_id"].strip())
        period_edges[period_key][canonical_pair(token_a, token_b)] += weight
        if period_key not in period_meta:
            start = row.get("Start") or row.get("period_start_date") or ""
            end = row.get("End") or row.get("period_end_date") or ""
            sort_order = row.get("sort_order") or row.get("period_sort_order") or ""
            known_period = PERIOD_DATES.get(period_key[1])
            if known_period:
                start = start or known_period[0]
                end = end or known_period[1]
                sort_order = sort_order or known_period[2]
            period_meta[period_key] = {
                "period_label": row["period_id"].strip(),
                "period_sort_order": int(float(sort_order)) if str(sort_order).strip() else 0,
                "period_start_date": start,
                "period_end_date": end,
            }
        if not metadata["token_profile"]:
            metadata["token_profile"] = normalize_profile(row.get("profile", ""))
        if not metadata["network_window"]:
            metadata["network_window"] = str(row.get("window_size", "")).strip()
    return {key: dict(edges) for key, edges in period_edges.items()}, period_meta, metadata


def build_raw_vectors(
    edges: Dict[Tuple[str, str], float]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, int], Set[str]]:
    raw_vectors: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    strengths: DefaultDict[str, float] = defaultdict(float)
    tokens: Set[str] = set()
    for (token_a, token_b), weight in edges.items():
        raw_vectors[token_a][token_b] = raw_vectors[token_a].get(token_b, 0.0) + weight
        raw_vectors[token_b][token_a] = raw_vectors[token_b].get(token_a, 0.0) + weight
        strengths[token_a] += weight
        strengths[token_b] += weight
        tokens.add(token_a)
        tokens.add(token_b)
    neighbor_counts = {token: len(vector) for token, vector in raw_vectors.items()}
    return dict(raw_vectors), dict(strengths), neighbor_counts, tokens


def rank_vector(vector: Dict[str, float], excluded: Set[str]) -> Dict[str, int]:
    items = [(token, weight) for token, weight in vector.items() if token not in excluded and weight > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    return {token: idx for idx, (token, _weight) in enumerate(items, start=1)}


def top_tokens(vector: Dict[str, float], excluded: Set[str], topn: int) -> Set[str]:
    ranked = rank_vector(vector, excluded)
    if topn <= 0:
        return set(ranked)
    return {token for token, rank in ranked.items() if rank <= topn}


def build_pair_token_rows(
    input_csv: Path,
    pairs: Sequence[Tuple[str, str]],
    candidate_scope: str,
    neighbor_scopes: Sequence[Tuple[str, int]],
    pair_count_modes: Sequence[str],
    weight_col: str,
) -> List[Dict[str, object]]:
    period_edges, period_meta, metadata = load_period_edges(input_csv, weight_col)
    rows: List[Dict[str, object]] = []
    excluded = set(CORE_TOKENS.values())
    source_file = str(input_csv)
    for period_key in sorted(period_edges):
        edges = period_edges[period_key]
        raw_vectors, strengths, neighbor_counts, _tokens = build_raw_vectors(edges)
        meta = period_meta.get(period_key, {})
        period_set_id, period_id = period_key
        for focus_a, focus_b in pairs:
            raw_a = raw_vectors.get(focus_a, {})
            raw_b = raw_vectors.get(focus_b, {})
            if not raw_a or not raw_b:
                continue
            strength_a = float(strengths.get(focus_a, 0.0))
            strength_b = float(strengths.get(focus_b, 0.0))
            if strength_a <= 0 or strength_b <= 0:
                continue
            norm_a = {token: value / strength_a for token, value in raw_a.items()}
            norm_b = {token: value / strength_b for token, value in raw_b.items()}
            rank_a = rank_vector(norm_a, excluded)
            rank_b = rank_vector(norm_b, excluded)
            pair_id = f"{focus_a}-{focus_b}"
            for scope_name, topn in neighbor_scopes:
                tokens_a = top_tokens(norm_a, excluded, topn)
                tokens_b = top_tokens(norm_b, excluded, topn)
                if candidate_scope == "shared":
                    candidates = tokens_a & tokens_b
                elif candidate_scope == "union":
                    candidates = tokens_a | tokens_b
                else:
                    raise ValueError(f"Unsupported candidate scope: {candidate_scope}")
                for token in sorted(candidates, key=lambda item: (-min(norm_a.get(item, 0.0), norm_b.get(item, 0.0)), item)):
                    edge_a = float(raw_a.get(token, 0.0))
                    edge_b = float(raw_b.get(token, 0.0))
                    for pair_count_mode in pair_count_modes:
                        if pair_count_mode == "sum":
                            pair_count = edge_a + edge_b
                        elif pair_count_mode == "min":
                            pair_count = min(edge_a, edge_b)
                        else:
                            raise ValueError(f"Unsupported pair count mode: {pair_count_mode}")
                        if pair_count <= 0:
                            continue
                        row = {
                            "token_profile": metadata["token_profile"],
                            "network_window": metadata["network_window"],
                            "period_set_id": period_set_id,
                            "period_id": period_id,
                            "period_label": meta.get("period_label", period_id),
                            "period_sort_order": meta.get("period_sort_order", 0),
                            "period_start_date": meta.get("period_start_date", ""),
                            "period_end_date": meta.get("period_end_date", ""),
                            "pair_id": pair_id,
                            "focus_a_norm": core_norm_for_token(focus_a),
                            "focus_a_label": focus_a,
                            "focus_b_norm": core_norm_for_token(focus_b),
                            "focus_b_label": focus_b,
                            "candidate_scope": candidate_scope,
                            "neighbor_scope": scope_name,
                            "topn": topn,
                            "pair_count_mode": pair_count_mode,
                            "token": token,
                            "pair_count": pair_count,
                            "edge_a": edge_a,
                            "edge_b": edge_b,
                            "norm_a": norm_a.get(token, 0.0),
                            "norm_b": norm_b.get(token, 0.0),
                            "shared_strength": min(norm_a.get(token, 0.0), norm_b.get(token, 0.0)),
                            "rank_a": rank_a.get(token, ""),
                            "rank_b": rank_b.get(token, ""),
                            "a_strength": strength_a,
                            "b_strength": strength_b,
                            "a_neighbor_count": neighbor_counts.get(focus_a, 0),
                            "b_neighbor_count": neighbor_counts.get(focus_b, 0),
                            "source_file": source_file,
                        }
                        rows.append(row)
    return rows


def two_by_two_log_likelihood(a: float, n1: float, c: float, n2: float) -> float:
    b = n1 - a
    d = n2 - c
    total = n1 + n2
    if total <= 0:
        return 0.0
    col1 = a + c
    col2 = b + d
    expected = [
        n1 * col1 / total,
        n1 * col2 / total,
        n2 * col1 / total,
        n2 * col2 / total,
    ]
    observed = [a, b, c, d]
    value = 0.0
    for obs, exp in zip(observed, expected):
        if obs > 0 and exp > 0:
            value += obs * math.log(obs / exp)
    return 2.0 * value


def two_by_two_chi_square(a: float, n1: float, c: float, n2: float) -> float:
    b = n1 - a
    d = n2 - c
    total = n1 + n2
    denom = (a + c) * (b + d) * n1 * n2
    if total <= 0 or denom <= 0:
        return 0.0
    return total * ((a * d - b * c) ** 2) / denom


def log_ratio(a: float, n1: float, c: float, n2: float, smoothing: float) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    rate_a = (a + smoothing) / (n1 + smoothing)
    rate_c = (c + smoothing) / (n2 + smoothing)
    if rate_a <= 0 or rate_c <= 0:
        return 0.0
    return math.log2(rate_a / rate_c)


def compute_log_odds_rows(
    matrix: pd.DataFrame,
    unit_meta: pd.DataFrame,
    comparison_type: str,
    prior_strength: float,
    alpha_floor: float,
    min_count: int,
    min_total_count: int,
    log_ratio_smoothing: float,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    unit_ids = [str(value) for value in unit_meta["unit_id"].tolist()]
    unit_meta_by_id = unit_meta.set_index("unit_id").to_dict(orient="index")
    background = matrix.sum(axis=1)
    background_total = float(background.sum())
    if background_total <= 0:
        return pd.DataFrame()
    alpha = prior_strength * (background / background_total)
    alpha = alpha.clip(lower=alpha_floor)
    alpha0 = float(alpha.sum())
    doc_freq = (matrix[unit_ids] > 0).sum(axis=1)
    num_docs = len(unit_ids)

    for unit_id in unit_ids:
        y_i = matrix[unit_id]
        y_j = background - y_i
        n_i = float(y_i.sum())
        n_j = float(y_j.sum())
        if n_i <= 0 or n_j <= 0:
            continue
        numerator_i = y_i + alpha
        numerator_j = y_j + alpha
        denominator_i = n_i + alpha0 - y_i - alpha
        denominator_j = n_j + alpha0 - y_j - alpha
        valid = (denominator_i > 0) & (denominator_j > 0)
        valid &= (pd.concat([y_i, y_j], axis=1).max(axis=1) >= min_count) & ((y_i + y_j) >= min_total_count)
        if not valid.any():
            continue

        delta = (numerator_i[valid] / denominator_i[valid]).map(math.log) - (
            numerator_j[valid] / denominator_j[valid]
        ).map(math.log)
        variance = (1.0 / numerator_i[valid]) + (1.0 / numerator_j[valid])
        z = delta / variance.map(math.sqrt)
        count_period = y_i[valid].astype(float)
        count_ref = y_j[valid].astype(float)
        tf = count_period / n_i
        idf = ((1.0 + num_docs) / (1.0 + doc_freq[valid])).map(math.log) + 1.0
        meta = unit_meta_by_id[unit_id]
        part = pd.DataFrame(
            {
                "token": matrix.index[valid],
                "comparison_type": comparison_type,
                "comparison_id": "all_other",
                "count_period": count_period.to_numpy(),
                "count_ref": count_ref.to_numpy(),
                "total_period": n_i,
                "total_ref": n_j,
                "rate_period": (count_period / n_i).to_numpy(),
                "rate_ref": (count_ref / n_j).to_numpy(),
                "log_odds_delta": delta.to_numpy(),
                "log_odds_z": z.to_numpy(),
                "log_likelihood": [
                    two_by_two_log_likelihood(a, n_i, c, n_j) for a, c in zip(count_period, count_ref)
                ],
                "log_ratio": [log_ratio(a, n_i, c, n_j, log_ratio_smoothing) for a, c in zip(count_period, count_ref)],
                "chi_square": [two_by_two_chi_square(a, n_i, c, n_j) for a, c in zip(count_period, count_ref)],
                "tfidf": (tf * idf).to_numpy(),
            }
        )
        for key, value in meta.items():
            if key != "unit_id":
                part[key] = value
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_keyness_for_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    unit_cols: Sequence[str],
    comparison_type: str,
    prior_strength: float,
    alpha_floor: float,
    min_count: int,
    min_total_count: int,
    log_ratio_smoothing: float,
) -> pd.DataFrame:
    out: List[pd.DataFrame] = []
    for _group_key, group in df.groupby(list(group_cols), dropna=False):
        unit_meta = group[list(unit_cols)].drop_duplicates().copy()
        if len(unit_meta) < 2:
            continue
        unit_meta["unit_id"] = unit_meta[list(unit_cols)].astype(str).agg("||".join, axis=1)
        unit_lookup = unit_meta.set_index(list(unit_cols))["unit_id"].to_dict()
        work = group.copy()
        work["unit_id"] = work.apply(lambda row: unit_lookup[tuple(row[col] for col in unit_cols)], axis=1)
        matrix = work.pivot_table(index="token", columns="unit_id", values="pair_count", aggfunc="sum", fill_value=0.0)
        for unit_id in unit_meta["unit_id"]:
            if unit_id not in matrix.columns:
                matrix[unit_id] = 0.0
        matrix = matrix[[str(unit_id) for unit_id in unit_meta["unit_id"].tolist()]].astype(float)
        part = compute_log_odds_rows(
            matrix=matrix,
            unit_meta=unit_meta,
            comparison_type=comparison_type,
            prior_strength=prior_strength,
            alpha_floor=alpha_floor,
            min_count=min_count,
            min_total_count=min_total_count,
            log_ratio_smoothing=log_ratio_smoothing,
        )
        if not part.empty:
            for col in group_cols:
                part[col] = group[col].iloc[0]
            out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_group_cols = [
        "token_profile",
        "network_window",
        "period_set_id",
        "period_id",
        "pair_id",
        "candidate_scope",
        "neighbor_scope",
        "pair_count_mode",
        "comparison_type",
    ]
    df = df.copy()
    df["direction"] = df["log_odds_z"].map(lambda value: "positive" if float(value) >= 0 else "negative")
    grouped = df.groupby(sort_group_cols, dropna=False)
    df["_abs_log_odds_z"] = df["log_odds_z"].abs()
    df["rank_abs_z"] = grouped["_abs_log_odds_z"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df["rank_positive"] = grouped["log_odds_z"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df["rank_negative"] = grouped["log_odds_z"].rank(method="first", ascending=True, na_option="bottom").astype(int)
    df["rank_log_likelihood"] = grouped["log_likelihood"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df["rank_log_ratio"] = grouped["log_ratio"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df["rank_chi_square"] = grouped["chi_square"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df["rank_tfidf"] = grouped["tfidf"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    df = df.drop(columns=["_abs_log_odds_z"])
    return df


def merge_pair_token_metadata(keyness_df: pd.DataFrame, token_df: pd.DataFrame) -> pd.DataFrame:
    if keyness_df.empty:
        return keyness_df
    merge_cols = [
        "token_profile",
        "network_window",
        "period_set_id",
        "period_id",
        "pair_id",
        "candidate_scope",
        "neighbor_scope",
        "topn",
        "pair_count_mode",
        "token",
    ]
    meta_cols = merge_cols + [
        "edge_a",
        "edge_b",
        "norm_a",
        "norm_b",
        "shared_strength",
        "rank_a",
        "rank_b",
        "a_strength",
        "b_strength",
        "a_neighbor_count",
        "b_neighbor_count",
        "source_file",
    ]
    meta = token_df[meta_cols].drop_duplicates(subset=merge_cols)
    return keyness_df.merge(meta, on=merge_cols, how="left")


def load_period_keyness(path: Optional[Path], comparison_type: str, count_mode: str) -> pd.DataFrame:
    if not path or not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[(df["comparison_type"] == comparison_type) & (df["count_mode"] == count_mode)].copy()
    if df.empty:
        return pd.DataFrame()
    rename = {
        "log_odds_z": "period_keyness_log_odds_z",
        "rank_positive": "period_keyness_rank_positive",
        "rank_negative": "period_keyness_rank_negative",
        "count_period": "period_keyness_count",
        "rate_period": "period_keyness_rate",
    }
    keep = [
        "token",
        "token_profile",
        "period_id",
        "log_odds_z",
        "rank_positive",
        "rank_negative",
        "count_period",
        "rate_period",
        "dominant_pos",
        "dict_lv1",
        "dict_lv2",
        "dict_ner_like_type",
        "is_core_exact",
        "is_core_contain",
        "matched_core_keyword",
    ]
    return df[[col for col in keep if col in df.columns]].rename(columns=rename)


def merge_period_keyness(
    keyness_df: pd.DataFrame,
    period_keyness_df: pd.DataFrame,
) -> pd.DataFrame:
    if keyness_df.empty or period_keyness_df.empty:
        return keyness_df
    return keyness_df.merge(period_keyness_df, on=["token", "token_profile", "period_id"], how="left")


def compute_keyness(
    token_rows: Sequence[Dict[str, object]],
    prior_strength: float,
    alpha_floor: float,
    min_count: int,
    min_total_count: int,
    log_ratio_smoothing: float,
) -> pd.DataFrame:
    token_df = pd.DataFrame(token_rows)
    if token_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        "network_window",
        "period_sort_order",
        "topn",
        "pair_count",
        "edge_a",
        "edge_b",
        "norm_a",
        "norm_b",
        "shared_strength",
        "a_strength",
        "b_strength",
        "a_neighbor_count",
        "b_neighbor_count",
    ]
    for col in numeric_cols:
        if col in token_df.columns:
            token_df[col] = pd.to_numeric(token_df[col], errors="coerce").fillna(0)

    fixed_group_cols = [
        "token_profile",
        "network_window",
        "period_set_id",
        "candidate_scope",
        "neighbor_scope",
        "topn",
        "pair_count_mode",
    ]
    period_only_df = token_df[~token_df["period_set_id"].astype(str).str.startswith("global")].copy()
    period_within_pair = build_keyness_for_group(
        df=period_only_df,
        group_cols=fixed_group_cols + ["pair_id"],
        unit_cols=[
            "period_id",
            "period_label",
            "period_sort_order",
            "period_start_date",
            "period_end_date",
            "pair_id",
            "focus_a_norm",
            "focus_a_label",
            "focus_b_norm",
            "focus_b_label",
        ],
        comparison_type="same_pair_other_periods",
        prior_strength=prior_strength,
        alpha_floor=alpha_floor,
        min_count=min_count,
        min_total_count=min_total_count,
        log_ratio_smoothing=log_ratio_smoothing,
    )
    pair_within_period = build_keyness_for_group(
        df=token_df,
        group_cols=fixed_group_cols + [
            "period_id",
            "period_label",
            "period_sort_order",
            "period_start_date",
            "period_end_date",
        ],
        unit_cols=["pair_id", "focus_a_norm", "focus_a_label", "focus_b_norm", "focus_b_label"],
        comparison_type="same_period_other_pairs",
        prior_strength=prior_strength,
        alpha_floor=alpha_floor,
        min_count=min_count,
        min_total_count=min_total_count,
        log_ratio_smoothing=log_ratio_smoothing,
    )
    keyness = pd.concat([period_within_pair, pair_within_period], ignore_index=True)
    if keyness.empty:
        return keyness
    keyness = add_ranks(keyness)
    keyness = merge_pair_token_metadata(keyness, token_df)
    return keyness


def compute_robust_candidates(
    keyness_df: pd.DataFrame,
    robust_topn: int,
    robust_min_metrics: int,
) -> pd.DataFrame:
    if keyness_df.empty:
        return pd.DataFrame()
    group_cols = [
        "token_profile",
        "network_window",
        "period_set_id",
        "period_id",
        "period_label",
        "period_sort_order",
        "period_start_date",
        "period_end_date",
        "pair_id",
        "focus_a_norm",
        "focus_a_label",
        "focus_b_norm",
        "focus_b_label",
        "candidate_scope",
        "neighbor_scope",
        "topn",
        "pair_count_mode",
        "comparison_type",
        "comparison_id",
    ]
    rows: List[Dict[str, object]] = []
    for _group_key, group in keyness_df[keyness_df["direction"] == "positive"].groupby(group_cols, dropna=False):
        if group.empty:
            continue
        included_by_token: DefaultDict[str, List[str]] = defaultdict(list)
        for metric in METRICS_FOR_ROBUSTNESS:
            ranked = group.sort_values(metric, ascending=False, kind="mergesort").head(robust_topn)
            for token in ranked["token"]:
                included_by_token[str(token)].append(metric)
        for token, metrics in included_by_token.items():
            if len(metrics) < robust_min_metrics:
                continue
            base = group[group["token"] == token].iloc[0].to_dict()
            robust_score = len(metrics)
            base["robust_topn"] = robust_topn
            base["robust_min_metrics"] = robust_min_metrics
            base["robust_score"] = robust_score
            base["robust_class"] = "strong" if robust_score >= 4 else "stable"
            base["included_metrics"] = ";".join(metrics)
            rows.append(base)
    if not rows:
        return pd.DataFrame()
    robust = pd.DataFrame(rows)
    robust = robust.sort_values(
        [
            "token_profile",
            "network_window",
            "period_sort_order",
            "pair_id",
            "neighbor_scope",
            "pair_count_mode",
            "comparison_type",
            "robust_score",
            "log_odds_z",
        ],
        ascending=[True, True, True, True, True, True, True, False, False],
    )
    return robust


def build_payload(
    keyness_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    args: argparse.Namespace,
    output_files: Dict[str, str],
) -> Dict[str, object]:
    keyness_rows = keyness_df.to_dict(orient="records") if not keyness_df.empty else []
    robust_rows = robust_df.to_dict(orient="records") if not robust_df.empty else []
    return {
        "meta": {
            "method": "pair_conditioned_keyness",
            "primary_metric": "log_odds_z",
            "secondary_metrics": ["log_likelihood", "log_ratio", "chi_square", "tfidf"],
            "keyness_method": "log_odds_ratio_informative_prior",
            "comparison_types": ["same_pair_other_periods", "same_period_other_pairs"],
            "count_definition": {
                "sum": "pair_count = edge_a + edge_b",
                "min": "pair_count = min(edge_a, edge_b)",
            },
            "pair_count_modes": args.pair_count_modes,
            "balance_definition": "shared_strength = min(norm_a, norm_b)",
            "period_set_id": args.period_set_id,
            "candidate_scope": args.candidate_scope,
            "neighbor_scopes": args.neighbor_scopes,
            "prior_strength": args.prior_strength,
            "alpha_floor": args.alpha_floor,
            "min_count": args.min_count,
            "min_total_count": args.min_total_count,
            "robust_topn": args.robust_topn,
            "robust_min_metrics": args.robust_min_metrics,
            "output_files": output_files,
        },
        "keyness_rows": json_ready(keyness_rows),
        "robust_rows": json_ready(robust_rows),
    }


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "kind",
        "path",
        "status",
        "row_count",
        "token_profile",
        "network_window",
        "source_file",
    ]
    write_csv(path, rows, fieldnames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pair-conditioned keyness from Shenbao undirected network edge CSVs."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing undirected edge CSVs.")
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Specific edge CSV. Can be passed multiple times. If omitted, files are discovered from --input-dir.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--period-set-id", default="long_period_manual", help="Period set id to process.")
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Also include global_all edge CSVs. These contribute to same_period_other_pairs only.",
    )
    parser.add_argument(
        "--global-input-dir",
        default=str(DEFAULT_GLOBAL_INPUT_DIR),
        help="Directory containing global_all undirected edge CSVs used with --include-global.",
    )
    parser.add_argument(
        "--period-override-dir",
        default=str(DEFAULT_PERIOD_OVERRIDE_DIR),
        help=(
            "Directory containing revised single-period edge CSVs. Matching rows replace the same "
            "profile/window/period rows from --input-dir. Use an empty string to disable."
        ),
    )
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES), help="Profiles to discover when --input-csv is omitted.")
    parser.add_argument("--windows", default="1,5,10,20", help="Windows to discover when --input-csv is omitted.")
    parser.add_argument("--pairs", action="append", default=[], help="Pairs such as \u7acb\u61b2:\u61b2\u653f or lixian:xianzheng.")
    parser.add_argument(
        "--candidate-scope",
        choices=["shared", "union"],
        default="shared",
        help="Use tokens connected to both focus tokens, or either focus token.",
    )
    parser.add_argument(
        "--neighbor-scopes",
        default=",".join(DEFAULT_NEIGHBOR_SCOPES),
        help="Comma-separated scopes: all,top20,top50,top100.",
    )
    parser.add_argument(
        "--pair-count-modes",
        default="sum,min",
        help="Comma-separated pair count modes: sum,min. raw_min is accepted as an alias for min.",
    )
    parser.add_argument("--weight-col", default="joint_count", help="Edge weight column.")
    parser.add_argument("--prior-strength", type=float, default=1000.0, help="Informative prior strength.")
    parser.add_argument("--alpha-floor", type=float, default=0.01, help="Minimum token prior.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum max(target, reference) count for keyness.")
    parser.add_argument("--min-total-count", type=int, default=10, help="Minimum target + reference count for keyness.")
    parser.add_argument("--log-ratio-smoothing", type=float, default=0.5, help="Smoothing for log ratio.")
    parser.add_argument("--robust-topn", type=int, default=30, help="Top N per metric for robust candidate scoring.")
    parser.add_argument("--robust-min-metrics", type=int, default=3, help="Minimum number of metric top-N lists.")
    parser.add_argument(
        "--period-keyness-csv",
        default=str(DEFAULT_PERIOD_KEYNESS_CSV),
        help="Existing period keyness CSV to merge as background. Use empty string to disable.",
    )
    parser.add_argument("--period-keyness-comparison", default="all_other", help="Period keyness comparison_type to merge.")
    parser.add_argument("--period-keyness-count-mode", default="mention", help="Period keyness count_mode to merge.")
    parser.add_argument("--output-prefix", default="pair_conditioned_keyness", help="Output file prefix.")
    parser.add_argument(
        "--max-json-keyness-rows",
        type=int,
        default=100000,
        help="Limit keyness rows in JSON payload. 0 means all rows. CSV outputs are always complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [normalize_profile(value) for value in parse_list(args.profiles)]
    windows = parse_int_list(args.windows)
    input_csvs = [Path(value).expanduser().resolve() for value in args.input_csv]
    if not input_csvs:
        input_csvs = discover_input_csvs(Path(args.input_dir), profiles, windows, args.period_set_id)
    if args.include_global:
        input_csvs.extend(discover_global_input_csvs(Path(args.global_input_dir), profiles, windows))
        input_csvs = unique_paths(input_csvs)
    if not input_csvs:
        raise FileNotFoundError("No input edge CSVs found.")

    pairs = parse_pairs(args.pairs)
    neighbor_scopes = parse_neighbor_scopes(parse_list(args.neighbor_scopes))
    pair_count_modes = parse_pair_count_modes(parse_list(args.pair_count_modes))
    args.pair_count_modes = pair_count_modes
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    token_rows: List[Dict[str, object]] = []
    for input_csv in input_csvs:
        log(f"Loading pair token counts: {input_csv}")
        rows = build_pair_token_rows(
            input_csv=input_csv,
            pairs=pairs,
            candidate_scope=args.candidate_scope,
            neighbor_scopes=neighbor_scopes,
            pair_count_modes=pair_count_modes,
            weight_col=args.weight_col,
        )
        token_rows.extend(rows)
        manifest_rows.append(
            {
                "kind": "input_edges",
                "path": str(input_csv),
                "status": "ok",
                "row_count": len(rows),
                "token_profile": parse_source_metadata(input_csv).get("token_profile", ""),
                "network_window": parse_source_metadata(input_csv).get("network_window", ""),
                "source_file": str(input_csv),
            }
        )

    override_csvs = (
        discover_period_override_csvs(Path(args.period_override_dir), profiles, windows, args.period_set_id)
        if args.period_override_dir
        else []
    )
    if override_csvs:
        override_rows: List[Dict[str, object]] = []
        for input_csv in override_csvs:
            log(f"Loading revised period token counts: {input_csv}")
            rows = build_pair_token_rows(
                input_csv=input_csv,
                pairs=pairs,
                candidate_scope=args.candidate_scope,
                neighbor_scopes=neighbor_scopes,
                pair_count_modes=pair_count_modes,
                weight_col=args.weight_col,
            )
            override_rows.extend(rows)
            metadata = parse_source_metadata(input_csv)
            manifest_rows.append(
                {
                    "kind": "input_edges_period_override",
                    "path": str(input_csv),
                    "status": "ok",
                    "row_count": len(rows),
                    "token_profile": metadata.get("token_profile", ""),
                    "network_window": metadata.get("network_window", ""),
                    "source_file": str(input_csv),
                }
            )
        override_keys = {
            (
                str(row["token_profile"]),
                str(row["network_window"]),
                str(row["period_set_id"]),
                str(row["period_id"]),
            )
            for row in override_rows
        }
        token_rows = [
            row
            for row in token_rows
            if (
                str(row["token_profile"]),
                str(row["network_window"]),
                str(row["period_set_id"]),
                str(row["period_id"]),
            )
            not in override_keys
        ]
        token_rows.extend(override_rows)

    token_counts_path = output_dir / f"{args.output_prefix}_token_counts.csv"
    write_csv(token_counts_path, token_rows, TOKEN_COUNT_FIELDNAMES)
    log(f"Wrote token counts: {token_counts_path} ({len(token_rows)} rows)")

    log("Computing pair-conditioned keyness.")
    keyness_df = compute_keyness(
        token_rows=token_rows,
        prior_strength=args.prior_strength,
        alpha_floor=args.alpha_floor,
        min_count=args.min_count,
        min_total_count=args.min_total_count,
        log_ratio_smoothing=args.log_ratio_smoothing,
    )

    period_keyness_path = Path(args.period_keyness_csv).expanduser().resolve() if args.period_keyness_csv else None
    period_keyness_df = load_period_keyness(
        period_keyness_path,
        comparison_type=args.period_keyness_comparison,
        count_mode=args.period_keyness_count_mode,
    )
    if not period_keyness_df.empty:
        keyness_df = merge_period_keyness(keyness_df, period_keyness_df)

    if not keyness_df.empty:
        keyness_df = keyness_df.sort_values(
            [
                "token_profile",
                "network_window",
                "period_sort_order",
                "pair_id",
                "neighbor_scope",
                "pair_count_mode",
                "comparison_type",
                "rank_positive",
                "token",
            ],
            ascending=[True, True, True, True, True, True, True, True, True],
        )
    keyness_path = output_dir / f"{args.output_prefix}.csv"
    write_csv(keyness_path, keyness_df.to_dict(orient="records"), KEYNESS_FIELDNAMES)
    log(f"Wrote keyness: {keyness_path} ({len(keyness_df)} rows)")

    log("Computing robust candidates.")
    robust_df = compute_robust_candidates(
        keyness_df=keyness_df,
        robust_topn=args.robust_topn,
        robust_min_metrics=args.robust_min_metrics,
    )
    robust_path = output_dir / f"{args.output_prefix}_robust_candidates.csv"
    write_csv(robust_path, robust_df.to_dict(orient="records") if not robust_df.empty else [], ROBUST_FIELDNAMES)
    log(f"Wrote robust candidates: {robust_path} ({len(robust_df)} rows)")

    payload_keyness_df = keyness_df
    if args.max_json_keyness_rows > 0 and len(payload_keyness_df) > args.max_json_keyness_rows:
        payload_keyness_df = payload_keyness_df.head(args.max_json_keyness_rows).copy()
    payload_path = output_dir / f"{args.output_prefix}_payload.json"
    output_files = {
        "token_counts_csv": str(token_counts_path),
        "keyness_csv": str(keyness_path),
        "robust_candidates_csv": str(robust_path),
        "payload_json": str(payload_path),
    }
    payload = build_payload(payload_keyness_df, robust_df, args, output_files)
    with payload_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    log(f"Wrote payload: {payload_path}")

    manifest_rows.extend(
        [
            {"kind": "token_counts", "path": str(token_counts_path), "status": "ok", "row_count": len(token_rows)},
            {"kind": "keyness", "path": str(keyness_path), "status": "ok", "row_count": len(keyness_df)},
            {"kind": "robust_candidates", "path": str(robust_path), "status": "ok", "row_count": len(robust_df)},
            {"kind": "payload", "path": str(payload_path), "status": "ok", "row_count": len(payload_keyness_df)},
        ]
    )
    manifest_path = output_dir / f"{args.output_prefix}_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    log(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
