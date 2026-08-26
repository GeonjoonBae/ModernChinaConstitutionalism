#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate overlap metrics directly from Shenbao network edge CSV files.

The script accepts one applied-network edge CSV from ``shenbao_network``. It is
intended to work with the undirected CSVs generated under ``undirected/``. If a
directed split-lr CSV is passed by accident, only ``direction == R`` rows are
used, matching the existing undirected conversion logic.
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


CORE_ORDER = ["lixian", "xianzheng", "xianfa", "zhixian"]
CORE_TOKENS = {
    "lixian": "\u7acb\u61b2",
    "xianzheng": "\u61b2\u653f",
    "xianfa": "\u61b2\u6cd5",
    "zhixian": "\u5236\u61b2",
}
CORE_LABELS = {
    "lixian": "\u7acb\u61b2",
    "xianzheng": "\u61b2\u653f",
    "xianfa": "\u61b2\u6cd5",
    "zhixian": "\u5236\u61b2",
}
INTRA_PAIRS = {("lixian", "xianzheng"), ("xianfa", "zhixian")}

CORE_FIELDNAMES = [
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "unit_type",
    "pair_class",
    "topn",
    "weight_col",
    "min_token_strength",
    "min_neighbor_count",
    "focus_a_norm",
    "focus_a_label",
    "focus_b_norm",
    "focus_b_label",
    "direct_strength",
    "p_b_given_a",
    "p_a_given_b",
    "weighted_jaccard",
    "jaccard",
    "cosine",
    "shared_neighbor_count",
    "shared_neighbors",
    "a_strength",
    "b_strength",
    "a_neighbor_count",
    "b_neighbor_count",
    "a_used_neighbor_count",
    "b_used_neighbor_count",
    "support_status",
    "support_notes",
    "source_file",
]

GROUP_FIELDNAMES = [
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "unit_type",
    "pair_class",
    "topn",
    "weight_col",
    "group_weight",
    "min_token_strength",
    "min_neighbor_count",
    "group_a_norm",
    "group_a_label",
    "group_b_norm",
    "group_b_label",
    "direct_strength",
    "p_b_given_a",
    "p_a_given_b",
    "weighted_jaccard",
    "jaccard",
    "cosine",
    "shared_neighbor_count",
    "shared_neighbors",
    "group_a_strength",
    "group_b_strength",
    "group_a_token_count",
    "group_b_token_count",
    "group_a_neighbor_count",
    "group_b_neighbor_count",
    "group_a_used_neighbor_count",
    "group_b_used_neighbor_count",
    "group_a_top_tokens",
    "group_b_top_tokens",
    "support_status",
    "support_notes",
    "source_file",
]

SUMMARY_FIELDNAMES = [
    "token_profile",
    "network_window",
    "period_set_id",
    "period_id",
    "unit_type",
    "metric",
    "topn",
    "weight_col",
    "group_weight",
    "pair_count",
    "low_support_pair_count",
    "intra_mean",
    "cross_mean",
    "gap_intra_minus_cross",
    "source_file",
]


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
        description="Calculate core and core-group network overlap metrics from one Shenbao edge CSV."
    )
    parser.add_argument("--input-csv", required=True, help="One network edge CSV in shenbao_network.")
    parser.add_argument(
        "--output-dir",
        help="Output folder. Default: <input-csv parent>/overlap_metrics",
    )
    parser.add_argument("--topn", type=int, default=100, help="Top neighbors per vector. Use 0 for all.")
    parser.add_argument(
        "--weight-col",
        default="joint_count",
        help="Edge weight column. Default: joint_count.",
    )
    parser.add_argument(
        "--group-weight",
        choices=["log", "raw", "equal"],
        default="log",
        help="Internal token weighting for core-containing token groups.",
    )
    parser.add_argument(
        "--min-token-strength",
        type=float,
        default=30.0,
        help="Low-support flag threshold for token/group total incident edge weight.",
    )
    parser.add_argument(
        "--min-neighbor-count",
        type=int,
        default=10,
        help="Low-support flag threshold for available neighbor count.",
    )
    parser.add_argument(
        "--max-shared-neighbors",
        type=int,
        default=20,
        help="Maximum shared-neighbor labels to write.",
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if fieldnames and fieldnames[0].startswith("\ufeff"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    return rows, fieldnames


def parse_float(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def canonical_pair(token_a: str, token_b: str) -> Tuple[str, str]:
    if token_a <= token_b:
        return token_a, token_b
    return token_b, token_a


def parse_source_metadata(input_csv: Path) -> Dict[str, str]:
    name = input_csv.name
    profile_match = re.search(r"_applied_(strict|full|regex-only)_", name)
    window_match = re.search(r"_w(\d+)_", name)
    return {
        "source_file": str(input_csv),
        "token_profile": profile_match.group(1) if profile_match else "",
        "network_window": window_match.group(1) if window_match else "",
    }


def should_use_row(row: Dict[str, str]) -> bool:
    direction = row.get("direction", "").strip()
    if not direction or direction == "U":
        return True
    return direction == "R"


def load_edges(
    input_csv: Path,
    weight_col: str,
) -> Dict[Tuple[str, str], Dict[Tuple[str, str], float]]:
    rows, fieldnames = read_csv_dicts(input_csv)
    required = {"period_set_id", "period_id", "center_token", "neighbor_token", weight_col}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    period_edges: DefaultDict[Tuple[str, str], DefaultDict[Tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        if not should_use_row(row):
            continue
        token_a = row["center_token"].strip()
        token_b = row["neighbor_token"].strip()
        if not token_a or not token_b or token_a == token_b:
            continue
        weight = parse_float(row.get(weight_col))
        if weight <= 0:
            continue
        period_key = (row["period_set_id"].strip(), row["period_id"].strip())
        period_edges[period_key][canonical_pair(token_a, token_b)] += weight
    return {period: dict(edges) for period, edges in period_edges.items()}


def build_vectors(
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

    normalized: Dict[str, Dict[str, float]] = {}
    neighbor_counts: Dict[str, int] = {}
    for token, vector in raw_vectors.items():
        denom = strengths[token]
        normalized[token] = {neighbor: value / denom for neighbor, value in vector.items()} if denom else {}
        neighbor_counts[token] = len(vector)
    return normalized, dict(strengths), neighbor_counts, tokens


def top_vector(vector: Dict[str, float], excluded: Set[str], topn: int) -> Dict[str, float]:
    items = [(token, value) for token, value in vector.items() if token not in excluded and value > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    if topn > 0:
        items = items[:topn]
    return dict(items)


def weighted_jaccard(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def jaccard(left: Dict[str, float], right: Dict[str, float]) -> float:
    left_keys = {key for key, value in left.items() if value > 0}
    right_keys = {key for key, value in right.items() if value > 0}
    union = left_keys | right_keys
    return len(left_keys & right_keys) / len(union) if union else 0.0


def cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    denom = norm_left * norm_right
    return dot / denom if denom else 0.0


def shared_neighbor_label(
    left: Dict[str, float],
    right: Dict[str, float],
    max_items: int,
) -> Tuple[int, str]:
    shared = [(token, min(left[token], right[token])) for token in set(left) & set(right)]
    shared.sort(key=lambda item: (-item[1], item[0]))
    if max_items > 0:
        shown = shared[:max_items]
    else:
        shown = shared
    return len(shared), "; ".join(token for token, _ in shown)


def pair_class(left_norm: str, right_norm: str) -> str:
    pair = tuple(sorted((left_norm, right_norm), key=CORE_ORDER.index))
    return "intra" if pair in INTRA_PAIRS else "cross"


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def support_status(notes: Sequence[str]) -> Tuple[str, str]:
    clean_notes = [note for note in notes if note]
    if clean_notes:
        return "low_support", "; ".join(clean_notes)
    return "ok", ""


def edge_weight(edges: Dict[Tuple[str, str], float], token_a: str, token_b: str) -> float:
    return edges.get(canonical_pair(token_a, token_b), 0.0)


def direct_strength_from_prob(left_to_right: float, right_to_left: float) -> float:
    return math.sqrt(left_to_right * right_to_left) if left_to_right > 0 and right_to_left > 0 else 0.0


def calculate_core_rows(
    metadata: Dict[str, str],
    period_key: Tuple[str, str],
    edges: Dict[Tuple[str, str], float],
    vectors: Dict[str, Dict[str, float]],
    strengths: Dict[str, float],
    neighbor_counts: Dict[str, int],
    topn: int,
    weight_col: str,
    min_token_strength: float,
    min_neighbor_count: int,
    max_shared_neighbors: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    excluded = set(CORE_TOKENS.values())
    for left_norm, right_norm in combinations(CORE_ORDER, 2):
        left_token = CORE_TOKENS[left_norm]
        right_token = CORE_TOKENS[right_norm]
        left_strength = strengths.get(left_token, 0.0)
        right_strength = strengths.get(right_token, 0.0)
        left_neighbors = neighbor_counts.get(left_token, 0)
        right_neighbors = neighbor_counts.get(right_token, 0)
        direct = edge_weight(edges, left_token, right_token)
        p_right_given_left = direct / left_strength if left_strength else 0.0
        p_left_given_right = direct / right_strength if right_strength else 0.0
        left_vector = top_vector(vectors.get(left_token, {}), excluded, topn)
        right_vector = top_vector(vectors.get(right_token, {}), excluded, topn)
        shared_count, shared_neighbors = shared_neighbor_label(left_vector, right_vector, max_shared_neighbors)
        notes: List[str] = []
        if left_strength < min_token_strength:
            notes.append(f"{left_norm}_strength<{min_token_strength:g}")
        if right_strength < min_token_strength:
            notes.append(f"{right_norm}_strength<{min_token_strength:g}")
        if left_neighbors < min_neighbor_count:
            notes.append(f"{left_norm}_neighbors<{min_neighbor_count}")
        if right_neighbors < min_neighbor_count:
            notes.append(f"{right_norm}_neighbors<{min_neighbor_count}")
        status, note_text = support_status(notes)
        period_set_id, period_id = period_key
        row = {
            **metadata,
            "period_set_id": period_set_id,
            "period_id": period_id,
            "unit_type": "core",
            "pair_class": pair_class(left_norm, right_norm),
            "topn": topn,
            "weight_col": weight_col,
            "min_token_strength": min_token_strength,
            "min_neighbor_count": min_neighbor_count,
            "focus_a_norm": left_norm,
            "focus_a_label": CORE_LABELS[left_norm],
            "focus_b_norm": right_norm,
            "focus_b_label": CORE_LABELS[right_norm],
            "direct_strength": direct_strength_from_prob(p_right_given_left, p_left_given_right),
            "p_b_given_a": p_right_given_left,
            "p_a_given_b": p_left_given_right,
            "weighted_jaccard": weighted_jaccard(left_vector, right_vector),
            "jaccard": jaccard(left_vector, right_vector),
            "cosine": cosine(left_vector, right_vector),
            "shared_neighbor_count": shared_count,
            "shared_neighbors": shared_neighbors,
            "a_strength": left_strength,
            "b_strength": right_strength,
            "a_neighbor_count": left_neighbors,
            "b_neighbor_count": right_neighbors,
            "a_used_neighbor_count": len(left_vector),
            "b_used_neighbor_count": len(right_vector),
            "support_status": status,
            "support_notes": note_text,
        }
        rows.append(row)
    return rows


def group_tokens_for_period(tokens: Iterable[str]) -> Dict[str, Set[str]]:
    groups: Dict[str, Set[str]] = {norm: set() for norm in CORE_ORDER}
    for token in tokens:
        for norm, core_token in CORE_TOKENS.items():
            if core_token in token:
                groups[norm].add(token)
    return groups


def calculate_group_alphas(
    group_tokens: Set[str],
    strengths: Dict[str, float],
    group_weight: str,
) -> Dict[str, float]:
    if not group_tokens:
        return {}
    if group_weight == "equal":
        value = 1.0 / len(group_tokens)
        return {token: value for token in group_tokens}

    raw_weights: Dict[str, float] = {}
    for token in group_tokens:
        support = max(strengths.get(token, 0.0), 0.0)
        raw_weights[token] = math.log1p(support) if group_weight == "log" else support
    total = sum(raw_weights.values())
    if total <= 0:
        value = 1.0 / len(group_tokens)
        return {token: value for token in group_tokens}
    return {token: value / total for token, value in raw_weights.items()}


def build_group_vector(
    group_alphas: Dict[str, float],
    vectors: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    combined: DefaultDict[str, float] = defaultdict(float)
    for token, alpha in group_alphas.items():
        for neighbor, value in vectors.get(token, {}).items():
            combined[neighbor] += alpha * value
    return dict(combined)


def group_probability_to_target(
    source_alphas: Dict[str, float],
    target_tokens: Set[str],
    vectors: Dict[str, Dict[str, float]],
) -> float:
    total = 0.0
    for token, alpha in source_alphas.items():
        token_vector = vectors.get(token, {})
        total += alpha * sum(token_vector.get(target, 0.0) for target in target_tokens)
    return total


def top_tokens_label(alphas: Dict[str, float], max_items: int = 20) -> str:
    items = sorted(alphas.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    return "; ".join(f"{token}:{format_float(value)}" for token, value in items)


def calculate_group_rows(
    metadata: Dict[str, str],
    period_key: Tuple[str, str],
    vectors: Dict[str, Dict[str, float]],
    strengths: Dict[str, float],
    neighbor_counts: Dict[str, int],
    tokens: Set[str],
    topn: int,
    weight_col: str,
    group_weight: str,
    min_token_strength: float,
    min_neighbor_count: int,
    max_shared_neighbors: int,
) -> List[Dict[str, object]]:
    groups = group_tokens_for_period(tokens)
    alphas = {
        norm: calculate_group_alphas(group_tokens, strengths, group_weight)
        for norm, group_tokens in groups.items()
    }
    group_vectors = {norm: build_group_vector(alphas[norm], vectors) for norm in CORE_ORDER}
    rows: List[Dict[str, object]] = []

    for left_norm, right_norm in combinations(CORE_ORDER, 2):
        left_tokens = groups[left_norm]
        right_tokens = groups[right_norm]
        p_right_given_left = group_probability_to_target(alphas[left_norm], right_tokens, vectors)
        p_left_given_right = group_probability_to_target(alphas[right_norm], left_tokens, vectors)
        excluded = left_tokens | right_tokens
        left_vector = top_vector(group_vectors[left_norm], excluded, topn)
        right_vector = top_vector(group_vectors[right_norm], excluded, topn)
        shared_count, shared_neighbors = shared_neighbor_label(left_vector, right_vector, max_shared_neighbors)
        left_strength = sum(strengths.get(token, 0.0) for token in left_tokens)
        right_strength = sum(strengths.get(token, 0.0) for token in right_tokens)
        left_neighbor_count = len({neighbor for token in left_tokens for neighbor in vectors.get(token, {})})
        right_neighbor_count = len({neighbor for token in right_tokens for neighbor in vectors.get(token, {})})
        notes: List[str] = []
        if left_strength < min_token_strength:
            notes.append(f"{left_norm}_group_strength<{min_token_strength:g}")
        if right_strength < min_token_strength:
            notes.append(f"{right_norm}_group_strength<{min_token_strength:g}")
        if left_neighbor_count < min_neighbor_count:
            notes.append(f"{left_norm}_group_neighbors<{min_neighbor_count}")
        if right_neighbor_count < min_neighbor_count:
            notes.append(f"{right_norm}_group_neighbors<{min_neighbor_count}")
        status, note_text = support_status(notes)
        period_set_id, period_id = period_key
        rows.append(
            {
                **metadata,
                "period_set_id": period_set_id,
                "period_id": period_id,
                "unit_type": "group",
                "pair_class": pair_class(left_norm, right_norm),
                "topn": topn,
                "weight_col": weight_col,
                "group_weight": group_weight,
                "min_token_strength": min_token_strength,
                "min_neighbor_count": min_neighbor_count,
                "group_a_norm": left_norm,
                "group_a_label": CORE_LABELS[left_norm],
                "group_b_norm": right_norm,
                "group_b_label": CORE_LABELS[right_norm],
                "direct_strength": direct_strength_from_prob(p_right_given_left, p_left_given_right),
                "p_b_given_a": p_right_given_left,
                "p_a_given_b": p_left_given_right,
                "weighted_jaccard": weighted_jaccard(left_vector, right_vector),
                "jaccard": jaccard(left_vector, right_vector),
                "cosine": cosine(left_vector, right_vector),
                "shared_neighbor_count": shared_count,
                "shared_neighbors": shared_neighbors,
                "group_a_strength": left_strength,
                "group_b_strength": right_strength,
                "group_a_token_count": len(left_tokens),
                "group_b_token_count": len(right_tokens),
                "group_a_neighbor_count": left_neighbor_count,
                "group_b_neighbor_count": right_neighbor_count,
                "group_a_used_neighbor_count": len(left_vector),
                "group_b_used_neighbor_count": len(right_vector),
                "group_a_top_tokens": top_tokens_label(alphas[left_norm]),
                "group_b_top_tokens": top_tokens_label(alphas[right_norm]),
                "support_status": status,
                "support_notes": note_text,
            }
        )
    return rows


def mean(values: Sequence[float]) -> str:
    if not values:
        return ""
    return format_float(sum(values) / len(values))


def build_summary_rows(
    rows: Sequence[Dict[str, object]],
    unit_type: str,
    group_weight: str,
) -> List[Dict[str, object]]:
    if not rows:
        return []
    first = rows[0]
    summary_rows: List[Dict[str, object]] = []
    metrics = ["direct_strength", "weighted_jaccard", "jaccard", "cosine"]
    low_support_pair_count = sum(1 for row in rows if row["support_status"] != "ok")
    for metric in metrics:
        intra_values = [float(row[metric]) for row in rows if row["pair_class"] == "intra"]
        cross_values = [float(row[metric]) for row in rows if row["pair_class"] == "cross"]
        if intra_values and cross_values:
            gap = (sum(intra_values) / len(intra_values)) - (sum(cross_values) / len(cross_values))
            gap_text = format_float(gap)
        else:
            gap_text = ""
        summary_rows.append(
            {
                "source_file": first["source_file"],
                "token_profile": first["token_profile"],
                "network_window": first["network_window"],
                "period_set_id": first["period_set_id"],
                "period_id": first["period_id"],
                "unit_type": unit_type,
                "metric": metric,
                "topn": first["topn"],
                "weight_col": first["weight_col"],
                "group_weight": group_weight if unit_type == "group" else "",
                "pair_count": len(rows),
                "low_support_pair_count": low_support_pair_count,
                "intra_mean": mean(intra_values),
                "cross_mean": mean(cross_values),
                "gap_intra_minus_cross": gap_text,
            }
        )
    return summary_rows


def stringify_row(row: Dict[str, object], fieldnames: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field in fieldnames:
        value = row.get(field, "")
        if isinstance(value, float):
            out[field] = format_float(value)
        else:
            out[field] = str(value)
    return out


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(stringify_row(row, fieldnames))


def output_paths(input_csv: Path, output_dir: Path) -> Tuple[Path, Path, Path]:
    stem = input_csv.stem
    return (
        output_dir / f"{stem}_network_core_overlap_metrics.csv",
        output_dir / f"{stem}_network_group_overlap_metrics.csv",
        output_dir / f"{stem}_network_overlap_summary.csv",
    )


def calculate_metrics(
    input_csv: Path,
    topn: int = 100,
    weight_col: str = "joint_count",
    group_weight: str = "log",
    min_token_strength: float = 30.0,
    min_neighbor_count: int = 10,
    max_shared_neighbors: int = 20,
) -> Dict[str, object]:
    input_csv = input_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if topn < 0:
        raise ValueError("--topn must be >= 0")

    metadata = parse_source_metadata(input_csv)
    period_edges = load_edges(input_csv, weight_col)

    all_core_rows: List[Dict[str, object]] = []
    all_group_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []

    for period_key in sorted(period_edges):
        edges = period_edges[period_key]
        vectors, strengths, neighbor_counts, tokens = build_vectors(edges)
        core_rows = calculate_core_rows(
            metadata=metadata,
            period_key=period_key,
            edges=edges,
            vectors=vectors,
            strengths=strengths,
            neighbor_counts=neighbor_counts,
            topn=topn,
            weight_col=weight_col,
            min_token_strength=min_token_strength,
            min_neighbor_count=min_neighbor_count,
            max_shared_neighbors=max_shared_neighbors,
        )
        group_rows = calculate_group_rows(
            metadata=metadata,
            period_key=period_key,
            vectors=vectors,
            strengths=strengths,
            neighbor_counts=neighbor_counts,
            tokens=tokens,
            topn=topn,
            weight_col=weight_col,
            group_weight=group_weight,
            min_token_strength=min_token_strength,
            min_neighbor_count=min_neighbor_count,
            max_shared_neighbors=max_shared_neighbors,
        )
        all_core_rows.extend(core_rows)
        all_group_rows.extend(group_rows)
        all_summary_rows.extend(build_summary_rows(core_rows, "core", ""))
        all_summary_rows.extend(build_summary_rows(group_rows, "group", group_weight))

    return {
        "core_rows": all_core_rows,
        "group_rows": all_group_rows,
        "summary_rows": all_summary_rows,
    }


def calculate_file(
    input_csv: Path,
    output_dir: Path,
    topn: int = 100,
    weight_col: str = "joint_count",
    group_weight: str = "log",
    min_token_strength: float = 30.0,
    min_neighbor_count: int = 10,
    max_shared_neighbors: int = 20,
) -> Dict[str, object]:
    input_csv = input_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    result = calculate_metrics(
        input_csv=input_csv,
        topn=topn,
        weight_col=weight_col,
        group_weight=group_weight,
        min_token_strength=min_token_strength,
        min_neighbor_count=min_neighbor_count,
        max_shared_neighbors=max_shared_neighbors,
    )
    all_core_rows = result["core_rows"]
    all_group_rows = result["group_rows"]
    all_summary_rows = result["summary_rows"]
    core_path, group_path, summary_path = output_paths(input_csv, output_dir)
    write_csv(core_path, CORE_FIELDNAMES, all_core_rows)
    write_csv(group_path, GROUP_FIELDNAMES, all_group_rows)
    write_csv(summary_path, SUMMARY_FIELDNAMES, all_summary_rows)
    return {
        "core_path": core_path,
        "group_path": group_path,
        "summary_path": summary_path,
        "core_rows": len(all_core_rows),
        "group_rows": len(all_group_rows),
        "summary_rows": len(all_summary_rows),
    }


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_csv.parent / "overlap_metrics"
    result = calculate_file(
        input_csv=input_csv,
        output_dir=output_dir,
        topn=args.topn,
        weight_col=args.weight_col,
        group_weight=args.group_weight,
        min_token_strength=args.min_token_strength,
        min_neighbor_count=args.min_neighbor_count,
        max_shared_neighbors=args.max_shared_neighbors,
    )

    print(f"[ok] core metrics: {result['core_path']} ({result['core_rows']} rows)")
    print(f"[ok] group metrics: {result['group_path']} ({result['group_rows']} rows)")
    print(f"[ok] summary: {result['summary_path']} ({result['summary_rows']} rows)")


if __name__ == "__main__":
    main()
