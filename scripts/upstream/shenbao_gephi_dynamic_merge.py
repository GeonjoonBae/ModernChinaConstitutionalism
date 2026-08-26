#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import re
from typing import DefaultDict, Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
NETWORK_ROOT = ROOT / "shenbao" / "shenbao_network" / "network_applied"
DEFAULT_PERIODS_CSV = (
    ROOT / "shenbao" / "shenbao_network" / "applied_tokens" / "regex_only" / "periods" / "periods.csv"
)
DEFAULT_OUTPUT_ROOT = NETWORK_ROOT / "for_gephi" / "dynamic"
NETWORK_FILENAME_RE = re.compile(
    r"^network_(?P<dataset_label>.+?)_applied_(?P<profile>regex-only|strict|full)_"
    r"(?:(?P<center_mode>all-tokens|keyword-only)_)?"
    r"w(?P<window_size>\d+)_"
    r"(?:split-lr_none_raw-freq_)?"
    r"(?P<period_id>.+?)"
    r"(?P<context_filter>_filtered(?:_.+?)?)?_"
    r"joint(?P<joint_threshold>\d+)up_"
    r"stopv(?P<stopword_version>[^.]+?)(?:filtered)?\.csv$"
)
PROFILE_ORDER = ["regex-only", "strict", "full"]
NODE_COLOR_RULES = [
    ("制憲", "#ffd400"),
    ("憲法", "#ef0000"),
    ("立憲", "#00a651"),
    ("憲政", "#0057ff"),
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
        description=(
            "Merge Shenbao period-specific stopfiltered network CSV files into Gephi-ready dynamic node/edge CSV files."
        )
    )
    parser.add_argument("--period-set-id", default="long_period_manual")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["regex-only", "strict", "full"],
        help="Token profiles to process. Default: regex-only strict full.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Window sizes to process. Default: 1 5 10 20.",
    )
    parser.add_argument(
        "--direction",
        choices=["undirected", "directed"],
        default="undirected",
        help="Which network direction to export. Default: undirected.",
    )
    parser.add_argument(
        "--weight-column",
        choices=["count", "ppmi"],
        default="count",
        help="Weight column for Gephi edge weight. Default: count.",
    )
    parser.add_argument("--dataset-label", default="constitutional")
    parser.add_argument("--center-mode", default="all-tokens")
    parser.add_argument(
        "--context-filter",
        default="filtered_pre_zhixian_context",
        help=(
            "Folder/file context-filter suffix without leading underscore. "
            "Use 'none' to match unfiltered period folders."
        ),
    )
    parser.add_argument(
        "--periods-csv",
        default=str(DEFAULT_PERIODS_CSV),
        help=f"Periods CSV path. Default: {DEFAULT_PERIODS_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def color_for_node_id(node_id: str) -> str:
    for needle, color in NODE_COLOR_RULES:
        if needle in node_id:
            return color
    return "#ffffff"


def normalize_multi_values(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in values:
        for piece in str(raw).split(";"):
            cleaned = piece.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def read_csv_with_fallback(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    for encoding in ("utf-8", "utf-8-sig"):
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if encoding == "utf-8" and fieldnames and fieldnames[0].startswith("\ufeff"):
                continue
            return fieldnames, list(reader)
    raise ValueError(f"Unable to read CSV header from {path}")


def parse_float(value: str) -> float:
    return float(str(value).strip())


def parse_int(value: str) -> int:
    return int(float(str(value).strip()))


def read_period_rows(periods_csv: Path, period_set_id: str) -> List[Dict[str, str]]:
    _, rows = read_csv_with_fallback(periods_csv)
    filtered = [row for row in rows if row.get("period_set_id") == period_set_id]
    if not filtered:
        raise ValueError(f"No periods found for period_set_id={period_set_id}")
    filtered.sort(key=lambda row: int(row.get("sort_order", "0")))
    return filtered


def resolve_period_folder(period_id: str, context_filter: str) -> Path:
    if context_filter and context_filter.lower() != "none":
        exact = NETWORK_ROOT / f"stopfiltered_{period_id}_{context_filter}"
        if exact.exists():
            return exact
    else:
        exact = NETWORK_ROOT / f"stopfiltered_{period_id}"
        if exact.exists():
            return exact

    prefix = f"stopfiltered_{period_id}"
    matches = [
        path
        for path in NETWORK_ROOT.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    if context_filter and context_filter.lower() != "none":
        matches = [path for path in matches if context_filter in path.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No stopfiltered folder matched period_id={period_id}, context_filter={context_filter}")
    raise ValueError(
        f"Multiple stopfiltered folders matched period_id={period_id}: {', '.join(path.name for path in matches)}"
    )


def parse_network_filename(path: Path) -> Dict[str, str]:
    match = NETWORK_FILENAME_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unsupported network filename pattern: {path.name}")
    meta = match.groupdict()
    if not meta.get("center_mode"):
        meta["center_mode"] = "all-tokens"
    return meta


def candidate_search_dirs(period_folder: Path, direction: str) -> List[Path]:
    if direction == "undirected":
        nested = period_folder / "undirected"
        if nested.exists():
            return [nested]
    return [period_folder]


def find_network_file(
    period_folder: Path,
    period_id: str,
    profile: str,
    window: int,
    dataset_label: str,
    center_mode: str,
    direction: str,
) -> Tuple[Path, Dict[str, str]]:
    matches: List[Tuple[Path, Dict[str, str]]] = []
    for search_dir in candidate_search_dirs(period_folder, direction):
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.glob("*.csv")):
            if not path.name.startswith(f"network_{dataset_label}_applied_{profile}_"):
                continue
            try:
                meta = parse_network_filename(path)
            except ValueError:
                continue
            if meta["center_mode"] != center_mode:
                continue
            if int(meta["window_size"]) != int(window):
                continue
            if meta["period_id"] != period_id:
                continue
            matches.append((path, meta))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No network CSV matched period_id={period_id}, profile={profile}, w={window}, direction={direction}"
        )
    raise ValueError(
        f"Multiple network CSV files matched period_id={period_id}, profile={profile}, w={window}: "
        + ", ".join(path.name for path, _ in matches)
    )


def validate_direction(rows: Sequence[Dict[str, str]], direction: str, path: Path) -> None:
    values = {str(row.get("direction", "")).strip() for row in rows if row.get("direction", "") is not None}
    if direction == "undirected":
        if values == {"U"}:
            return
        if "R" in values or "L" in values:
            return
        raise ValueError(f"Unsupported direction values for undirected conversion in {path}: {sorted(values)}")
    else:
        if values == {"U"}:
            raise ValueError(f"Expected directed rows in {path}, but the file is undirected.")


def format_label(joint_count: str, ppmi: str, pos_pair_json: str) -> str:
    try:
        ppmi_text = f"{float(ppmi):.3f}"
    except (TypeError, ValueError):
        ppmi_text = str(ppmi)
    return f"{joint_count}, {ppmi_text}, {pos_pair_json}"


def edge_type_label(direction: str) -> str:
    return "Undirected" if direction == "undirected" else "Directed"


def edge_weight_value(row: Dict[str, str], weight_column: str) -> float:
    key = "joint_count" if weight_column == "count" else "ppmi"
    return float(row.get(key, "0") or 0)


def canonical_pair(token_a: str, token_b: str) -> Tuple[str, str]:
    if token_a <= token_b:
        return token_a, token_b
    return token_b, token_a


def rank_rows(rows: List[Dict[str, str]], metric_key: str, rank_key: str) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[metric_key]),
            str(row["center_token"]),
            str(row["neighbor_token"]),
        ),
    )
    for idx, row in enumerate(ordered, start=1):
        row[rank_key] = str(idx)


def convert_directed_rows_to_undirected_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    pair_joint: DefaultDict[Tuple[str, str, str, str], float] = defaultdict(float)
    pair_raw_joint: DefaultDict[Tuple[str, str, str, str], int] = defaultdict(int)
    pair_raw_distance_sum: DefaultDict[Tuple[str, str, str, str], float] = defaultdict(float)
    token_joint_marginal: DefaultDict[Tuple[str, str, str], float] = defaultdict(float)
    token_raw_marginal: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    event_total: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    raw_event_total: DefaultDict[Tuple[str, str], int] = defaultdict(int)

    for row in rows:
        if str(row.get("direction", "")).strip() != "R":
            continue
        period_set_id = str(row["period_set_id"]).strip()
        period_id = str(row["period_id"]).strip()
        center_token = str(row["center_token"]).strip()
        neighbor_token = str(row["neighbor_token"]).strip()
        if center_token == neighbor_token:
            continue

        joint_count = parse_float(row.get("joint_count", "0"))
        raw_joint_count = parse_int(row.get("raw_joint_event_count", "0"))
        avg_distance = parse_float(row.get("avg_distance", "0"))

        token_a, token_b = canonical_pair(center_token, neighbor_token)
        pair_key = (period_set_id, period_id, token_a, token_b)
        token_joint_marginal[(period_set_id, period_id, center_token)] += joint_count
        token_joint_marginal[(period_set_id, period_id, neighbor_token)] += joint_count
        token_raw_marginal[(period_set_id, period_id, center_token)] += raw_joint_count
        token_raw_marginal[(period_set_id, period_id, neighbor_token)] += raw_joint_count
        pair_joint[pair_key] += joint_count
        pair_raw_joint[pair_key] += raw_joint_count
        pair_raw_distance_sum[pair_key] += avg_distance * raw_joint_count
        event_total[(period_set_id, period_id)] += joint_count
        raw_event_total[(period_set_id, period_id)] += raw_joint_count

    output_rows: List[Dict[str, str]] = []
    for (period_set_id, period_id, token_a, token_b), joint_count in pair_joint.items():
        period_key = (period_set_id, period_id)
        center_marginal = token_joint_marginal[(period_set_id, period_id, token_a)]
        neighbor_marginal = token_joint_marginal[(period_set_id, period_id, token_b)]
        total_joint = event_total[period_key]
        total_raw = raw_event_total[period_key]
        raw_joint_count = pair_raw_joint[(period_set_id, period_id, token_a, token_b)]
        pmi = math.log2((joint_count * 2.0 * total_joint) / (center_marginal * neighbor_marginal))
        avg_distance = pair_raw_distance_sum[(period_set_id, period_id, token_a, token_b)] / float(raw_joint_count)
        output_rows.append(
            {
                "period_set_id": period_set_id,
                "period_id": period_id,
                "center_token": token_a,
                "neighbor_token": token_b,
                "direction": "U",
                "joint_count": str(joint_count),
                "raw_joint_event_count": str(raw_joint_count),
                "center_marginal_count": str(center_marginal),
                "neighbor_marginal_count": str(neighbor_marginal),
                "event_total": str(total_joint),
                "raw_event_total": str(total_raw),
                "center_raw_count": str(token_raw_marginal[(period_set_id, period_id, token_a)]),
                "neighbor_raw_count": str(token_raw_marginal[(period_set_id, period_id, token_b)]),
                "pmi": str(pmi),
                "ppmi": str(max(pmi, 0.0)),
                "distinct_context_count": "",
                "distinct_article_count": "",
                "avg_distance": str(avg_distance),
                "pos_pair_json": "",
                "rank_by_count": "0",
                "rank_by_pmi": "0",
            }
        )

    rank_rows(output_rows, "joint_count", "rank_by_count")
    rank_rows(output_rows, "pmi", "rank_by_pmi")
    output_rows.sort(
        key=lambda row: (
            row["period_set_id"],
            row["period_id"],
            int(row["rank_by_count"]),
            row["center_token"],
            row["neighbor_token"],
        )
    )
    return output_rows


def prepare_rows_for_direction(rows: Sequence[Dict[str, str]], direction: str) -> List[Dict[str, str]]:
    values = {str(row.get("direction", "")).strip() for row in rows if row.get("direction", "") is not None}
    if direction == "undirected":
        if values == {"U"}:
            return [dict(row) for row in rows if str(row.get("direction", "")).strip() == "U"]
        return convert_directed_rows_to_undirected_rows(rows)
    return [dict(row) for row in rows if str(row.get("direction", "")).strip() in {"L", "R"}]


def build_edge_row(
    row: Dict[str, str],
    period_row: Dict[str, str],
    weight_column: str,
    direction: str,
    profile: str,
    window: int,
    meta: Dict[str, str],
) -> Dict[str, object]:
    source = row["center_token"]
    target = row["neighbor_token"]
    period_id = period_row["period_id"]
    edge_id = f"{profile}|w{window}|{weight_column}|{period_id}|{source}|{target}"
    return {
        "Id": edge_id,
        "Source": source,
        "Target": target,
        "Type": edge_type_label(direction),
        "Weight": edge_weight_value(row, weight_column),
        "Start": period_row["start_date"],
        "End": period_row["end_date"],
        "period_set_id": period_row["period_set_id"],
        "period_id": period_id,
        "sort_order": int(period_row["sort_order"]),
        "profile": profile,
        "window_size": window,
        "weight_type": weight_column,
        "joint_threshold": int(meta["joint_threshold"]),
        "stopword_version": meta["stopword_version"],
        "context_filter": (meta.get("context_filter") or "").lstrip("_"),
        "joint_count": int(float(row.get("joint_count", "0") or 0)),
        "ppmi": float(row.get("ppmi", "0") or 0),
        "rank_by_count": int(float(row.get("rank_by_count", "0") or 0)) if row.get("rank_by_count") else "",
        "rank_by_pmi": int(float(row.get("rank_by_pmi", "0") or 0)) if row.get("rank_by_pmi") else "",
        "distinct_context_count": row.get("distinct_context_count", ""),
        "distinct_article_count": row.get("distinct_article_count", ""),
        "avg_distance": row.get("avg_distance", ""),
        "pos_pair_json": row.get("pos_pair_json", ""),
        "label": format_label(
            row.get("joint_count", ""),
            row.get("ppmi", ""),
            row.get("pos_pair_json", ""),
        ),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_component(value: str) -> str:
    forbidden = set('<>:"/\\|?*')
    return "".join("_" if ch in forbidden else ch for ch in value).strip().strip(".")


def build_output_stem(
    period_set_id: str,
    profile: str,
    center_mode: str,
    window: int,
    direction: str,
    weight_column: str,
    joint_threshold_label: str,
    stopword_version: str,
    context_filter: str,
) -> str:
    suffix = context_filter or "no_filter"
    return (
        f"for_gephi_dynamic_{sanitize_component(period_set_id)}_"
        f"{sanitize_component(profile)}_{sanitize_component(center_mode)}_"
        f"w{window}_{direction}_{weight_column}_"
        f"{sanitize_component(suffix)}_joint{sanitize_component(joint_threshold_label)}_stopv{sanitize_component(stopword_version)}"
    )


def sort_profile_key(profile: str) -> Tuple[int, str]:
    if profile in PROFILE_ORDER:
        return PROFILE_ORDER.index(profile), profile
    return len(PROFILE_ORDER), profile


def build_node_rows(
    edge_rows: Sequence[Dict[str, object]],
    period_rows_by_id: Dict[str, Dict[str, str]],
    profile: str,
    window: int,
    direction: str,
    weight_column: str,
) -> List[Dict[str, object]]:
    node_map: Dict[str, Dict[str, object]] = {}
    for edge in edge_rows:
        source = str(edge["Source"])
        target = str(edge["Target"])
        weight = float(edge["Weight"])
        period_id = str(edge["period_id"])
        sort_order = int(edge["sort_order"])
        for token, neighbor in ((source, target), (target, source)):
            if token not in node_map:
                node_map[token] = {
                    "Id": token,
                    "Label": token,
                    "profile": profile,
                    "window_size": window,
                    "direction": direction,
                    "weight_type": weight_column,
                    "_period_ids": set(),
                    "_period_info": {},
                    "_neighbors": set(),
                    "_edge_count": 0,
                    "_weighted_degree": 0.0,
                }
            bucket = node_map[token]
            bucket["_period_ids"].add(period_id)
            bucket["_period_info"][period_id] = sort_order
            bucket["_neighbors"].add(neighbor)
            bucket["_edge_count"] += 1
            bucket["_weighted_degree"] += weight

    output_rows: List[Dict[str, object]] = []
    for token, bucket in node_map.items():
        active_period_ids = sorted(bucket["_period_ids"], key=lambda pid: bucket["_period_info"][pid])
        first_period_id = active_period_ids[0]
        last_period_id = active_period_ids[-1]
        first_period = period_rows_by_id[first_period_id]
        last_period = period_rows_by_id[last_period_id]
        output_rows.append(
            {
                "Id": bucket["Id"],
                "Label": bucket["Label"],
                "color": color_for_node_id(str(bucket["Id"])),
                "profile": bucket["profile"],
                "window_size": bucket["window_size"],
                "direction": bucket["direction"],
                "weight_type": bucket["weight_type"],
                "Start": first_period["start_date"],
                "End": last_period["end_date"],
                "first_period_id": first_period_id,
                "last_period_id": last_period_id,
                "active_period_count": len(active_period_ids),
                "active_period_ids": ";".join(active_period_ids),
                "neighbor_count_total": len(bucket["_neighbors"]),
                "edge_instance_count_total": bucket["_edge_count"],
                "weighted_degree_total": round(float(bucket["_weighted_degree"]), 6),
            }
        )
    output_rows.sort(
        key=lambda row: (
            int(period_rows_by_id[row["first_period_id"]]["sort_order"]),
            row["Label"],
        )
    )
    return output_rows


def collect_edge_rows(
    period_rows: Sequence[Dict[str, str]],
    profile: str,
    window: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    edge_rows: List[Dict[str, object]] = []
    reference_meta: Dict[str, str] | None = None
    joint_thresholds: List[str] = []
    for period_row in period_rows:
        period_id = period_row["period_id"]
        period_folder = resolve_period_folder(period_id, args.context_filter)
        input_path, meta = find_network_file(
            period_folder=period_folder,
            period_id=period_id,
            profile=profile,
            window=window,
            dataset_label=args.dataset_label,
            center_mode=args.center_mode,
            direction=args.direction,
        )
        _, rows = read_csv_with_fallback(input_path)
        validate_direction(rows, args.direction, input_path)
        rows_for_export = prepare_rows_for_direction(rows, args.direction)
        if reference_meta is None:
            reference_meta = meta
        else:
            for key in ("stopword_version", "context_filter", "center_mode", "dataset_label"):
                if reference_meta.get(key) != meta.get(key):
                    raise ValueError(
                        f"Inconsistent metadata across periods for profile={profile}, w={window}: "
                        f"{key} differs ({reference_meta.get(key)} vs {meta.get(key)})"
                    )
        joint_thresholds.append(meta["joint_threshold"])
        for row in rows_for_export:
            edge_rows.append(
                build_edge_row(
                    row=row,
                    period_row=period_row,
                    weight_column=args.weight_column,
                    direction=args.direction,
                    profile=profile,
                    window=window,
                    meta=meta,
                )
            )
    if reference_meta is None:
        raise ValueError(f"No input files found for profile={profile}, w={window}")
    unique_thresholds = sorted({int(value) for value in joint_thresholds})
    if len(unique_thresholds) == 1:
        reference_meta["joint_threshold_label"] = f"{unique_thresholds[0]}up"
    else:
        reference_meta["joint_threshold_label"] = "var-" + "-".join(str(value) for value in unique_thresholds) + "up"
    reference_meta["joint_thresholds"] = ",".join(str(value) for value in unique_thresholds)
    edge_rows.sort(
        key=lambda row: (
            int(row["sort_order"]),
            str(row["Source"]),
            str(row["Target"]),
        )
    )
    return edge_rows, reference_meta


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    args.profiles = normalize_multi_values(args.profiles)
    period_rows = read_period_rows(Path(args.periods_csv), args.period_set_id)
    period_rows_by_id = {row["period_id"]: row for row in period_rows}
    output_root = Path(args.output_dir) / sanitize_component(args.period_set_id)
    manifest_rows: List[Dict[str, object]] = []

    for profile in sorted(args.profiles, key=sort_profile_key):
        for window in sorted(set(args.windows)):
            edge_rows, meta = collect_edge_rows(period_rows, profile, window, args)
            node_rows = build_node_rows(
                edge_rows=edge_rows,
                period_rows_by_id=period_rows_by_id,
                profile=profile,
                window=window,
                direction=args.direction,
                weight_column=args.weight_column,
            )
            output_stem = build_output_stem(
                period_set_id=args.period_set_id,
                profile=profile,
                center_mode=args.center_mode,
                window=window,
                direction=args.direction,
                weight_column=args.weight_column,
                joint_threshold_label=meta["joint_threshold_label"],
                stopword_version=meta["stopword_version"],
                context_filter=(meta.get("context_filter") or "").lstrip("_"),
            )
            edge_path = output_root / f"{output_stem}_edges.csv"
            node_path = output_root / f"{output_stem}_nodes.csv"
            manifest_path = output_root / f"{output_stem}_manifest.json"

            edge_fieldnames = [
                "Id",
                "Source",
                "Target",
                "Type",
                "Weight",
                "Start",
                "End",
                "period_set_id",
                "period_id",
                "sort_order",
                "profile",
                "window_size",
                "weight_type",
                "joint_threshold",
                "stopword_version",
                "context_filter",
                "joint_count",
                "ppmi",
                "rank_by_count",
                "rank_by_pmi",
                "distinct_context_count",
                "distinct_article_count",
                "avg_distance",
                "pos_pair_json",
                "label",
            ]
            node_fieldnames = [
                "Id",
                "Label",
                "color",
                "profile",
                "window_size",
                "direction",
                "weight_type",
                "Start",
                "End",
                "first_period_id",
                "last_period_id",
                "active_period_count",
                "active_period_ids",
                "neighbor_count_total",
                "edge_instance_count_total",
                "weighted_degree_total",
            ]

            write_csv(edge_path, edge_fieldnames, edge_rows)
            write_csv(node_path, node_fieldnames, node_rows)
            write_manifest(
                manifest_path,
                {
                    "period_set_id": args.period_set_id,
                    "profile": profile,
                    "window_size": window,
                    "direction": args.direction,
                    "weight_column": args.weight_column,
                    "dataset_label": args.dataset_label,
                    "center_mode": args.center_mode,
                    "context_filter": (meta.get("context_filter") or "").lstrip("_"),
                    "joint_thresholds": meta["joint_thresholds"],
                    "stopword_version": meta["stopword_version"],
                    "periods": [
                        {
                            "period_id": row["period_id"],
                            "sort_order": int(row["sort_order"]),
                            "start_date": row["start_date"],
                            "end_date": row["end_date"],
                        }
                        for row in period_rows
                    ],
                    "edge_csv": str(edge_path),
                    "node_csv": str(node_path),
                    "edge_row_count": len(edge_rows),
                    "node_row_count": len(node_rows),
                },
            )
            log(
                f"[ok] profile={profile}, w={window} -> "
                f"edges={edge_path.name} ({len(edge_rows)} rows), "
                f"nodes={node_path.name} ({len(node_rows)} rows)"
            )
            manifest_rows.append(
                {
                    "profile": profile,
                    "window_size": window,
                    "edge_csv": str(edge_path),
                    "node_csv": str(node_path),
                    "manifest_json": str(manifest_path),
                    "edge_row_count": len(edge_rows),
                    "node_row_count": len(node_rows),
                }
            )

    summary_manifest_path = output_root / f"{sanitize_component(args.period_set_id)}__batch_manifest.json"
    write_manifest(
        summary_manifest_path,
        {
            "period_set_id": args.period_set_id,
            "direction": args.direction,
            "weight_column": args.weight_column,
            "profiles": args.profiles,
            "windows": sorted(set(args.windows)),
            "items": manifest_rows,
        },
    )
    log(f"[ok] batch manifest -> {summary_manifest_path}")


if __name__ == "__main__":
    main()
