# -*- coding: utf-8 -*-

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


OUTPUT_FIELDNAMES = [
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
            "Build undirected, self-loop-removed PMI/PPMI CSV files from Shenbao "
            "directed network CSV files using the original count base."
        )
    )
    parser.add_argument(
        "--input-csv",
        nargs="+",
        required=True,
        help="Input CSV file(s) or folder(s). Multiple values and semicolon-separated values are supported.",
    )
    return parser.parse_args()


def normalize_input_paths(raw_values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in raw_values:
        for chunk in raw.split(";"):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            paths.append(Path(cleaned))
    if not paths:
        raise ValueError("No input paths provided.")
    return paths


def expand_input_files(input_paths: Sequence[Path]) -> List[Path]:
    expanded: List[Path] = []
    for path in input_paths:
        if path.is_dir():
            expanded.extend(sorted(child for child in path.glob("*.csv") if child.is_file()))
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    if not expanded:
        raise FileNotFoundError("No CSV files found in the provided input paths.")
    return expanded


def canonical_pair(token_a: str, token_b: str) -> Tuple[str, str]:
    if token_a <= token_b:
        return token_a, token_b
    return token_b, token_a


def parse_float(value: str) -> float:
    return float(value.strip())


def parse_int(value: str) -> int:
    return int(float(value.strip()))


def format_number(value: float, force_int: bool = False) -> str:
    if force_int or float(value).is_integer():
        return str(int(round(value)))
    return repr(value)


def build_output_path(input_path: Path) -> Path:
    output_dir = input_path.parent / "undirected"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / input_path.name


def rank_rows(rows: List[Dict[str, object]], metric_key: str, rank_key: str) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[metric_key]),
            str(row["center_token"]),
            str(row["neighbor_token"]),
        ),
    )
    for idx, row in enumerate(ordered, start=1):
        row[rank_key] = idx


def convert_file(input_path: Path) -> Tuple[int, int]:
    pair_joint: DefaultDict[Tuple[str, str, str, str], float] = defaultdict(float)
    pair_raw_joint: DefaultDict[Tuple[str, str, str, str], int] = defaultdict(int)
    pair_raw_distance_sum: DefaultDict[Tuple[str, str, str, str], float] = defaultdict(float)

    token_joint_marginal: DefaultDict[Tuple[str, str, str], float] = defaultdict(float)
    token_raw_marginal: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    event_total: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    raw_event_total: DefaultDict[Tuple[str, str], int] = defaultdict(int)

    input_rows = 0
    kept_r_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            input_rows += 1
            if row["direction"].strip() != "R":
                continue

            period_set_id = row["period_set_id"].strip()
            period_id = row["period_id"].strip()
            center_token = row["center_token"].strip()
            neighbor_token = row["neighbor_token"].strip()

            if center_token == neighbor_token:
                continue

            joint_count = parse_float(row["joint_count"])
            raw_joint_count = parse_int(row["raw_joint_event_count"])
            avg_distance = parse_float(row["avg_distance"])

            pair_token_a, pair_token_b = canonical_pair(center_token, neighbor_token)
            pair_key = (period_set_id, period_id, pair_token_a, pair_token_b)
            period_key = (period_set_id, period_id)

            pair_joint[pair_key] += joint_count
            pair_raw_joint[pair_key] += raw_joint_count
            pair_raw_distance_sum[pair_key] += avg_distance * raw_joint_count

            token_joint_marginal[(period_set_id, period_id, center_token)] += joint_count
            token_joint_marginal[(period_set_id, period_id, neighbor_token)] += joint_count
            token_raw_marginal[(period_set_id, period_id, center_token)] += raw_joint_count
            token_raw_marginal[(period_set_id, period_id, neighbor_token)] += raw_joint_count

            event_total[period_key] += joint_count
            raw_event_total[period_key] += raw_joint_count
            kept_r_rows += 1

    output_rows: List[Dict[str, object]] = []
    for (period_set_id, period_id, token_a, token_b), joint_count in pair_joint.items():
        period_key = (period_set_id, period_id)
        center_marginal = token_joint_marginal[(period_set_id, period_id, token_a)]
        neighbor_marginal = token_joint_marginal[(period_set_id, period_id, token_b)]
        center_raw = token_raw_marginal[(period_set_id, period_id, token_a)]
        neighbor_raw = token_raw_marginal[(period_set_id, period_id, token_b)]
        total_joint = event_total[period_key]
        total_raw = raw_event_total[period_key]
        raw_joint_count = pair_raw_joint[(period_set_id, period_id, token_a, token_b)]

        pmi = math.log2((joint_count * 2.0 * total_joint) / (center_marginal * neighbor_marginal))
        ppmi = max(pmi, 0.0)
        avg_distance = pair_raw_distance_sum[(period_set_id, period_id, token_a, token_b)] / float(raw_joint_count)

        output_rows.append(
            {
                "period_set_id": period_set_id,
                "period_id": period_id,
                "center_token": token_a,
                "neighbor_token": token_b,
                "direction": "U",
                "joint_count": joint_count,
                "raw_joint_event_count": raw_joint_count,
                "center_marginal_count": center_marginal,
                "neighbor_marginal_count": neighbor_marginal,
                "event_total": total_joint,
                "raw_event_total": total_raw,
                "center_raw_count": center_raw,
                "neighbor_raw_count": neighbor_raw,
                "pmi": pmi,
                "ppmi": ppmi,
                "distinct_context_count": "",
                "distinct_article_count": "",
                "avg_distance": avg_distance,
                "pos_pair_json": "",
                "rank_by_count": 0,
                "rank_by_pmi": 0,
            }
        )

    rank_rows(output_rows, "joint_count", "rank_by_count")
    rank_rows(output_rows, "pmi", "rank_by_pmi")

    output_rows.sort(
        key=lambda row: (
            str(row["period_set_id"]),
            str(row["period_id"]),
            int(row["rank_by_count"]),
            str(row["center_token"]),
            str(row["neighbor_token"]),
        )
    )

    output_path = build_output_path(input_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    key: (
                        format_number(float(value))
                        if key
                        in {
                            "joint_count",
                            "center_marginal_count",
                            "neighbor_marginal_count",
                            "event_total",
                            "pmi",
                            "ppmi",
                            "avg_distance",
                        }
                        else format_number(float(value), force_int=True)
                        if key
                        in {
                            "raw_joint_event_count",
                            "raw_event_total",
                            "center_raw_count",
                            "neighbor_raw_count",
                            "rank_by_count",
                            "rank_by_pmi",
                        }
                        else value
                    )
                    for key, value in row.items()
                }
            )

    return kept_r_rows, len(output_rows)


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    input_paths = normalize_input_paths(args.input_csv)
    input_files = expand_input_files(input_paths)

    for input_path in input_files:
        kept_r_rows, output_rows = convert_file(input_path)
        output_path = build_output_path(input_path)
        print(
            f"[ok] {input_path.name} -> {output_path} "
            f"(R_rows_without_loops={kept_r_rows}, undirected_rows={output_rows})"
        )


if __name__ == "__main__":
    main()
