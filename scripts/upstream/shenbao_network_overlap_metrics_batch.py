#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch CLI for Shenbao network overlap metrics.

Default target:
    shenbao/shenbao_network/network_applied/
        stopfiltered_*_filtered_pre_zhixian_context/undirected/*.csv

Default output:
    shenbao/shenbao_network/network_overlap_metrics/<condition_dir>/
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shenbao_network_overlap_metrics import (
    CORE_FIELDNAMES,
    GROUP_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    calculate_file,
    calculate_metrics,
    output_paths,
    raise_csv_field_limit,
    read_csv_dicts,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ROOT / "shenbao" / "shenbao_network" / "network_applied"
DEFAULT_OUTPUT_ROOT = ROOT / "shenbao" / "shenbao_network" / "network_overlap_metrics"
DEFAULT_INCLUDE_GLOB = "stopfiltered_*_filtered_pre_zhixian_context/undirected/*.csv"
DEFAULT_PROFILES = ["strict", "full", "regex-only"]
DEFAULT_WINDOWS = ["1", "5", "10", "20"]

MANIFEST_FIELDNAMES = [
    "status",
    "input_csv",
    "condition_dir",
    "period_scope",
    "token_profile",
    "network_window",
    "topn",
    "output_dir",
    "core_rows",
    "group_rows",
    "summary_rows",
    "message",
]


def move_field_to_end(fieldnames: Sequence[str], field: str) -> List[str]:
    return [name for name in fieldnames if name != field] + [field]


COMBINED_CORE_FIELDNAMES = move_field_to_end(CORE_FIELDNAMES, "source_file")
COMBINED_GROUP_FIELDNAMES = move_field_to_end(GROUP_FIELDNAMES, "source_file")
COMBINED_SUMMARY_FIELDNAMES = move_field_to_end(SUMMARY_FIELDNAMES, "source_file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run network overlap metrics for all selected Shenbao network edge CSV conditions."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--include-glob",
        default=DEFAULT_INCLUDE_GLOB,
        help="Glob relative to input-root. Default targets filtered_pre_zhixian_context undirected CSVs.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated profiles. Use all for no profile filtering.",
    )
    parser.add_argument(
        "--windows",
        default=",".join(DEFAULT_WINDOWS),
        help="Comma-separated window sizes. Use all for no window filtering.",
    )
    parser.add_argument(
        "--periods",
        default="all",
        help=(
            "Comma-separated period scopes from condition dir, e.g. "
            "global,long_period_manual_p000. Use all for no period filtering."
        ),
    )
    parser.add_argument("--topn", type=int, help="Single top-N setting. Use 0 for all neighbors.")
    parser.add_argument(
        "--topns",
        help=(
            "Comma-separated top-N settings. Use 0 for all neighbors. "
            "Default: 20,50,100,0. Ignored when --topn is supplied."
        ),
    )
    parser.add_argument("--weight-col", default="joint_count")
    parser.add_argument("--group-weight", choices=["log", "raw", "equal"], default="log")
    parser.add_argument("--min-token-strength", type=float, default=30.0)
    parser.add_argument("--min-neighbor-count", type=int, default=10)
    parser.add_argument("--max-shared-neighbors", type=int, default=20)
    parser.add_argument(
        "--output-mode",
        choices=["combined", "individual", "both"],
        default="combined",
        help=(
            "combined writes one core, one group, and one summary CSV under output-root. "
            "individual writes per-input CSV outputs. both writes both forms."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Recalculate files even if all outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files without calculating.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected files.")
    return parser.parse_args()


def split_filter(value: str) -> Optional[set]:
    normalized = value.strip()
    if not normalized or normalized.lower() == "all":
        return None
    return {part.strip() for part in normalized.split(",") if part.strip()}


def parse_topns(topn: Optional[int], topns: Optional[str]) -> List[int]:
    if topn is not None:
        values = [topn]
    elif topns:
        values = [int(part.strip()) for part in topns.split(",") if part.strip()]
    else:
        values = [20, 50, 100, 0]
    if not values:
        raise ValueError("At least one topn value is required")
    seen = set()
    out: List[int] = []
    for value in values:
        if value < 0:
            raise ValueError("topn values must be >= 0")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_file_condition(path: Path) -> Dict[str, str]:
    name = path.name
    profile_match = re.search(r"_applied_(strict|full|regex-only)_", name)
    window_match = re.search(r"_w(\d+)_", name)
    condition_dir = path.parent.parent.name if path.parent.name == "undirected" else path.parent.name
    period_scope = condition_dir
    if period_scope.startswith("stopfiltered_"):
        period_scope = period_scope[len("stopfiltered_") :]
    suffix = "_filtered_pre_zhixian_context"
    if period_scope.endswith(suffix):
        period_scope = period_scope[: -len(suffix)]
    return {
        "condition_dir": condition_dir,
        "period_scope": period_scope,
        "token_profile": profile_match.group(1) if profile_match else "",
        "network_window": window_match.group(1) if window_match else "",
    }


def selected_files(
    input_root: Path,
    include_glob: str,
    profiles: Optional[set],
    windows: Optional[set],
    periods: Optional[set],
) -> List[Tuple[Path, Dict[str, str]]]:
    files = sorted(path for path in input_root.glob(include_glob) if path.is_file())
    selected: List[Tuple[Path, Dict[str, str]]] = []
    for path in files:
        condition = parse_file_condition(path)
        if profiles is not None and condition["token_profile"] not in profiles:
            continue
        if windows is not None and condition["network_window"] not in windows:
            continue
        if periods is not None and condition["period_scope"] not in periods:
            continue
        selected.append((path, condition))
    return selected


def output_dir_for_file(output_root: Path, condition: Dict[str, str]) -> Path:
    return output_root / condition["condition_dir"]


def outputs_exist(input_csv: Path, output_dir: Path) -> bool:
    return all(path.exists() for path in output_paths(input_csv, output_dir))


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in MANIFEST_FIELDNAMES})


def write_combined_summary(path: Path, summary_paths: Iterable[Path]) -> int:
    rows: List[Dict[str, str]] = []
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        summary_rows, _ = read_csv_dicts(summary_path)
        rows.extend(summary_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in COMBINED_SUMMARY_FIELDNAMES})
    return len(rows)


def combined_output_paths(output_root: Path) -> Tuple[Path, Path, Path]:
    return (
        output_root / "network_core_overlap_metrics_combined.csv",
        output_root / "network_group_overlap_metrics_combined.csv",
        output_root / "network_overlap_summary_combined.csv",
    )


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    topns = parse_topns(args.topn, args.topns)
    if len(topns) > 1 and args.output_mode != "combined":
        raise ValueError("Multiple topn values are supported only with --output-mode combined")

    files = selected_files(
        input_root=input_root,
        include_glob=args.include_glob,
        profiles=split_filter(args.profiles),
        windows=split_filter(args.windows),
        periods=split_filter(args.periods),
    )
    if args.limit is not None:
        files = files[: args.limit]

    print(f"[select] files={len(files)} topns={','.join(str(value) for value in topns)}")
    if args.dry_run:
        for input_csv, condition in files:
            print(
                "[dry-run] "
                f"{condition['period_scope']} profile={condition['token_profile']} "
                f"w{condition['network_window']} -> {input_csv}"
            )
        return

    manifest_rows: List[Dict[str, object]] = []
    combined_core_rows: List[Dict[str, object]] = []
    combined_group_rows: List[Dict[str, object]] = []
    combined_summary_rows: List[Dict[str, object]] = []
    summary_paths: List[Path] = []
    for index, (input_csv, condition) in enumerate(files, start=1):
        output_dir = output_dir_for_file(output_root, condition)
        core_path, group_path, summary_path = output_paths(input_csv, output_dir)
        row: Dict[str, object] = {
            "input_csv": input_csv,
            "condition_dir": condition["condition_dir"],
            "period_scope": condition["period_scope"],
            "token_profile": condition["token_profile"],
            "network_window": condition["network_window"],
            "topn": ",".join(str(value) for value in topns),
            "output_dir": output_dir,
        }
        if args.output_mode == "individual" and not args.overwrite and outputs_exist(input_csv, output_dir):
            print(f"[skip {index}/{len(files)}] exists: {input_csv.name}")
            row.update(
                {
                    "status": "skipped_exists",
                    "core_rows": "",
                    "group_rows": "",
                    "summary_rows": "",
                    "message": "all outputs already exist",
                }
            )
            manifest_rows.append(row)
            summary_paths.append(summary_path)
            continue

        try:
            print(f"[run {index}/{len(files)}] {condition['period_scope']} {input_csv.name}")
            if args.output_mode == "individual":
                result = calculate_file(
                    input_csv=input_csv,
                    output_dir=output_dir,
                    topn=topns[0],
                    weight_col=args.weight_col,
                    group_weight=args.group_weight,
                    min_token_strength=args.min_token_strength,
                    min_neighbor_count=args.min_neighbor_count,
                    max_shared_neighbors=args.max_shared_neighbors,
                )
                core_row_count = result["core_rows"]
                group_row_count = result["group_rows"]
                summary_row_count = result["summary_rows"]
                summary_paths.append(Path(result["summary_path"]))
            else:
                core_row_count = 0
                group_row_count = 0
                summary_row_count = 0
                for topn_value in topns:
                    metric_rows = calculate_metrics(
                        input_csv=input_csv,
                        topn=topn_value,
                        weight_col=args.weight_col,
                        group_weight=args.group_weight,
                        min_token_strength=args.min_token_strength,
                        min_neighbor_count=args.min_neighbor_count,
                        max_shared_neighbors=args.max_shared_neighbors,
                    )
                    combined_core_rows.extend(metric_rows["core_rows"])
                    combined_group_rows.extend(metric_rows["group_rows"])
                    combined_summary_rows.extend(metric_rows["summary_rows"])
                    core_row_count += len(metric_rows["core_rows"])
                    group_row_count += len(metric_rows["group_rows"])
                    summary_row_count += len(metric_rows["summary_rows"])
            row.update(
                {
                    "status": "ok",
                    "core_rows": core_row_count,
                    "group_rows": group_row_count,
                    "summary_rows": summary_row_count,
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - batch manifest should retain per-file failures.
            row.update(
                {
                    "status": "error",
                    "core_rows": "",
                    "group_rows": "",
                    "summary_rows": "",
                    "message": repr(exc),
                }
            )
            print(f"[error] {input_csv}: {exc}", file=sys.stderr)
        manifest_rows.append(row)

    manifest_path = output_root / "network_overlap_metrics_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    core_combined_path, group_combined_path, summary_combined_path = combined_output_paths(output_root)
    if args.output_mode in {"combined", "both"}:
        write_csv(core_combined_path, COMBINED_CORE_FIELDNAMES, combined_core_rows)
        write_csv(group_combined_path, COMBINED_GROUP_FIELDNAMES, combined_group_rows)
        write_csv(summary_combined_path, COMBINED_SUMMARY_FIELDNAMES, combined_summary_rows)
        combined_rows = len(combined_summary_rows)
    else:
        combined_rows = write_combined_summary(summary_combined_path, summary_paths)
    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    skipped_count = sum(1 for row in manifest_rows if row["status"] == "skipped_exists")
    error_count = sum(1 for row in manifest_rows if row["status"] == "error")
    print(f"[done] ok={ok_count} skipped={skipped_count} errors={error_count}")
    print(f"[done] manifest: {manifest_path}")
    if args.output_mode in {"combined", "both"}:
        print(f"[done] combined core: {core_combined_path} ({len(combined_core_rows)} rows)")
        print(f"[done] combined group: {group_combined_path} ({len(combined_group_rows)} rows)")
    print(f"[done] combined summary: {summary_combined_path} ({combined_rows} rows)")


if __name__ == "__main__":
    main()
