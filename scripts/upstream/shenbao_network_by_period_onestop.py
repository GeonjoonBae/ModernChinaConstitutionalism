#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from shenbao_context_filter_utils import (
    DEFAULT_CONTEXT_FILTER_NAME,
    DEFAULT_FILTER_ROOT,
    context_filter_stem_part,
    load_context_filter,
)


ROOT = Path(__file__).resolve().parent
SHENBAO_DIR = ROOT / "shenbao"
NETWORK_APPLIED_DIR = SHENBAO_DIR / "shenbao_network" / "network_applied"
DEFAULT_APPLIED_TOKENS_DIR = SHENBAO_DIR / "shenbao_network" / "applied_tokens"
DONE_DIR = NETWORK_APPLIED_DIR / "done"
PART_FILTERED_DIR = NETWORK_APPLIED_DIR / "part_already_joint_filtered"
JOINTUP_DIR = NETWORK_APPLIED_DIR / "jointup"
STOPFILTER_SUMMARY_DIR = NETWORK_APPLIED_DIR / "stopfilter_summary"
RUNTIME_RULE_DIR = SHENBAO_DIR / "shenbao_dictionary" / "runtime_rule_splits"
DEFAULT_PERIODS_PARQUET = (
    SHENBAO_DIR / "shenbao_network" / "applied_tokens" / "full" / "periods" / "periods.parquet"
)

NETWORK_APPLIED_SCRIPT = ROOT / "shenbao_network_applied.py"
STOPWORD_SCRIPT = ROOT / "shenbao_stopword_postfilter_v3.py"
NODES_SCRIPT = ROOT / "shenbao_pmi_nodes_extract.py"

DEFAULT_WINDOW_SIZES = (1, 5, 10, 20)
DEFAULT_CENTER_MODE = "all-tokens"
DEFAULT_BOUNDARY_POS = "none"
DEFAULT_STOPWORD_PROFILE = "always"
DEFAULT_RETENTION_W1 = 0.50
DEFAULT_RETENTION_OTHER = 0.30
DEFAULT_MIN_JOINT_FLOOR = 3
DEFAULT_PROFILES = ("regex-only", "strict", "full")
PERIOD_ID_RE = re.compile(r"^(?P<period_set_id>.+)_p\d{3}$")
STOPFILTER_FILENAME_RE = re.compile(
    r"^(?:pmi|network)_"
    r"(?P<dataset_label>.+?)_"
    r"(?:raw-layer_)?"
    r"(?P<center_mode>all-tokens|keyword-only)_"
    r"w(?P<window_size>\d+)_split-lr_none_raw-freq_"
    r"(?P<period_scope>.+?)"
    r"(?P<context_filter>_filtered_.+?)?"
    r"(?:_joint(?P<joint_threshold>\d+)up)?"
    r"(?:\.part(?P<part>\d+))?"
    r"_stopv(?P<stopword_version>[^.]+)filtered\.csv$"
)


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


def sanitize_slug(value: str) -> str:
    out: List[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("_")
    return text or "all"


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_window_sizes(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        value = int(cleaned)
        if value < 1:
            raise ValueError("window sizes must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("No valid window sizes provided.")
    return values


def parse_profiles(raw_values: Sequence[str]) -> List[str]:
    allowed = {"regex-only", "strict", "full"}
    normalized: List[str] = []
    for raw in raw_values:
        for piece in re.split(r"[;,]", raw):
            cleaned = piece.strip().lower()
            if not cleaned:
                continue
            if cleaned not in allowed:
                raise ValueError(f"Unsupported profile: {cleaned}")
            if cleaned not in normalized:
                normalized.append(cleaned)
    if not normalized:
        raise ValueError("No valid profiles provided.")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-stop pipeline for period-specific applied-token networks: "
            "network_applied -> joint-count upper extract -> stopword postfilter -> nodes extract."
        )
    )
    parser.add_argument("--dataset-label")
    parser.add_argument("--keywords")
    parser.add_argument(
        "--applied-tokens-dir",
        default=str(DEFAULT_APPLIED_TOKENS_DIR),
        help="Root directory containing applied token profile folders.",
    )
    parser.add_argument(
        "--applied-token-dir-prefix",
        default="",
        help=(
            "Optional prefix for applied token profile folders. For example, "
            "classified_ad_ maps full to classified_ad_full and regex-only to "
            "classified_ad_regex_only."
        ),
    )
    parser.add_argument(
        "--periods-parquet",
        type=Path,
        default=None,
        help=(
            "Optional explicit periods.parquet forwarded to shenbao_network_applied.py. "
            "If omitted, use each applied token profile folder's periods/periods.parquet."
        ),
    )
    parser.add_argument(
        "--output-scope-suffix",
        default="",
        help=(
            "Optional suffix appended to period output folders, e.g. classified_ad "
            "creates stopfiltered_long_period_manual_p001_classified_ad."
        ),
    )
    parser.add_argument(
        "--period-target",
        nargs="+",
        help=(
            "One or more period targets. Each target may be a specific period_id or a "
            "period_set_id. Multiple targets may be given as separate arguments or "
            "semicolon/comma-separated values. Use global or global_all for global scope."
        ),
    )
    parser.add_argument("--period-set-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--period-id",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--center-mode",
        choices=["all-tokens", "keyword-only"],
        default=DEFAULT_CENTER_MODE,
    )
    parser.add_argument(
        "--boundary-pos",
        default=DEFAULT_BOUNDARY_POS,
        help="Boundary POS list for shenbao_network_applied.py. Default: none.",
    )
    parser.add_argument(
        "--window-sizes",
        default="1,5,10,20",
        help="Comma-separated window sizes. Default: 1,5,10,20",
    )
    parser.add_argument(
        "--stopword-profile",
        choices=["always", "full"],
        default=DEFAULT_STOPWORD_PROFILE,
        help="Stopword profile passed to shenbao_stopword_postfilter_v3.py. Default: always.",
    )
    parser.add_argument(
        "--retention-ratio-w1",
        type=float,
        default=DEFAULT_RETENTION_W1,
        help="Target retained edge ratio for w1 after joint filtering. Default: 0.50.",
    )
    parser.add_argument(
        "--retention-ratio-other",
        type=float,
        default=DEFAULT_RETENTION_OTHER,
        help="Target retained edge ratio for windows other than 1. Default: 0.30.",
    )
    parser.add_argument(
        "--min-joint-floor",
        type=int,
        default=DEFAULT_MIN_JOINT_FLOOR,
        help="Lower bound for adaptive raw_joint_event_count threshold. Default: 3.",
    )
    parser.add_argument(
        "--network-write-pause-seconds",
        type=float,
        default=0.1,
        help="Forwarded to shenbao_network_applied.py. Default: 0.1.",
    )
    parser.add_argument(
        "--network-rows-per-output-file",
        type=int,
        default=500000,
        help="Forwarded to shenbao_network_applied.py. Default: 500000.",
    )
    parser.add_argument(
        "--network-write-chunk-size",
        type=int,
        default=50000,
        help="Forwarded to shenbao_network_applied.py. Default: 50000.",
    )
    parser.add_argument(
        "--nodes-output-mode",
        choices=["minimal", "extended", "both"],
        default="both",
        help="Forwarded to shenbao_pmi_nodes_extract.py. Default: both.",
    )
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
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        help="Applied token profiles to run. Default: regex-only strict full.",
    )
    return parser.parse_args()


def prompt_nonempty(label: str, current: Optional[str]) -> str:
    if current and current.strip():
        return current.strip()
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("Input cannot be empty.", flush=True)


def normalize_optional_period_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "ALL":
        return None
    return cleaned


def normalize_period_target_values(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    normalized: List[str] = []
    for value in values:
        for chunk in re.split(r"[;,]", value):
            cleaned = chunk.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def resolve_one_period_target(
    target: str,
    legacy_period_set_id: Optional[str],
    legacy_period_id: Optional[str],
) -> Tuple[str, Optional[str]]:
    legacy_set = (legacy_period_set_id or "").strip()
    legacy_id = normalize_optional_period_id(legacy_period_id)

    if target in {"global", "global_all"}:
        return "global_all", None

    match = PERIOD_ID_RE.fullmatch(target)
    if match:
        derived_set = match.group("period_set_id")
        if legacy_set and legacy_set != derived_set:
            raise ValueError(
                f"period target implies set '{derived_set}', but --period-set-id was '{legacy_set}'."
            )
        return derived_set, target

    if legacy_id:
        derived = PERIOD_ID_RE.fullmatch(legacy_id)
        if derived and derived.group("period_set_id") != target:
            raise ValueError(
                f"--period-id belongs to set '{derived.group('period_set_id')}', not '{target}'."
            )
    return target, None


def resolve_period_targets(
    period_targets: Optional[Sequence[str]],
    legacy_period_set_id: Optional[str],
    legacy_period_id: Optional[str],
) -> List[Tuple[str, Optional[str]]]:
    targets = normalize_period_target_values(period_targets)
    legacy_set = (legacy_period_set_id or "").strip()
    legacy_id = normalize_optional_period_id(legacy_period_id)

    if not targets:
        if legacy_id:
            targets = [legacy_id]
        elif legacy_set:
            targets = [legacy_set]
        else:
            prompted = prompt_nonempty(
                (
                    "Enter period target(s) "
                    + "(period_id or period_set_id; multiple allowed with ; or comma; "
                    + "use global/global_all for global scope)"
                ),
                None,
            )
            targets = normalize_period_target_values([prompted])

    resolved: List[Tuple[str, Optional[str]]] = []
    seen: set[Tuple[str, Optional[str]]] = set()
    for target in targets:
        item = resolve_one_period_target(target, legacy_period_set_id, legacy_period_id)
        if item not in seen:
            resolved.append(item)
            seen.add(item)
    return resolved


def ensure_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")


def load_period_ids_for_set(periods_path: Path, period_set_id: str) -> List[str]:
    if period_set_id == "global_all":
        return []
    if not periods_path.exists():
        raise FileNotFoundError(f"Periods parquet not found: {periods_path}")
    periods_df = pd.read_parquet(periods_path, columns=["period_set_id", "period_id", "sort_order"])
    subset = periods_df[periods_df["period_set_id"] == period_set_id].copy()
    if subset.empty:
        raise ValueError(f"No period IDs found for period_set_id={period_set_id}.")
    subset = subset.sort_values(["sort_order", "period_id"], kind="mergesort")
    return [str(value) for value in subset["period_id"].tolist()]


def run_command(command: Sequence[str]) -> None:
    log("RUN " + " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def build_effective_dataset_label(dataset_label: str, profile: str) -> str:
    return f"{dataset_label}_applied_{profile}"


def profile_token_dir_name(profile: str, prefix: str) -> str:
    profile_dir = "regex_only" if profile == "regex-only" else profile
    return f"{prefix}{profile_dir}"


def profile_dir_path(args: argparse.Namespace, profile: str) -> Path:
    prefix = (args.applied_token_dir_prefix or "").strip()
    return (
        Path(args.applied_tokens_dir).expanduser().resolve()
        / profile_token_dir_name(profile, prefix)
    )


def resolve_profile_tokens_path(args: argparse.Namespace, profile: str) -> Optional[Path]:
    prefix = (args.applied_token_dir_prefix or "").strip()
    if not prefix:
        return None
    tokens_path = profile_dir_path(args, profile) / "tokens.parquet"
    if not tokens_path.exists():
        raise FileNotFoundError(f"Applied tokens parquet not found: {tokens_path}")
    return tokens_path


def resolve_profile_periods_path(args: argparse.Namespace, profile: str) -> Path:
    if args.periods_parquet is not None:
        periods_path = args.periods_parquet.expanduser().resolve()
    else:
        periods_path = (profile_dir_path(args, profile) / "periods" / "periods.parquet").resolve()
    if not periods_path.exists():
        raise FileNotFoundError(f"Periods parquet not found for profile={profile}: {periods_path}")
    return periods_path


def build_network_base_path(
    dataset_label: str,
    profile: str,
    center_mode: str,
    window_size: int,
    period_scope: str,
    context_filter_info,
) -> Path:
    filename = (
        f"network_{sanitize_slug(build_effective_dataset_label(dataset_label, profile))}"
        f"_{sanitize_slug(center_mode)}"
        f"_w{window_size}_split-lr_none_raw-freq_{sanitize_slug(period_scope)}.csv"
    )
    if context_filter_info:
        filename = filename[:-4] + f"{context_filter_stem_part(context_filter_info)}.csv"
    return NETWORK_APPLIED_DIR / filename


def build_period_folder_name(
    period_set_id: str,
    period_id: Optional[str],
    context_filter_info=None,
    output_scope_suffix: str = "",
) -> str:
    if period_id:
        name = sanitize_slug(period_id)
    elif period_set_id == "global_all":
        name = "global"
    else:
        name = sanitize_slug(period_set_id)
    name = f"{name}{context_filter_stem_part(context_filter_info)}"
    suffix = sanitize_slug(output_scope_suffix.strip()) if output_scope_suffix else ""
    if suffix:
        name = f"{name}_{suffix}"
    return name


def build_stopfiltered_dir(period_folder_name: str) -> Path:
    return NETWORK_APPLIED_DIR / f"stopfiltered_{period_folder_name}"


def find_network_output_parts(base_output_path: Path, period_folder_name: str) -> List[Path]:
    part_files = sorted(base_output_path.parent.glob(f"{base_output_path.stem}.part*.csv"))
    if part_files:
        return part_files
    if base_output_path.exists():
        return [base_output_path]
    archived_part_dir = PART_FILTERED_DIR / period_folder_name
    archived_parts = sorted(archived_part_dir.glob(f"{base_output_path.stem}.part*.csv"))
    if archived_parts:
        return archived_parts
    archived_base = archived_part_dir / base_output_path.name
    if archived_base.exists():
        return [archived_base]
    raise FileNotFoundError(f"No network output found for stem: {base_output_path.stem}")


def build_part_done_marker_path(part_path: Path) -> Path:
    return part_path.with_name(f"{part_path.name}.done.json")


def build_empty_network_marker_path(base_output_path: Path) -> Path:
    return base_output_path.with_name(f"{base_output_path.stem}.empty.json")


def detect_existing_empty_network_marker(
    base_output_path: Path,
    period_folder_name: str,
) -> Tuple[Optional[str], Optional[Path]]:
    root_marker = build_empty_network_marker_path(base_output_path)
    if root_marker.exists():
        return "root", root_marker
    archived_marker = DONE_DIR / period_folder_name / root_marker.name
    if archived_marker.exists():
        return "archive", archived_marker
    return None, None


def list_completed_network_parts(
    base_output_path: Path,
    part_dir: Path,
    marker_dir: Optional[Path] = None,
) -> List[Path]:
    effective_marker_dir = marker_dir if marker_dir is not None else part_dir
    completed_parts: List[Path] = []
    for part_path in sorted(part_dir.glob(f"{base_output_path.stem}.part*.csv")):
        marker_path = effective_marker_dir / f"{part_path.name}.done.json"
        if marker_path.exists():
            completed_parts.append(part_path)
    return completed_parts


def detect_existing_completed_network_parts(
    base_output_path: Path,
    period_folder_name: str,
) -> Tuple[Optional[str], List[Path]]:
    root_parts = list_completed_network_parts(base_output_path, base_output_path.parent)
    if root_parts:
        return "root", root_parts
    archived_part_dir = PART_FILTERED_DIR / period_folder_name
    archived_done_dir = DONE_DIR / period_folder_name
    archived_parts = list_completed_network_parts(
        base_output_path,
        archived_part_dir,
        archived_done_dir,
    )
    if archived_parts:
        return "archive", archived_parts
    return None, []


def collect_raw_joint_distribution(paths: Sequence[Path]) -> Tuple[Counter, int]:
    distribution: Counter = Counter()
    total_rows = 0
    count_index: Optional[int] = None
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                continue
            if count_index is None:
                if "raw_joint_event_count" not in header:
                    raise ValueError(f"raw_joint_event_count column not found: {path}")
                count_index = header.index("raw_joint_event_count")
            for row in reader:
                total_rows += 1
                try:
                    value = int(float(row[count_index]))
                except (ValueError, IndexError):
                    continue
                distribution[value] += 1
    if total_rows == 0:
        raise ValueError("No rows available for threshold selection.")
    return distribution, total_rows


def choose_adaptive_threshold(
    distribution: Counter,
    total_rows: int,
    target_ratio: float,
    floor_value: int,
) -> Tuple[int, int, float]:
    sorted_counts = sorted(distribution.items(), key=lambda item: item[0], reverse=True)
    retained = 0
    candidates: List[Tuple[float, int, int, float]] = []
    for value, freq in sorted_counts:
        retained += freq
        retained_ratio = retained / total_rows
        if value < floor_value:
            continue
        candidates.append((abs(retained_ratio - target_ratio), value, retained, retained_ratio))
    if not candidates:
        threshold = max(floor_value, max(distribution))
        retained_rows = sum(freq for value, freq in distribution.items() if value >= threshold)
        retained_ratio = retained_rows / total_rows
        return threshold, retained_rows, retained_ratio
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, threshold, retained_rows, retained_ratio = candidates[0]
    return threshold, retained_rows, retained_ratio


def write_joint_filtered_csv(
    input_paths: Sequence[Path],
    output_path: Path,
    threshold: int,
) -> int:
    header_written = False
    kept_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = None
        count_index: Optional[int] = None
        for path in input_paths:
            with path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.reader(in_handle)
                header = next(reader, None)
                if header is None:
                    continue
                if count_index is None:
                    if "raw_joint_event_count" not in header:
                        raise ValueError(f"raw_joint_event_count column not found: {path}")
                    count_index = header.index("raw_joint_event_count")
                if not header_written:
                    writer = csv.writer(out_handle)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    try:
                        value = int(float(row[count_index]))
                    except (ValueError, IndexError):
                        continue
                    if value >= threshold:
                        writer.writerow(row)
                        kept_rows += 1
    if not header_written:
        raise ValueError(f"No readable CSV rows found for output: {output_path}")
    return kept_rows


def build_joint_output_path(base_output_path: Path, threshold: int) -> Path:
    return base_output_path.with_name(f"{base_output_path.stem}_joint{threshold}up.csv")


def find_existing_joint_output_path(
    base_output_path: Path,
    threshold: int,
    period_folder_name: str,
) -> Optional[Path]:
    joint_output_path = build_joint_output_path(base_output_path, threshold)
    if joint_output_path.exists():
        return joint_output_path
    archived_path = JOINTUP_DIR / period_folder_name / joint_output_path.name
    if archived_path.exists():
        return archived_path
    return None


def resolve_stopword_version_label(rule_dir: Path, profile: str) -> str:
    def latest_version(stem: str) -> int:
        pattern = re.compile(rf"^{re.escape(stem)}_v(?P<version>\d+)\.csv$")
        versions: List[int] = []
        for path in rule_dir.glob(f"{stem}_v*.csv"):
            match = pattern.fullmatch(path.name)
            if match:
                versions.append(int(match.group("version")))
        if not versions:
            raise FileNotFoundError(f"No runtime stopword rule file found for {stem}_v*.csv")
        return max(versions)

    versions = [
        latest_version("stopword_exact_always"),
        latest_version("stopword_regex_always"),
    ]
    if profile == "full":
        versions.append(latest_version("stopword_exact_optional"))
    uniq = sorted(set(versions))
    if len(uniq) == 1:
        return f"v{uniq[0]}"
    return "mixedv" + "-".join(str(v) for v in uniq)


def move_file_if_exists(source: Path, target_dir: Path) -> bool:
    if not source.exists():
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source.name
    if source.resolve() == target_path.resolve():
        return False
    source.replace(target_path)
    return True


def parse_stopfiltered_filename(path: Path) -> Tuple[str, str, int, str, str, Optional[int], str]:
    match = STOPFILTER_FILENAME_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unsupported stopfiltered filename pattern: {path.name}")
    joint_raw = match.group("joint_threshold")
    return (
        match.group("dataset_label"),
        match.group("center_mode"),
        int(match.group("window_size")),
        match.group("period_scope"),
        match.group("context_filter") or "",
        int(joint_raw) if joint_raw else None,
        match.group("stopword_version"),
    )


def build_nodes_base_name(filtered_path: Path) -> str:
    dataset_label, center_mode, window_size, _period_scope, context_filter, joint_threshold, stopword_version = (
        parse_stopfiltered_filename(filtered_path)
    )
    if center_mode == "all-tokens":
        if joint_threshold is None:
            raise ValueError("all-tokens node output requires joint_threshold metadata.")
        return (
            f"pmi_nodes_{dataset_label}_all-tokens_"
            f"w{window_size}_joint{joint_threshold}{context_filter}_stopv{stopword_version}"
        )
    return f"pmi_nodes_{dataset_label}_keyword-only_w{window_size}{context_filter}_stopv{stopword_version}"


def node_output_matches_period(path: Path, expected_period_scope: str) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    if first_row is None:
        return False
    period_id = (first_row.get("period_id") or "").strip()
    period_set_id = (first_row.get("period_set_id") or "").strip()
    return expected_period_scope in {period_id, period_set_id}


def detect_existing_nodes_outputs(
    filtered_path: Path,
    output_mode: str,
) -> Tuple[bool, List[Path]]:
    nodes_output_dir = ROOT / "shenbao" / "shenbao_network" / "pmi_nodes"
    _dataset_label, _center_mode, _window_size, period_scope, _context_filter, _joint_threshold, _stopword_version = (
        parse_stopfiltered_filename(filtered_path)
    )
    base_name = build_nodes_base_name(filtered_path)
    expected_paths: List[Path] = []
    if output_mode in {"minimal", "both"}:
        expected_paths.append(nodes_output_dir / f"{base_name}.csv")
    if output_mode in {"extended", "both"}:
        expected_paths.append(nodes_output_dir / f"{base_name}_extended.csv")
    for path in expected_paths:
        if not path.exists():
            return False, expected_paths
        if not node_output_matches_period(path, period_scope):
            return False, expected_paths
    return True, expected_paths


def finalize_output_artifacts(
    period_folder_name: str,
    network_part_paths: Sequence[Path],
    empty_marker_paths: Sequence[Path],
    joint_output_paths: Sequence[Path],
    stopword_filtered_paths: Sequence[Path],
) -> Tuple[int, int, int, int, int]:
    moved_done = 0
    moved_parts = 0
    moved_empty = 0
    moved_jointup = 0
    moved_summaries = 0

    for part_path in network_part_paths:
        if move_file_if_exists(part_path, PART_FILTERED_DIR / period_folder_name):
            moved_parts += 1
        marker_path = build_part_done_marker_path(part_path)
        if move_file_if_exists(marker_path, DONE_DIR / period_folder_name):
            moved_done += 1

    for marker_path in empty_marker_paths:
        if move_file_if_exists(marker_path, DONE_DIR / period_folder_name):
            moved_empty += 1

    for joint_output_path in joint_output_paths:
        if move_file_if_exists(joint_output_path, JOINTUP_DIR / period_folder_name):
            moved_jointup += 1

    for filtered_path in stopword_filtered_paths:
        summary_path = filtered_path.with_suffix(".summary.json")
        if move_file_if_exists(summary_path, STOPFILTER_SUMMARY_DIR / period_folder_name):
            moved_summaries += 1

    return moved_done, moved_parts, moved_empty, moved_jointup, moved_summaries


def run_one_period(
    args: argparse.Namespace,
    dataset_label: str,
    keywords: str,
    period_set_id: str,
    period_id: Optional[str],
    window_sizes: Sequence[int],
    center_mode: str,
    boundary_pos: str,
    stopword_profile: str,
    stopword_version_label: str,
) -> None:
    period_scope = period_id if period_id else period_set_id
    period_folder_name = build_period_folder_name(
        period_set_id,
        period_id,
        args.context_filter_info,
        args.output_scope_suffix,
    )
    stopfiltered_output_dir = build_stopfiltered_dir(period_folder_name)
    profiles = args.profiles

    network_base_outputs: List[Tuple[str, int, Path]] = []
    network_part_paths: List[Path] = []
    empty_marker_paths: List[Path] = []
    joint_outputs: List[Path] = []
    stopword_outputs: List[Path] = []

    log(
        f"Start one-stop run: dataset={dataset_label}, period_set={period_set_id}, "
        f"period_id={period_id or 'ALL'}, center_mode={center_mode}, windows={window_sizes}"
    )

    for profile in profiles:
        for window_size in window_sizes:
            output_base = build_network_base_path(
                dataset_label=dataset_label,
                profile=profile,
                center_mode=center_mode,
                window_size=window_size,
                period_scope=period_scope,
                context_filter_info=args.context_filter_info,
            )
            existing_location, existing_parts = detect_existing_completed_network_parts(
                output_base,
                period_folder_name,
            )
            if existing_parts:
                log(
                    f"network skip: profile={profile}, w={window_size}, "
                    f"completed parts found in {existing_location} ({len(existing_parts)})"
                )
                network_base_outputs.append((profile, window_size, output_base))
                continue
            empty_location, empty_marker_path = detect_existing_empty_network_marker(
                output_base,
                period_folder_name,
            )
            if empty_marker_path is not None:
                log(
                    f"network empty skip: profile={profile}, w={window_size}, "
                    f"empty marker found in {empty_location}"
                )
                empty_marker_paths.append(empty_marker_path)
                continue
            command = [
                sys.executable,
                str(NETWORK_APPLIED_SCRIPT),
                "--dataset-label",
                dataset_label,
                "--keywords",
                keywords,
                "--center-mode",
                center_mode,
                "--window-size",
                str(window_size),
                "--boundary-pos",
                boundary_pos,
                "--period-set-id",
                period_set_id,
                "--periods-parquet",
                str(resolve_profile_periods_path(args, profile)),
                "--window-type",
                "split-lr",
                "--distance-weight",
                "none",
                "--count-mode",
                "raw-freq",
                "--rows-per-output-file",
                str(args.network_rows_per_output_file),
                "--write-chunk-size",
                str(args.network_write_chunk_size),
                "--write-pause-seconds",
                str(args.network_write_pause_seconds),
                "--context-filter",
                args.context_filter,
                "--context-filter-root",
                str(Path(args.context_filter_root).expanduser().resolve()),
                "--output-csv",
                str(output_base),
            ]
            tokens_path = resolve_profile_tokens_path(args, profile)
            if tokens_path is None:
                command.extend(["--applied-token-profile", profile])
            else:
                command.extend(["--tokens-parquet", str(tokens_path)])
            if period_id:
                command.extend(["--period-id", period_id])
            run_command(command)
            empty_location, empty_marker_path = detect_existing_empty_network_marker(
                output_base,
                period_folder_name,
            )
            if empty_marker_path is not None:
                log(
                    f"network empty: profile={profile}, w={window_size}, "
                    f"no surviving rows after final filters ({empty_location})"
                )
                empty_marker_paths.append(empty_marker_path)
                continue
            network_base_outputs.append((profile, window_size, output_base))

    for profile, window_size, output_base in network_base_outputs:
        input_paths = find_network_output_parts(output_base, period_folder_name)
        network_part_paths.extend(input_paths)
        distribution, total_rows = collect_raw_joint_distribution(input_paths)
        target_ratio = args.retention_ratio_w1 if window_size == 1 else args.retention_ratio_other
        threshold, kept_rows, kept_ratio = choose_adaptive_threshold(
            distribution=distribution,
            total_rows=total_rows,
            target_ratio=target_ratio,
            floor_value=args.min_joint_floor,
        )
        existing_joint_output = find_existing_joint_output_path(
            output_base,
            threshold,
            period_folder_name,
        )
        if existing_joint_output is not None:
            joint_output_path = existing_joint_output
            log(
                f"joint filter skip: profile={profile}, w={window_size}, threshold={threshold}, "
                f"existing={joint_output_path}"
            )
        else:
            joint_output_path = build_joint_output_path(output_base, threshold)
            actual_kept_rows = write_joint_filtered_csv(
                input_paths=input_paths,
                output_path=joint_output_path,
                threshold=threshold,
            )
            if actual_kept_rows != kept_rows:
                kept_rows = actual_kept_rows
                kept_ratio = kept_rows / total_rows if total_rows else 0.0
        log(
            f"joint filter: profile={profile}, w={window_size}, threshold={threshold}, "
            f"kept={kept_rows:,}/{total_rows:,} ({kept_ratio:.2%}), target={target_ratio:.2%}"
        )
        joint_outputs.append(joint_output_path)

    for joint_output_path in joint_outputs:
        expected_stopfiltered_path = stopfiltered_output_dir / (
            f"{joint_output_path.stem}_stop{stopword_version_label}{stopword_profile}filtered.csv"
        )
        if expected_stopfiltered_path.exists():
            log(f"stopfilter skip: existing={expected_stopfiltered_path}")
            stopword_outputs.append(expected_stopfiltered_path)
            continue
        command = [
            sys.executable,
            str(STOPWORD_SCRIPT),
            "--input-csv",
            str(joint_output_path),
            "--stopword-profile",
            stopword_profile,
            "--runtime-rule-dir",
            str(RUNTIME_RULE_DIR),
            "--output-dir",
            str(stopfiltered_output_dir),
        ]
        run_command(command)
        stopword_outputs.append(expected_stopfiltered_path)

    for filtered_path in stopword_outputs:
        if not filtered_path.exists():
            raise FileNotFoundError(f"Expected stopword-filtered CSV not found: {filtered_path}")
        nodes_ready, node_outputs = detect_existing_nodes_outputs(filtered_path, args.nodes_output_mode)
        if nodes_ready:
            log(f"nodes skip: existing={'; '.join(str(path) for path in node_outputs)}")
            continue
        command = [
            sys.executable,
            str(NODES_SCRIPT),
            "--input-csv",
            str(filtered_path),
            "--output-mode",
            args.nodes_output_mode,
        ]
        run_command(command)

    moved_done, moved_parts, moved_empty, moved_jointup, moved_summaries = finalize_output_artifacts(
        period_folder_name=period_folder_name,
        network_part_paths=network_part_paths,
        empty_marker_paths=empty_marker_paths,
        joint_output_paths=joint_outputs,
        stopword_filtered_paths=stopword_outputs,
    )

    log("One-stop run completed.")
    log(f"Network files: {len(network_base_outputs)}")
    log(f"Empty network outputs: {len(empty_marker_paths)}")
    log(f"Joint-filtered files: {len(joint_outputs)}")
    log(f"Stopword-filtered files: {len(stopword_outputs)}")
    log(f"Moved done markers: {moved_done}")
    log(f"Moved part files: {moved_parts}")
    log(f"Moved empty markers: {moved_empty}")
    log(f"Moved jointup files: {moved_jointup}")
    log(f"Moved stopfilter summaries: {moved_summaries}")


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    args.profiles = parse_profiles(args.profiles)
    args.context_filter_root = Path(args.context_filter_root).expanduser().resolve()
    args.context_filter_info = load_context_filter(args.context_filter, args.context_filter_root)

    for script_path in (NETWORK_APPLIED_SCRIPT, STOPWORD_SCRIPT, NODES_SCRIPT):
        ensure_script(script_path)

    dataset_label = prompt_nonempty("Enter dataset-label", args.dataset_label)
    keywords = prompt_nonempty("Enter keywords (comma-separated)", args.keywords)
    resolved_targets = resolve_period_targets(
        period_targets=args.period_target,
        legacy_period_set_id=args.period_set_id,
        legacy_period_id=args.period_id,
    )

    window_sizes = parse_window_sizes(args.window_sizes)
    center_mode = args.center_mode
    boundary_pos = args.boundary_pos
    stopword_profile = args.stopword_profile
    stopword_version_label = resolve_stopword_version_label(RUNTIME_RULE_DIR, stopword_profile)

    if args.context_filter_info:
        log(
            "Context filter: "
            f"{args.context_filter_info.name} "
            f"({len(args.context_filter_info.excluded_context_uids):,} excluded context_uid)"
        )
    else:
        log("Context filter: none")

    if not (0.0 < args.retention_ratio_w1 <= 1.0):
        raise ValueError("--retention-ratio-w1 must be in (0, 1].")
    if not (0.0 < args.retention_ratio_other <= 1.0):
        raise ValueError("--retention-ratio-other must be in (0, 1].")
    if args.min_joint_floor < 1:
        raise ValueError("--min-joint-floor must be >= 1.")

    run_targets: List[Tuple[str, Optional[str]]] = []
    seen_run_targets: set[Tuple[str, Optional[str]]] = set()
    for period_set_id, period_id in resolved_targets:
        if period_id:
            candidates = [(period_set_id, period_id)]
        elif period_set_id == "global_all":
            candidates = [(period_set_id, None)]
        else:
            period_ids = load_period_ids_for_set(DEFAULT_PERIODS_PARQUET, period_set_id)
            log(f"Expanded period_set_id={period_set_id} into {len(period_ids)} period(s).")
            candidates = [(period_set_id, target_period_id) for target_period_id in period_ids]
        for candidate in candidates:
            if candidate not in seen_run_targets:
                run_targets.append(candidate)
                seen_run_targets.add(candidate)

    for target_period_set_id, target_period_id in run_targets:
        run_one_period(
            args=args,
            dataset_label=dataset_label,
            keywords=keywords,
            period_set_id=target_period_set_id,
            period_id=target_period_id,
            window_sizes=window_sizes,
            center_mode=center_mode,
            boundary_pos=boundary_pos,
            stopword_profile=stopword_profile,
            stopword_version_label=stopword_version_label,
        )


if __name__ == "__main__":
    main()
