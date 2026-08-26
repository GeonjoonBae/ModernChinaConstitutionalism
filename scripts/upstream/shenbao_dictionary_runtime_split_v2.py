from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DICTIONARY_DIR = ROOT / "shenbao" / "shenbao_dictionary"
DEFAULT_VARIANT_MASTER = DEFAULT_DICTIONARY_DIR / "variant_char_normalization_rules_v2.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_DICTIONARY_DIR / "runtime_rule_splits"


@dataclass(frozen=True)
class VersionedFile:
    path: Path
    version: int


def raise_csv_field_limit() -> None:
    max_size = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV header missing: {path}")
                rows = list(reader)
                return list(reader.fieldnames), rows
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def write_csv_dicts(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def normalize_value(value: str | None) -> str:
    return (value or "").strip()


def find_latest_versioned_file(directory: Path, regex: re.Pattern[str], glob_pattern: str) -> VersionedFile:
    matches: list[VersionedFile] = []
    for path in directory.glob(glob_pattern):
        match = regex.fullmatch(path.name)
        if not match:
            continue
        matches.append(VersionedFile(path=path.resolve(), version=int(match.group("version"))))
    if not matches:
        raise FileNotFoundError(f"No files matching {glob_pattern} in {directory}")
    return max(matches, key=lambda item: item.version)


def resolve_dictionary_master(directory: Path, explicit_path: Path | None) -> VersionedFile:
    if explicit_path is not None:
        match = re.fullmatch(r"dictionary_v(?P<version>\d+)_master\.csv", explicit_path.name)
        version = int(match.group("version")) if match else 0
        return VersionedFile(path=explicit_path.expanduser().resolve(), version=version)
    return find_latest_versioned_file(
        directory,
        re.compile(r"dictionary_v(?P<version>\d+)_master\.csv"),
        "dictionary_v*_master.csv",
    )


def resolve_stopword_master(directory: Path, explicit_path: Path | None) -> VersionedFile:
    if explicit_path is not None:
        match = re.fullmatch(r"stopword_v(?P<version>\d+)_master\.csv", explicit_path.name)
        version = int(match.group("version")) if match else 0
        return VersionedFile(path=explicit_path.expanduser().resolve(), version=version)
    return find_latest_versioned_file(
        directory,
        re.compile(r"stopword_v(?P<version>\d+)_master\.csv"),
        "stopword_v*_master.csv",
    )


def resolve_variant_master(directory: Path, explicit_path: Path | None) -> VersionedFile:
    if explicit_path is not None:
        match = re.fullmatch(r"variant_char_normalization_rules_v(?P<version>\d+)\.csv", explicit_path.name)
        version = int(match.group("version")) if match else 0
        return VersionedFile(path=explicit_path.expanduser().resolve(), version=version)
    return find_latest_versioned_file(
        directory,
        re.compile(r"variant_char_normalization_rules_v(?P<version>\d+)\.csv"),
        "variant_char_normalization_rules_v*.csv",
    )


def split_dictionary_rules(
    rows: list[dict[str, str]],
    dictionary_version: int,
) -> dict[str, list[dict[str, str]]]:
    exact_merge: list[dict[str, str]] = []
    exact_merge_optional: list[dict[str, str]] = []
    regex_merge: list[dict[str, str]] = []

    for row in rows:
        dict_action = normalize_value(row.get("dict_action"))
        domain_tag = normalize_value(row.get("domain_tag"))

        if domain_tag == "regex":
            regex_merge.append(row)
            continue
        if dict_action == "merge":
            exact_merge.append(row)
            continue
        if dict_action == "merge_optional":
            exact_merge_optional.append(row)
            continue

    suffix = f"v{dictionary_version}" if dictionary_version else "custom"
    return {
        f"dictionary_exact_merge_{suffix}.csv": exact_merge,
        f"dictionary_exact_merge_optional_{suffix}.csv": exact_merge_optional,
        f"dictionary_regex_merge_{suffix}.csv": regex_merge,
    }


def split_stopword_rules(
    rows: list[dict[str, str]],
    stopword_version: int,
) -> dict[str, list[dict[str, str]]]:
    exact_always: list[dict[str, str]] = []
    exact_optional: list[dict[str, str]] = []
    regex_always: list[dict[str, str]] = []

    for row in rows:
        stop_layer = normalize_value(row.get("stop_layer"))
        rule_type = normalize_value(row.get("type"))

        if rule_type:
            if stop_layer == "always":
                regex_always.append(row)
            continue

        if stop_layer == "always":
            exact_always.append(row)
            continue
        if stop_layer == "optional":
            exact_optional.append(row)
            continue

    suffix = f"v{stopword_version}" if stopword_version else "custom"
    return {
        f"stopword_exact_always_{suffix}.csv": exact_always,
        f"stopword_exact_optional_{suffix}.csv": exact_optional,
        f"stopword_regex_always_{suffix}.csv": regex_always,
    }


def copy_variant_rules(
    rows: list[dict[str, str]],
    variant_version: int,
) -> dict[str, list[dict[str, str]]]:
    suffix = f"v{variant_version}" if variant_version else "custom"
    return {f"variant_char_normalization_runtime_{suffix}.csv": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split latest Shenbao dictionary_v*_master.csv and stopword_v*_master.csv "
            "from shenbao_dictionary into runtime rule CSVs."
        )
    )
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=DEFAULT_DICTIONARY_DIR,
        help=f"Directory containing dictionary_v*_master.csv and stopword_v*_master.csv (default: {DEFAULT_DICTIONARY_DIR})",
    )
    parser.add_argument(
        "--dictionary-master",
        type=Path,
        help="Optional explicit dictionary master CSV. Default: latest dictionary_v*_master.csv in --dictionary-dir.",
    )
    parser.add_argument(
        "--stopword-master",
        type=Path,
        help="Optional explicit stopword master CSV. Default: latest stopword_v*_master.csv in --dictionary-dir.",
    )
    parser.add_argument(
        "--variant-master",
        type=Path,
        help="Optional explicit variant normalization CSV. Default: latest variant_char_normalization_rules_v*.csv in --dictionary-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report split counts without writing output files.",
    )
    return parser


def main() -> None:
    raise_csv_field_limit()
    parser = build_parser()
    args = parser.parse_args()

    dictionary_dir = args.dictionary_dir.expanduser().resolve()
    dictionary_master = resolve_dictionary_master(dictionary_dir, args.dictionary_master)
    stopword_master = resolve_stopword_master(dictionary_dir, args.stopword_master)
    variant_master = resolve_variant_master(dictionary_dir, args.variant_master)
    output_dir = args.output_dir.expanduser().resolve()

    dictionary_fields, dictionary_rows = read_csv_dicts(dictionary_master.path)
    stopword_fields, stopword_rows = read_csv_dicts(stopword_master.path)
    variant_fields, variant_rows = read_csv_dicts(variant_master.path)

    print(f"DICTIONARY_MASTER\t{dictionary_master.path}\tv{dictionary_master.version}")
    print(f"STOPWORD_MASTER\t{stopword_master.path}\tv{stopword_master.version}")
    print(f"VARIANT_MASTER\t{variant_master.path}\tv{variant_master.version}")

    planned_outputs: list[tuple[str, list[str], list[dict[str, str]]]] = []

    for filename, rows in split_dictionary_rules(dictionary_rows, dictionary_master.version).items():
        planned_outputs.append((filename, dictionary_fields, rows))
    for filename, rows in split_stopword_rules(stopword_rows, stopword_master.version).items():
        planned_outputs.append((filename, stopword_fields, rows))
    for filename, rows in copy_variant_rules(variant_rows, variant_master.version).items():
        planned_outputs.append((filename, variant_fields, rows))

    for filename, fieldnames, rows in planned_outputs:
        output_path = output_dir / filename
        if args.dry_run:
            print(f"DRY-RUN\t{output_path}\t{len(rows)}")
            continue
        written = write_csv_dicts(output_path, fieldnames, rows)
        print(f"WROTE\t{output_path}\t{written}")


if __name__ == "__main__":
    main()
