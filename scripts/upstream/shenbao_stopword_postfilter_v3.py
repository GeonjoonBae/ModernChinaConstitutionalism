#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_RULE_DIR = (
    ROOT / "shenbao" / "shenbao_dictionary" / "runtime_rule_splits"
)


@dataclass(frozen=True)
class RuntimeRuleFile:
    path: Path
    version: int


@dataclass(frozen=True)
class RegexRule:
    pattern: str
    compiled: re.Pattern[str]
    rule_type: str


@dataclass(frozen=True)
class StopwordRuntimeRules:
    exact_always_path: Path
    exact_always_version: int
    exact_optional_path: Optional[Path]
    exact_optional_version: Optional[int]
    regex_always_path: Path
    regex_always_version: int
    version_label: str


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter network CSV rows using latest runtime stopword split files. "
            "always loads exact_always and regex_always; full also loads exact_optional."
        )
    )
    parser.add_argument(
        "--input-csv",
        nargs="+",
        help="One or more network CSV files to filter. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--runtime-rule-dir",
        default=str(DEFAULT_RUNTIME_RULE_DIR),
        help="Directory containing stopword_exact_always_v*.csv etc.",
    )
    parser.add_argument(
        "--stopword-profile",
        choices=["always", "full"],
        help="always = exact_always + regex_always; full = always profile + exact_optional.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--summary-json",
        default="true",
        choices=["true", "false"],
        help="Whether to write a summary JSON next to each filtered CSV. Default: true.",
    )
    return parser.parse_args()


def prompt_nonempty(message: str) -> str:
    while True:
        value = input(f"{message}: ").strip()
        if value:
            return value
        print("Input cannot be empty.", flush=True)


def prompt_choice(message: str, allowed: Sequence[str]) -> str:
    allowed_text = "/".join(allowed)
    while True:
        value = input(f"{message} [{allowed_text}]: ").strip().lower()
        if value in allowed:
            return value
        print(f"Input must be one of: {allowed_text}", flush=True)


def read_dict_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                if fieldnames is None:
                    raise ValueError(f"CSV has no header: {path}")
                return list(fieldnames), list(reader)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV: {path}")


def find_latest_runtime_file(rule_dir: Path, stem: str) -> RuntimeRuleFile:
    pattern = re.compile(rf"^{re.escape(stem)}_v(?P<version>\d+)\.csv$")
    matches: List[RuntimeRuleFile] = []
    for path in rule_dir.glob(f"{stem}_v*.csv"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        matches.append(RuntimeRuleFile(path=path.resolve(), version=int(match.group("version"))))
    if not matches:
        raise FileNotFoundError(f"No runtime rule file found for {stem}_v*.csv in {rule_dir}")
    return max(matches, key=lambda item: item.version)


def resolve_runtime_rules(rule_dir: Path, profile: str) -> StopwordRuntimeRules:
    exact_always = find_latest_runtime_file(rule_dir, "stopword_exact_always")
    regex_always = find_latest_runtime_file(rule_dir, "stopword_regex_always")
    exact_optional: Optional[RuntimeRuleFile] = None
    if profile == "full":
        exact_optional = find_latest_runtime_file(rule_dir, "stopword_exact_optional")

    versions = [exact_always.version, regex_always.version]
    if exact_optional is not None:
        versions.append(exact_optional.version)
    version_label = f"v{versions[0]}" if len(set(versions)) == 1 else "mixedv" + "-".join(str(v) for v in sorted(set(versions)))

    return StopwordRuntimeRules(
        exact_always_path=exact_always.path,
        exact_always_version=exact_always.version,
        exact_optional_path=exact_optional.path if exact_optional else None,
        exact_optional_version=exact_optional.version if exact_optional else None,
        regex_always_path=regex_always.path,
        regex_always_version=regex_always.version,
        version_label=version_label,
    )


def resolve_inputs(
    args: argparse.Namespace,
) -> Tuple[List[Path], Path, str, Optional[Path], bool]:
    if args.input_csv:
        input_paths = [Path(raw_path).expanduser().resolve() for raw_path in args.input_csv]
    else:
        raw_input_paths = prompt_nonempty(
            "Enter input CSV path(s). Use ';' to separate multiple files"
        )
        input_paths = [
            Path(part.strip()).expanduser().resolve()
            for part in raw_input_paths.split(";")
            if part.strip()
        ]
    if not input_paths:
        raise ValueError("No input CSV paths were provided.")
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"input CSV not found: {input_path}")

    rule_dir = Path(args.runtime_rule_dir).expanduser().resolve()
    if not rule_dir.exists():
        raise FileNotFoundError(f"runtime rule directory not found: {rule_dir}")

    profile = args.stopword_profile or prompt_choice(
        "Select stopword profile", ("always", "full")
    )

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()

    return input_paths, rule_dir, profile, output_dir, (args.summary_json == "true")


def load_exact_stopwords(paths: Sequence[Path]) -> Tuple[Set[str], int]:
    exact_tokens: Set[str] = set()
    loaded_rows = 0
    for path in paths:
        _, rows = read_dict_rows(path)
        for row in rows:
            token = (row.get("token") or "").strip()
            if not token:
                continue
            exact_tokens.add(token)
            loaded_rows += 1
    return exact_tokens, loaded_rows


def load_regex_stopwords(path: Path) -> Tuple[List[RegexRule], int]:
    _, rows = read_dict_rows(path)
    regex_rules: List[RegexRule] = []
    loaded_rows = 0
    for idx, row in enumerate(rows, start=2):
        token = (row.get("token") or "").strip()
        if not token:
            continue
        rule_type = (row.get("type") or "").strip()
        try:
            compiled = re.compile(token)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in {path.name}:{idx}: {token} ({exc})"
            ) from exc
        regex_rules.append(RegexRule(pattern=token, compiled=compiled, rule_type=rule_type))
        loaded_rows += 1
    return regex_rules, loaded_rows


def build_output_path(
    input_path: Path,
    output_dir: Optional[Path],
    profile: str,
    version_label: str,
) -> Path:
    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{input_path.stem}_stop{version_label}{profile}filtered.csv"


def build_summary_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_suffix(".summary.json")


def token_matches_regex(token: str, regex_rules: Sequence[RegexRule]) -> bool:
    for rule in regex_rules:
        if rule.compiled.fullmatch(token):
            return True
    return False


def filter_rows(
    input_path: Path,
    output_csv_path: Path,
    summary_json: bool,
    exact_stopwords: Set[str],
    regex_rules: Sequence[RegexRule],
    runtime_rules: StopwordRuntimeRules,
    profile: str,
) -> None:
    fieldnames, rows = read_dict_rows(input_path)
    if "center_token" not in fieldnames or "neighbor_token" not in fieldnames:
        raise ValueError(
            f"Input CSV must contain center_token and neighbor_token columns: {input_path}"
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    removed_rows = 0
    removed_by_center = 0
    removed_by_neighbor = 0
    removed_by_both = 0
    removed_by_exact = 0
    removed_by_regex = 0
    kept_rows = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            input_rows += 1
            center_token = row["center_token"].strip()
            neighbor_token = row["neighbor_token"].strip()

            center_exact = center_token in exact_stopwords
            neighbor_exact = neighbor_token in exact_stopwords
            center_regex = token_matches_regex(center_token, regex_rules)
            neighbor_regex = token_matches_regex(neighbor_token, regex_rules)

            center_hit = center_exact or center_regex
            neighbor_hit = neighbor_exact or neighbor_regex

            if center_hit or neighbor_hit:
                removed_rows += 1
                if center_hit:
                    removed_by_center += 1
                if neighbor_hit:
                    removed_by_neighbor += 1
                if center_hit and neighbor_hit:
                    removed_by_both += 1
                if center_exact or neighbor_exact:
                    removed_by_exact += 1
                if center_regex or neighbor_regex:
                    removed_by_regex += 1
                continue

            writer.writerow(row)
            kept_rows += 1

    log(f"Filtered: {input_path.name}")
    log(f"  input rows: {input_rows:,}")
    log(f"  removed rows: {removed_rows:,}")
    log(f"  kept rows: {kept_rows:,}")
    log(f"  output: {output_csv_path}")

    if summary_json:
        summary = {
            "source_file": str(input_path),
            "output_file": str(output_csv_path),
            "runtime_rule_files": {
                "exact_always": str(runtime_rules.exact_always_path),
                "exact_optional": str(runtime_rules.exact_optional_path) if runtime_rules.exact_optional_path else None,
                "regex_always": str(runtime_rules.regex_always_path),
            },
            "runtime_rule_versions": {
                "exact_always": runtime_rules.exact_always_version,
                "exact_optional": runtime_rules.exact_optional_version,
                "regex_always": runtime_rules.regex_always_version,
                "label": runtime_rules.version_label,
            },
            "stopword_profile": profile,
            "input_rows": input_rows,
            "removed_rows": removed_rows,
            "kept_rows": kept_rows,
            "removed_by_center": removed_by_center,
            "removed_by_neighbor": removed_by_neighbor,
            "removed_by_both": removed_by_both,
            "removed_by_exact": removed_by_exact,
            "removed_by_regex": removed_by_regex,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path = build_summary_path(output_csv_path)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"  summary: {summary_path}")


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    input_paths, rule_dir, profile, output_dir, write_summary = resolve_inputs(args)
    runtime_rules = resolve_runtime_rules(rule_dir, profile)

    exact_paths = [runtime_rules.exact_always_path]
    if runtime_rules.exact_optional_path is not None:
        exact_paths.append(runtime_rules.exact_optional_path)
    exact_stopwords, exact_loaded_rows = load_exact_stopwords(exact_paths)
    regex_rules, regex_loaded_rows = load_regex_stopwords(runtime_rules.regex_always_path)

    log(f"Runtime rule directory: {rule_dir}")
    log(f"Stopword profile: {profile}")
    log(f"Version label: {runtime_rules.version_label}")
    log(f"Exact always file: {runtime_rules.exact_always_path}")
    if runtime_rules.exact_optional_path:
        log(f"Exact optional file: {runtime_rules.exact_optional_path}")
    log(f"Regex always file: {runtime_rules.regex_always_path}")
    log(f"Loaded exact rows: {exact_loaded_rows:,}")
    log(f"Loaded regex rows: {regex_loaded_rows:,}")
    log(f"Exact stopwords: {len(exact_stopwords):,}")
    log(f"Regex stopwords: {len(regex_rules):,}")
    log(f"Input CSV files: {len(input_paths):,}")

    for input_path in input_paths:
        output_csv_path = build_output_path(
            input_path=input_path,
            output_dir=output_dir,
            profile=profile,
            version_label=runtime_rules.version_label,
        )
        filter_rows(
            input_path=input_path,
            output_csv_path=output_csv_path,
            summary_json=write_summary,
            exact_stopwords=exact_stopwords,
            regex_rules=regex_rules,
            runtime_rules=runtime_rules,
            profile=profile,
        )


if __name__ == "__main__":
    main()
