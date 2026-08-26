#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shenbao_applied_token_build.py

Build an analysis-oriented tokens.parquet from existing CKIP JSONL output without
rerunning CKIP. The script applies:

1. variant character normalization
2. token whitespace normalization
3. dictionary regex merges
4. dictionary exact merges
5. dictionary optional merges

The output parquet preserves the columns required by the downstream PMI pipeline and
adds lightweight audit columns for applied dictionary metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SHENBAO_DIR = ROOT / "shenbao"
DEFAULT_INPUT_JSONL = (
    SHENBAO_DIR / "shenbao_nlp" / "shenbao_nlp_ckip_bert-base-chinese_context_constitutional.jsonl"
)
DEFAULT_RULE_SPLIT_DIR = SHENBAO_DIR / "shenbao_dictionary" / "runtime_rule_splits"
DEFAULT_CROSSWALK_CSV = DEFAULT_RULE_SPLIT_DIR / "dictionary_annotation_crosswalk_v0.csv"
DEFAULT_APPLIED_TOKENS_DIR = SHENBAO_DIR / "shenbao_network" / "applied_tokens"


def find_latest_runtime_rule_file(rule_dir: Path, stem: str) -> Path:
    pattern = re.compile(rf"^{re.escape(stem)}_v(?P<version>\d+)\.csv$")
    matches: List[Tuple[int, Path]] = []
    for path in rule_dir.glob(f"{stem}_v*.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            matches.append((int(match.group("version")), path))
    if not matches:
        raise FileNotFoundError(f"No runtime rule file found for {stem}_v*.csv in {rule_dir}")
    return max(matches, key=lambda item: item[0])[1]


DEFAULT_VARIANT_CSV = find_latest_runtime_rule_file(DEFAULT_RULE_SPLIT_DIR, "variant_char_normalization_runtime")
DEFAULT_DICT_REGEX_CSV = find_latest_runtime_rule_file(DEFAULT_RULE_SPLIT_DIR, "dictionary_regex_merge")
DEFAULT_DICT_EXACT_CSV = find_latest_runtime_rule_file(DEFAULT_RULE_SPLIT_DIR, "dictionary_exact_merge")
DEFAULT_DICT_OPTIONAL_CSV = find_latest_runtime_rule_file(DEFAULT_RULE_SPLIT_DIR, "dictionary_exact_merge_optional")

BOUNDARY_POS = {"EXCLAMATIONCATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY"}
HAN_INTERNAL_SPACE_RE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef])"
    r"\s+"
    r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef])"
)
REPORT_FIELDNAMES = [
    "line_no",
    "context_uid",
    "article_uid",
    "dataset_index",
    "article_id",
    "date",
    "merge_stage",
    "dict_action",
    "dict_lv1",
    "dict_lv2",
    "dict_ner_like_type",
    "matched_string",
    "merged_token",
    "merged_from_tokens",
    "merged_from_pos",
    "start",
    "end",
    "global_start",
    "global_end",
]
TOKEN_SCHEMA = pa.schema(
    [
        ("token_uid", pa.string()),
        ("context_uid", pa.int64()),
        ("article_uid", pa.int64()),
        ("dataset_index", pa.int64()),
        ("article_id", pa.string()),
        ("date", pa.string()),
        ("token_order_in_context", pa.int64()),
        ("token", pa.string()),
        ("pos", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("global_start", pa.int64()),
        ("global_end", pa.int64()),
        ("char_len", pa.int64()),
        ("is_punctuation", pa.bool_()),
        ("is_boundary", pa.bool_()),
        ("token_source", pa.string()),
        ("pos_source", pa.string()),
        ("dict_lv1", pa.string()),
        ("dict_lv2", pa.string()),
        ("dict_ner_like_type", pa.string()),
        ("dict_match_type", pa.string()),
        ("merged_from_tokens", pa.string()),
        ("merged_from_pos", pa.string()),
    ]
)


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


def parse_on_off(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "on"}:
        return True
    if lowered in {"false", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected 'true' or 'false'.")


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[dict]:
    raise_csv_field_limit()

    def _read(encoding: str) -> Tuple[List[str], List[dict]]:
        with path.open("r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        return fieldnames, rows

    fieldnames, rows = _read("utf-8")
    if any((name or "").startswith("\ufeff") for name in fieldnames):
        _, rows = _read("utf-8-sig")
    return rows


def iter_jsonl(path: Path) -> Iterator[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


@dataclass(frozen=True)
class CrosswalkEntry:
    lv1: str
    lv2: str
    dict_ner_like_type: str
    default_pos: str


@dataclass(frozen=True)
class DictRule:
    normalized_string: str
    lv1: str
    lv2: str
    dict_action: str
    domain_tag: str
    dict_ner_like_type: str
    default_pos: str


@dataclass(frozen=True)
class RegexRule:
    normalized_pattern: str
    compiled: re.Pattern[str]
    lv1: str
    lv2: str
    dict_action: str
    domain_tag: str
    dict_ner_like_type: str
    default_pos: str


class TokenBatchWriter:
    def __init__(self, path: Path):
        ensure_parent_dir(path)
        self.path = path
        self.writer: Optional[pq.ParquetWriter] = None

    def write_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=TOKEN_SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, TOKEN_SCHEMA, compression="snappy")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def load_crosswalk(path: Path) -> Dict[Tuple[str, str], CrosswalkEntry]:
    rows = read_csv_rows(path)
    crosswalk: Dict[Tuple[str, str], CrosswalkEntry] = {}
    for row in rows:
        key = ((row.get("lv1") or "").strip(), (row.get("lv2") or "").strip())
        crosswalk[key] = CrosswalkEntry(
            lv1=key[0],
            lv2=key[1],
            dict_ner_like_type=(row.get("dict_ner_like_type") or "").strip(),
            default_pos=(row.get("default_pos") or "").strip(),
        )
    return crosswalk


def load_variant_rules(path: Path) -> Tuple[Dict[int, str], dict]:
    rows = read_csv_rows(path)
    translation: Dict[int, str] = {}
    malformed = 0
    for row in rows:
        replace_from = (row.get("replace_from") or "").strip()
        replace_to = (row.get("replace_to") or "").strip()
        if len(replace_from) != 1 or len(replace_to) != 1:
            malformed += 1
            continue
        translation[ord(replace_from)] = replace_to
    metadata = {
        "input_rows": len(rows),
        "applied_rules": len(translation),
        "skipped_malformed_rows": malformed,
    }
    return translation, metadata


def normalize_text(text: str, translation: Dict[int, str], enabled: bool) -> str:
    if not enabled or not translation:
        return text
    return text.translate(translation)


def load_exact_rules(
    path: Path,
    crosswalk: Dict[Tuple[str, str], CrosswalkEntry],
    translation: Dict[int, str],
    apply_variant: bool,
) -> Tuple[Dict[str, DictRule], dict]:
    rows = read_csv_rows(path)
    rule_map: Dict[str, DictRule] = {}
    duplicate_count = 0
    for row in rows:
        original_string = (row.get("string") or "").strip()
        if not original_string:
            continue
        normalized_string = normalize_text(original_string, translation, apply_variant)
        lv1 = (row.get("lv1") or "").strip()
        lv2 = (row.get("lv2") or "").strip()
        dict_action = (row.get("dict_action") or "").strip()
        domain_tag = (row.get("domain_tag") or "").strip()
        cw = crosswalk.get((lv1, lv2))
        rule = DictRule(
            normalized_string=normalized_string,
            lv1=lv1,
            lv2=lv2,
            dict_action=dict_action,
            domain_tag=domain_tag,
            dict_ner_like_type=cw.dict_ner_like_type if cw else "",
            default_pos=cw.default_pos if cw else "",
        )
        if normalized_string in rule_map:
            duplicate_count += 1
            continue
        rule_map[normalized_string] = rule
    metadata = {
        "input_rows": len(rows),
        "loaded_rules": len(rule_map),
        "skipped_duplicate_normalized_strings": duplicate_count,
    }
    return rule_map, metadata


def load_regex_rules(
    path: Path,
    crosswalk: Dict[Tuple[str, str], CrosswalkEntry],
    translation: Dict[int, str],
    apply_variant: bool,
) -> Tuple[List[RegexRule], dict]:
    rows = read_csv_rows(path)
    rules: List[RegexRule] = []
    compile_error_count = 0
    for row in rows:
        original_pattern = (row.get("string") or "").strip()
        if not original_pattern:
            continue
        normalized_pattern = normalize_text(original_pattern, translation, apply_variant)
        lv1 = (row.get("lv1") or "").strip()
        lv2 = (row.get("lv2") or "").strip()
        dict_action = (row.get("dict_action") or "").strip()
        domain_tag = (row.get("domain_tag") or "").strip()
        cw = crosswalk.get((lv1, lv2))
        try:
            compiled = re.compile(normalized_pattern)
        except re.error:
            compile_error_count += 1
            continue
        rules.append(
            RegexRule(
                normalized_pattern=normalized_pattern,
                compiled=compiled,
                lv1=lv1,
                lv2=lv2,
                dict_action=dict_action,
                domain_tag=domain_tag,
                dict_ner_like_type=cw.dict_ner_like_type if cw else "",
                default_pos=cw.default_pos if cw else "",
            )
        )
    metadata = {
        "input_rows": len(rows),
        "loaded_rules": len(rules),
        "skipped_compile_errors": compile_error_count,
    }
    return rules, metadata


def normalize_token_whitespace(
    text: str,
    start: int,
    end: int,
    global_start: int,
    global_end: int,
    enabled: bool,
) -> Tuple[str, int, int, int, int, bool]:
    if not enabled or not text:
        return text, start, end, global_start, global_end, False

    leading = len(text) - len(text.lstrip())
    trailing = len(text) - len(text.rstrip())
    stripped = text.strip()
    if not stripped:
        return text, start, end, global_start, global_end, False

    compacted = HAN_INTERNAL_SPACE_RE.sub("", stripped)
    if compacted == text:
        return text, start, end, global_start, global_end, False

    return (
        compacted,
        start + leading,
        end - trailing,
        global_start + leading,
        global_end - trailing,
        True,
    )


def make_internal_token(
    token: dict,
    translation: Dict[int, str],
    apply_variant: bool,
    normalize_whitespace: bool,
) -> Tuple[dict, bool, bool]:
    original_text = str(token.get("token", ""))
    normalized_text = normalize_text(original_text, translation, apply_variant)
    variant_changed = normalized_text != original_text
    start = int(token.get("start", 0))
    end = int(token.get("end", 0))
    global_start = int(token.get("global_start", 0))
    global_end = int(token.get("global_end", 0))
    normalized_text, start, end, global_start, global_end, whitespace_changed = normalize_token_whitespace(
        normalized_text,
        start,
        end,
        global_start,
        global_end,
        normalize_whitespace,
    )
    token_source = "original"
    if variant_changed and whitespace_changed:
        token_source = "variant_whitespace_normalized"
    elif variant_changed:
        token_source = "variant_only"
    elif whitespace_changed:
        token_source = "whitespace_normalized"
    return (
        {
            "token": normalized_text,
            "pos": str(token.get("pos", "")),
            "start": start,
            "end": end,
            "global_start": global_start,
            "global_end": global_end,
            "token_source": token_source,
            "pos_source": "ckip",
            "dict_lv1": "",
            "dict_lv2": "",
            "dict_ner_like_type": "",
            "dict_match_type": "",
            "merged_from_tokens": "",
            "merged_from_pos": "",
        },
        variant_changed,
        whitespace_changed,
    )


def choose_pos_for_merged_token(old_tokens: Sequence[dict], rule: DictRule | RegexRule) -> Tuple[str, str]:
    pos_values = [str(tok.get("pos", "")) for tok in old_tokens if str(tok.get("pos", ""))]
    if pos_values and len(set(pos_values)) == 1:
        return pos_values[0], "uniform_merged"
    if rule.default_pos:
        return rule.default_pos, "crosswalk_default"
    return "Na", "fallback_na"


def make_merged_internal_token(
    old_tokens: Sequence[dict],
    merged_text: str,
    stage: str,
    rule: DictRule | RegexRule,
) -> dict:
    pos_value, pos_source = choose_pos_for_merged_token(old_tokens, rule)
    return {
        "token": merged_text,
        "pos": pos_value,
        "start": int(old_tokens[0]["start"]),
        "end": int(old_tokens[-1]["end"]),
        "global_start": int(old_tokens[0]["global_start"]),
        "global_end": int(old_tokens[-1]["global_end"]),
        "token_source": stage,
        "pos_source": pos_source,
        "dict_lv1": rule.lv1,
        "dict_lv2": rule.lv2,
        "dict_ner_like_type": rule.dict_ner_like_type,
        "dict_match_type": stage,
        "merged_from_tokens": "|".join(str(tok["token"]) for tok in old_tokens),
        "merged_from_pos": "|".join(str(tok["pos"]) for tok in old_tokens),
    }


def build_boundary_maps(tokens: Sequence[dict]) -> Tuple[Dict[int, int], Dict[int, int]]:
    start_map: Dict[int, int] = {}
    end_map: Dict[int, int] = {}
    for index, token in enumerate(tokens):
        start_map[int(token["start"])] = index
        end_map[int(token["end"])] = index
    return start_map, end_map


def filter_regex_rules_by_action(rules: Sequence[RegexRule], dict_action: str) -> List[RegexRule]:
    return [rule for rule in rules if rule.dict_action == dict_action]


def apply_regex_merge_pass(
    tokens: List[dict],
    context_text: str,
    rules: Sequence[RegexRule],
    stage: str,
    line_no: int,
    row: dict,
    report_writer: Optional[csv.DictWriter],
) -> Tuple[List[dict], int]:
    if not tokens or not rules:
        return tokens, 0

    start_map, end_map = build_boundary_maps(tokens)
    candidates: List[dict] = []
    for rule in rules:
        for match in rule.compiled.finditer(context_text):
            start_char, end_char = match.span()
            if end_char <= start_char:
                continue
            start_i = start_map.get(start_char)
            end_i_inclusive = end_map.get(end_char)
            if start_i is None or end_i_inclusive is None:
                continue
            end_i_exclusive = end_i_inclusive + 1
            if end_i_exclusive - start_i < 2:
                continue
            candidates.append(
                {
                    "start_i": start_i,
                    "end_i": end_i_exclusive,
                    "start_char": start_char,
                    "end_char": end_char,
                    "matched_text": match.group(0),
                    "rule": rule,
                }
            )
    if not candidates:
        return tokens, 0

    candidates.sort(
        key=lambda item: (
            item["start_i"],
            -(item["end_i"] - item["start_i"]),
            -(item["end_char"] - item["start_char"]),
        )
    )
    accepted: List[dict] = []
    last_end = -1
    for candidate in candidates:
        if candidate["start_i"] < last_end:
            continue
        accepted.append(candidate)
        last_end = candidate["end_i"]

    if not accepted:
        return tokens, 0

    accepted_by_start = {item["start_i"]: item for item in accepted}
    new_tokens: List[dict] = []
    merge_count = 0
    i = 0
    while i < len(tokens):
        candidate = accepted_by_start.get(i)
        if candidate is None:
            new_tokens.append(dict(tokens[i]))
            i += 1
            continue
        old_span = tokens[candidate["start_i"] : candidate["end_i"]]
        merged_token = make_merged_internal_token(old_span, candidate["matched_text"], stage, candidate["rule"])
        new_tokens.append(merged_token)
        merge_count += 1
        if report_writer is not None:
            report_writer.writerow(
                {
                    "line_no": line_no,
                    "context_uid": row.get("nlp_index", ""),
                    "article_uid": row.get("dataset_index", ""),
                    "dataset_index": row.get("dataset_index", ""),
                    "article_id": row.get("article_id", ""),
                    "date": row.get("date", ""),
                    "merge_stage": stage,
                    "dict_action": candidate["rule"].dict_action,
                    "dict_lv1": candidate["rule"].lv1,
                    "dict_lv2": candidate["rule"].lv2,
                    "dict_ner_like_type": candidate["rule"].dict_ner_like_type,
                    "matched_string": candidate["rule"].normalized_pattern,
                    "merged_token": merged_token["token"],
                    "merged_from_tokens": merged_token["merged_from_tokens"],
                    "merged_from_pos": merged_token["merged_from_pos"],
                    "start": merged_token["start"],
                    "end": merged_token["end"],
                    "global_start": merged_token["global_start"],
                    "global_end": merged_token["global_end"],
                }
            )
        i = candidate["end_i"]
    return new_tokens, merge_count


def build_exact_rule_stats(rule_map: Dict[str, DictRule]) -> Tuple[set[int], int]:
    if not rule_map:
        return set(), 0
    lengths = {len(key) for key in rule_map}
    max_char_len = max(lengths)
    return lengths, max_char_len


def find_longest_exact_match(
    tokens: Sequence[dict],
    start_i: int,
    rule_map: Dict[str, DictRule],
    allowed_lengths: set[int],
    max_char_len: int,
    max_window_tokens: int,
) -> Optional[Tuple[int, DictRule, str]]:
    if not rule_map:
        return None
    concat = ""
    best: Optional[Tuple[int, DictRule, str]] = None
    max_j = min(len(tokens), start_i + max_window_tokens)
    for j in range(start_i, max_j):
        concat += str(tokens[j]["token"])
        concat_len = len(concat)
        if concat_len > max_char_len:
            break
        if concat_len in allowed_lengths and concat in rule_map:
            rule = rule_map[concat]
            best = (j + 1, rule, concat)
    return best


def apply_exact_merge_pass(
    tokens: List[dict],
    rule_map: Dict[str, DictRule],
    stage: str,
    line_no: int,
    row: dict,
    report_writer: Optional[csv.DictWriter],
    max_window_tokens: int,
) -> Tuple[List[dict], int]:
    if not tokens or not rule_map:
        return tokens, 0
    allowed_lengths, max_char_len = build_exact_rule_stats(rule_map)
    new_tokens: List[dict] = []
    merge_count = 0
    i = 0
    while i < len(tokens):
        match = find_longest_exact_match(tokens, i, rule_map, allowed_lengths, max_char_len, max_window_tokens)
        if match is None or match[0] <= i + 1:
            new_tokens.append(dict(tokens[i]))
            i += 1
            continue
        end_i, rule, matched_string = match
        old_span = tokens[i:end_i]
        merged_token = make_merged_internal_token(old_span, matched_string, stage, rule)
        new_tokens.append(merged_token)
        merge_count += 1
        if report_writer is not None:
            report_writer.writerow(
                {
                    "line_no": line_no,
                    "context_uid": row.get("nlp_index", ""),
                    "article_uid": row.get("dataset_index", ""),
                    "dataset_index": row.get("dataset_index", ""),
                    "article_id": row.get("article_id", ""),
                    "date": row.get("date", ""),
                    "merge_stage": stage,
                    "dict_action": rule.dict_action,
                    "dict_lv1": rule.lv1,
                    "dict_lv2": rule.lv2,
                    "dict_ner_like_type": rule.dict_ner_like_type,
                    "matched_string": rule.normalized_string,
                    "merged_token": merged_token["token"],
                    "merged_from_tokens": merged_token["merged_from_tokens"],
                    "merged_from_pos": merged_token["merged_from_pos"],
                    "start": merged_token["start"],
                    "end": merged_token["end"],
                    "global_start": merged_token["global_start"],
                    "global_end": merged_token["global_end"],
                }
            )
        i = end_i
    return new_tokens, merge_count


def annotate_single_token_matches(
    tokens: List[dict],
    exact_required_map: Dict[str, DictRule],
    exact_optional_map: Dict[str, DictRule],
    regex_required_rules: Sequence[RegexRule],
    regex_optional_rules: Sequence[RegexRule],
    apply_dictionary_exact: bool,
    apply_dictionary_optional: bool,
    apply_dictionary_regex: bool,
) -> Dict[str, int]:
    counts = {
        "exact_single": 0,
        "exact_optional_single": 0,
        "regex_single": 0,
        "regex_optional_single": 0,
    }
    for token in tokens:
        if token["dict_match_type"]:
            continue
        token_text = str(token["token"])
        if apply_dictionary_exact and token_text in exact_required_map:
            rule = exact_required_map[token_text]
            token["dict_lv1"] = rule.lv1
            token["dict_lv2"] = rule.lv2
            token["dict_ner_like_type"] = rule.dict_ner_like_type
            token["dict_match_type"] = "exact_single"
            counts["exact_single"] += 1
            continue
        if apply_dictionary_optional and token_text in exact_optional_map:
            rule = exact_optional_map[token_text]
            token["dict_lv1"] = rule.lv1
            token["dict_lv2"] = rule.lv2
            token["dict_ner_like_type"] = rule.dict_ner_like_type
            token["dict_match_type"] = "exact_optional_single"
            counts["exact_optional_single"] += 1
            continue
        if apply_dictionary_regex:
            matched_required = next((rule for rule in regex_required_rules if rule.compiled.fullmatch(token_text)), None)
            if matched_required is not None:
                token["dict_lv1"] = matched_required.lv1
                token["dict_lv2"] = matched_required.lv2
                token["dict_ner_like_type"] = matched_required.dict_ner_like_type
                token["dict_match_type"] = "regex_single"
                counts["regex_single"] += 1
                continue
            if apply_dictionary_optional:
                matched_optional = next(
                    (rule for rule in regex_optional_rules if rule.compiled.fullmatch(token_text)),
                    None,
                )
                if matched_optional is not None:
                    token["dict_lv1"] = matched_optional.lv1
                    token["dict_lv2"] = matched_optional.lv2
                    token["dict_ner_like_type"] = matched_optional.dict_ner_like_type
                    token["dict_match_type"] = "regex_optional_single"
                    counts["regex_optional_single"] += 1
                    continue
    return counts


def internal_tokens_to_parquet_rows(tokens: Sequence[dict], row: dict) -> List[dict]:
    context_uid = int(row["nlp_index"])
    dataset_index = int(row["dataset_index"])
    article_uid = dataset_index
    article_id = str(row.get("article_id", ""))
    article_date = str(row.get("date", ""))
    parquet_rows: List[dict] = []
    for token_order, token in enumerate(tokens):
        pos_value = str(token["pos"])
        start_value = int(token["start"])
        end_value = int(token["end"])
        parquet_rows.append(
            {
                "token_uid": f"{context_uid}:{token_order}",
                "context_uid": context_uid,
                "article_uid": article_uid,
                "dataset_index": dataset_index,
                "article_id": article_id,
                "date": article_date,
                "token_order_in_context": token_order,
                "token": str(token["token"]),
                "pos": pos_value,
                "start": start_value,
                "end": end_value,
                "global_start": int(token["global_start"]),
                "global_end": int(token["global_end"]),
                "char_len": end_value - start_value,
                "is_punctuation": pos_value.endswith("CATEGORY"),
                "is_boundary": pos_value in BOUNDARY_POS,
                "token_source": str(token["token_source"]),
                "pos_source": str(token["pos_source"]),
                "dict_lv1": str(token["dict_lv1"]),
                "dict_lv2": str(token["dict_lv2"]),
                "dict_ner_like_type": str(token["dict_ner_like_type"]),
                "dict_match_type": str(token["dict_match_type"]),
                "merged_from_tokens": str(token["merged_from_tokens"]),
                "merged_from_pos": str(token["merged_from_pos"]),
            }
        )
    return parquet_rows


def default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build applied tokens.parquet from existing CKIP JSONL.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, choose automatically: "
            "regex_only when regex-only rules are applied, "
            "full when optional dictionary rules are applied, "
            "otherwise strict."
        ),
    )
    parser.add_argument("--variant-csv", type=Path, default=DEFAULT_VARIANT_CSV)
    parser.add_argument("--dictionary-regex-csv", type=Path, default=DEFAULT_DICT_REGEX_CSV)
    parser.add_argument("--dictionary-exact-csv", type=Path, default=DEFAULT_DICT_EXACT_CSV)
    parser.add_argument("--dictionary-optional-csv", type=Path, default=DEFAULT_DICT_OPTIONAL_CSV)
    parser.add_argument("--crosswalk-csv", type=Path, default=DEFAULT_CROSSWALK_CSV)
    parser.add_argument("--model-key", default="ckip_bert_base_chinese")
    parser.add_argument("--apply-variant", type=parse_on_off, default=True)
    parser.add_argument("--apply-dictionary-regex", type=parse_on_off, default=True)
    parser.add_argument("--apply-dictionary-exact", type=parse_on_off, default=True)
    parser.add_argument("--apply-dictionary-optional", type=parse_on_off, default=False)
    parser.add_argument(
        "--normalize-token-whitespace",
        type=parse_on_off,
        default=True,
        help=(
            "Strip leading/trailing token whitespace and remove whitespace between "
            "adjacent Han characters before dictionary merging. Default: true."
        ),
    )
    parser.add_argument("--write-merge-report", type=parse_on_off, default=True)
    parser.add_argument("--token-batch-size", type=int, default=50000)
    parser.add_argument("--max-window-tokens", type=int, default=12)
    parser.add_argument("--progress-every-contexts", type=int, default=500)
    parser.add_argument("--processing-pause-every-contexts", type=int, default=500)
    parser.add_argument("--processing-pause-seconds", type=float, default=0.02)
    parser.add_argument("--write-pause-seconds", type=float, default=0.05)
    parser.add_argument("--max-contexts", type=int, default=None)
    return parser


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    if args.apply_dictionary_regex and not args.apply_dictionary_exact and not args.apply_dictionary_optional:
        return DEFAULT_APPLIED_TOKENS_DIR / "regex_only"
    if args.apply_dictionary_optional:
        return DEFAULT_APPLIED_TOKENS_DIR / "full"
    return DEFAULT_APPLIED_TOKENS_DIR / "strict"


def main() -> None:
    args = default_parser().parse_args()
    if args.apply_dictionary_optional and not args.apply_dictionary_exact:
        log("Warning: dictionary optional is on while dictionary exact is off. This is allowed, but not recommended.")

    output_dir: Path = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_parquet = output_dir / "tokens.parquet"
    merge_report_csv = output_dir / "merge_report.csv"
    metadata_json = output_dir / "build_metadata.json"

    log("Loading rule files.")
    crosswalk = load_crosswalk(args.crosswalk_csv)
    variant_translation, variant_meta = load_variant_rules(args.variant_csv)
    regex_rules_all, regex_meta = load_regex_rules(
        args.dictionary_regex_csv,
        crosswalk,
        variant_translation,
        args.apply_variant,
    )
    exact_required_map, exact_required_meta = load_exact_rules(
        args.dictionary_exact_csv,
        crosswalk,
        variant_translation,
        args.apply_variant,
    )
    exact_optional_map, exact_optional_meta = load_exact_rules(
        args.dictionary_optional_csv,
        crosswalk,
        variant_translation,
        args.apply_variant,
    )
    regex_required_rules = filter_regex_rules_by_action(regex_rules_all, "merge")
    regex_optional_rules = filter_regex_rules_by_action(regex_rules_all, "merge_optional")

    report_handle = None
    report_writer: Optional[csv.DictWriter] = None
    if args.write_merge_report:
        ensure_parent_dir(merge_report_csv)
        report_handle = merge_report_csv.open("w", encoding="utf-8", newline="")
        report_writer = csv.DictWriter(report_handle, fieldnames=REPORT_FIELDNAMES)
        report_writer.writeheader()

    token_writer = TokenBatchWriter(tokens_parquet)
    token_buffer: List[dict] = []

    metadata = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "variant_csv": str(args.variant_csv),
        "dictionary_regex_csv": str(args.dictionary_regex_csv),
        "dictionary_exact_csv": str(args.dictionary_exact_csv),
        "dictionary_optional_csv": str(args.dictionary_optional_csv),
        "crosswalk_csv": str(args.crosswalk_csv),
        "model_key": args.model_key,
        "apply_variant": args.apply_variant,
        "apply_dictionary_regex": args.apply_dictionary_regex,
        "apply_dictionary_exact": args.apply_dictionary_exact,
        "apply_dictionary_optional": args.apply_dictionary_optional,
        "normalize_token_whitespace": args.normalize_token_whitespace,
        "variant_rule_stats": variant_meta,
        "regex_rule_stats": regex_meta,
        "exact_required_rule_stats": exact_required_meta,
        "exact_optional_rule_stats": exact_optional_meta,
        "contexts_seen": 0,
        "contexts_processed": 0,
        "contexts_skipped_non_success": 0,
        "contexts_skipped_missing_model_key": 0,
        "tokens_output": 0,
        "variant_changed_tokens": 0,
        "whitespace_normalized_tokens": 0,
        "regex_merge_count": 0,
        "regex_optional_merge_count": 0,
        "exact_merge_count": 0,
        "exact_optional_merge_count": 0,
        "exact_single_annotation_count": 0,
        "exact_optional_single_annotation_count": 0,
        "regex_single_annotation_count": 0,
        "regex_optional_single_annotation_count": 0,
    }

    try:
        for line_no, row in iter_jsonl(args.input_jsonl):
            if args.max_contexts is not None and metadata["contexts_seen"] >= args.max_contexts:
                break
            metadata["contexts_seen"] += 1

            if row.get("status") not in ("", None, "success"):
                metadata["contexts_skipped_non_success"] += 1
                continue
            model_obj = row.get(args.model_key)
            if not isinstance(model_obj, dict):
                metadata["contexts_skipped_missing_model_key"] += 1
                continue
            raw_tokens = model_obj.get("tokens")
            if not isinstance(raw_tokens, list):
                metadata["contexts_skipped_missing_model_key"] += 1
                continue

            context_text = str(row.get("context_text", ""))
            normalized_context_text = normalize_text(context_text, variant_translation, args.apply_variant)
            working_tokens: List[dict] = []
            for raw_token in raw_tokens:
                new_token, changed, whitespace_changed = make_internal_token(
                    raw_token,
                    variant_translation,
                    args.apply_variant,
                    args.normalize_token_whitespace,
                )
                if changed:
                    metadata["variant_changed_tokens"] += 1
                if whitespace_changed:
                    metadata["whitespace_normalized_tokens"] += 1
                working_tokens.append(new_token)

            if args.apply_dictionary_regex:
                working_tokens, merge_count = apply_regex_merge_pass(
                    working_tokens,
                    normalized_context_text,
                    regex_required_rules,
                    "regex_merge",
                    line_no,
                    row,
                    report_writer,
                )
                metadata["regex_merge_count"] += merge_count
                if args.apply_dictionary_optional:
                    working_tokens, merge_count = apply_regex_merge_pass(
                        working_tokens,
                        normalized_context_text,
                        regex_optional_rules,
                        "regex_optional_merge",
                        line_no,
                        row,
                        report_writer,
                    )
                    metadata["regex_optional_merge_count"] += merge_count

            if args.apply_dictionary_exact:
                working_tokens, merge_count = apply_exact_merge_pass(
                    working_tokens,
                    exact_required_map,
                    "exact_merge",
                    line_no,
                    row,
                    report_writer,
                    args.max_window_tokens,
                )
                metadata["exact_merge_count"] += merge_count

            if args.apply_dictionary_optional:
                working_tokens, merge_count = apply_exact_merge_pass(
                    working_tokens,
                    exact_optional_map,
                    "exact_optional_merge",
                    line_no,
                    row,
                    report_writer,
                    args.max_window_tokens,
                )
                metadata["exact_optional_merge_count"] += merge_count

            annotation_counts = annotate_single_token_matches(
                working_tokens,
                exact_required_map,
                exact_optional_map,
                regex_required_rules,
                regex_optional_rules,
                args.apply_dictionary_exact,
                args.apply_dictionary_optional,
                args.apply_dictionary_regex,
            )
            metadata["exact_single_annotation_count"] += annotation_counts["exact_single"]
            metadata["exact_optional_single_annotation_count"] += annotation_counts["exact_optional_single"]
            metadata["regex_single_annotation_count"] += annotation_counts["regex_single"]
            metadata["regex_optional_single_annotation_count"] += annotation_counts["regex_optional_single"]

            parquet_rows = internal_tokens_to_parquet_rows(working_tokens, row)
            token_buffer.extend(parquet_rows)
            metadata["tokens_output"] += len(parquet_rows)
            metadata["contexts_processed"] += 1

            if len(token_buffer) >= args.token_batch_size:
                token_writer.write_rows(token_buffer)
                token_buffer.clear()
                sleep_if_needed(args.write_pause_seconds)

            if (
                args.processing_pause_every_contexts > 0
                and metadata["contexts_seen"] % args.processing_pause_every_contexts == 0
            ):
                sleep_if_needed(args.processing_pause_seconds)

            if args.progress_every_contexts > 0 and metadata["contexts_seen"] % args.progress_every_contexts == 0:
                log(
                    f"Processed {metadata['contexts_seen']} contexts; "
                    f"output tokens={metadata['tokens_output']}, "
                    f"regex merges={metadata['regex_merge_count'] + metadata['regex_optional_merge_count']}, "
                    f"exact merges={metadata['exact_merge_count'] + metadata['exact_optional_merge_count']}"
                )

        if token_buffer:
            token_writer.write_rows(token_buffer)
            token_buffer.clear()
            sleep_if_needed(args.write_pause_seconds)
    finally:
        token_writer.close()
        if report_handle is not None:
            report_handle.close()

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log(
        "Done. "
        f"contexts_processed={metadata['contexts_processed']}, "
        f"tokens_output={metadata['tokens_output']}, "
        f"variant_changed_tokens={metadata['variant_changed_tokens']}"
    )


if __name__ == "__main__":
    main()
