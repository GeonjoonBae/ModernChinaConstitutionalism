#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply the 2026-08-17 制憲 context-filter revision and patch affected networks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from shenbao_network_undirected_pmi import convert_file as convert_undirected_file


ROOT = Path(__file__).resolve().parent
SHENBAO = ROOT / "shenbao"
FILTER_DIR = SHENBAO / "shenbao_filters" / "context"
FILTER_CSV = FILTER_DIR / "filter_context_pre_zhixian.csv"
FILTER_METADATA = FILTER_DIR / "filter_context_pre_zhixian_metadata.json"
CONTEXT_JSONL = SHENBAO / "shenbao_nlp" / "shenbao_nlp_ckip_bert-base-chinese_context_constitutional.jsonl"
TOKENS_ROOT = SHENBAO / "shenbao_network" / "applied_tokens"
NETWORK_ROOT = SHENBAO / "shenbao_network" / "network_applied"
REPORT_DIR = FILTER_DIR / "pre_zhixian_revision_20260817"
NETWORK_MARKER = REPORT_DIR / "network_patch_report.json"
TOKEN_FILTER_DIR = SHENBAO / "shenbao_filters" / "token"
TOKEN_MASK_CSV = TOKEN_FILTER_DIR / "filter_token_pre_zhixian_retained_official_title.csv"
TOKEN_MASK_METADATA = TOKEN_FILTER_DIR / "filter_token_pre_zhixian_retained_official_title_metadata.json"
STOPWORD_SCRIPT = ROOT / "shenbao_stopword_postfilter_v3.py"
RUNTIME_RULE_DIR = SHENBAO / "shenbao_dictionary" / "runtime_rule_splits"

ADDED_CONTEXT_UIDS = {
    1646,
    1647,
    3156,
    6465,
    6518,
    6519,
    7725,
    8081,
    8598,
    8599,
    8611,
    8647,
    8648,
    9498,
    10144,
    10475,
    10512,
    10603,
    10612,
    11440,
    11901,
}
RETAINED_PRE1913_CONTEXT_UIDS = {9499, 10143, 10904}
RETAINED_POST1912_EXCLUSIONS = {26955, 32404}
TOKEN_MASK_CONTEXT_UIDS = RETAINED_PRE1913_CONTEXT_UIDS
TOKEN_MASK_TEXT = "制憲"
NETWORK_PATCH_VERSION = 2
CORE_TOKENS = ("立憲", "憲政", "憲法", "制憲")
PROFILES = ("regex_only", "strict", "full")
WINDOWS = (1, 5, 10, 20)
PROFILE_LABEL = {"regex_only": "regex-only", "strict": "strict", "full": "full"}

PairKey = Tuple[str, str, str]
TokenKey = Tuple[str, str]


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_context_jsonl_rows(context_uids: Set[int]) -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    remaining = set(context_uids)
    with CONTEXT_JSONL.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not remaining:
                break
            if line_number not in remaining:
                continue
            obj = json.loads(line)
            context_uid = int(obj.get("nlp_index", line_number))
            if context_uid in remaining:
                rows[context_uid] = obj
                remaining.remove(context_uid)
    if remaining:
        raise ValueError(f"Context JSONL rows not found: {sorted(remaining)}")
    return rows


def derive_filter_rule() -> Tuple[Set[int], Set[int], Set[int]]:
    tokens = pd.read_parquet(
        TOKENS_ROOT / "regex_only" / "tokens.parquet",
        columns=["context_uid", "date", "token"],
    )
    tokens["context_uid"] = tokens["context_uid"].astype(int)
    tokens["date"] = tokens["date"].astype(str)
    tokens["token"] = tokens["token"].astype(str)
    pre1913 = tokens[tokens["date"].le("1912-12-31")]
    zhixian_contexts = set(
        pre1913.loc[
            pre1913["token"].str.contains("制憲", regex=False), "context_uid"
        ].tolist()
    )
    companion_contexts = set(
        pre1913.loc[
            pre1913["token"].isin(("立憲", "憲政")), "context_uid"
        ].tolist()
    )
    retained = zhixian_contexts & companion_contexts
    if retained != RETAINED_PRE1913_CONTEXT_UIDS:
        raise ValueError(
            "Unexpected retained pre-1913 contexts: "
            f"expected={sorted(RETAINED_PRE1913_CONTEXT_UIDS)}, actual={sorted(retained)}"
        )
    pre1913_exclusions = zhixian_contexts - retained
    desired = pre1913_exclusions | RETAINED_POST1912_EXCLUSIONS
    return desired, pre1913_exclusions, retained


def apply_filter_revision() -> Set[int]:
    backup_csv = FILTER_CSV.with_name("filter_context_pre_zhixian.before_20260817.csv")
    backup_metadata = FILTER_METADATA.with_name("filter_context_pre_zhixian_metadata.before_20260817.json")
    if not backup_csv.exists():
        shutil.copy2(FILTER_CSV, backup_csv)
    if FILTER_METADATA.exists() and not backup_metadata.exists():
        shutil.copy2(FILTER_METADATA, backup_metadata)

    desired_uids, pre1913_exclusions, retained = derive_filter_rule()
    fieldnames, rows = read_csv_rows(FILTER_CSV)
    existing = {int(row["context_uid"]): row for row in rows if str(row.get("context_uid", "")).strip()}
    missing = desired_uids - set(existing)
    removed = set(existing) - desired_uids
    context_rows = load_context_jsonl_rows(missing) if missing else {}

    for context_uid in sorted(missing):
        obj = context_rows[context_uid]
        existing[context_uid] = {
            "context_uid": context_uid,
            "article_id": obj.get("article_id", ""),
            "date": obj.get("date", ""),
            "context_char_len": obj.get("context_char_len", ""),
            "keyword_occurrence_count": obj.get("keyword_occurrence_count", ""),
            "context_text": obj.get("context_text", ""),
        }

    ordered_rows = []
    for context_uid in sorted(desired_uids):
        row = existing[context_uid]
        reason = (
            "rule_pre1913_no_core_companion"
            if context_uid in pre1913_exclusions
            else "manual_exclude"
        )
        row.update(
            {
                "class_label": "negative",
                "predicted_class_id": "",
                "predicted_period_label": reason,
                "prob_positive": "",
                "prob_negative": "",
                "decision_score": "",
                "actual_period_label": reason,
                "exclude_reason": reason,
            }
        )
        ordered_rows.append(row)
    write_csv_rows(FILTER_CSV, fieldnames, ordered_rows)
    excluded_uids = set(desired_uids)

    metadata = json.loads(FILTER_METADATA.read_text(encoding="utf-8-sig")) if FILTER_METADATA.exists() else {}
    reason_counts = Counter(str(row.get("exclude_reason", "")) for row in ordered_rows)
    metadata.update(
        {
            "filter_name": "filter_context_pre_zhixian",
            "filter_basis": "context",
            "filter_reason": "pre_zhixian",
            "filename_suffix": "filtered_pre_zhixian_context",
            "action": "exclude_context_uid",
            "source_csv": str(backup_csv.relative_to(ROOT)),
            "source_context_jsonl": str(CONTEXT_JSONL.relative_to(ROOT)),
            "source_tokens_parquet": str(
                (TOKENS_ROOT / "regex_only" / "tokens.parquet").relative_to(ROOT)
            ),
            "generator_script": str(Path(__file__).resolve().relative_to(ROOT)),
            "filter_csv": str(FILTER_CSV.relative_to(ROOT)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "row_count": len(ordered_rows),
            "excluded_context_uid_count": len(excluded_uids),
            "exclude_reason_counts": dict(sorted(reason_counts.items())),
            "revision_20260817": {
                "rule": (
                    "Exclude contexts through 1912 containing a token that contains 制憲, "
                    "except contexts with exact-token 立憲 or 憲政; retain post-1912 manual "
                    "official-title exclusions 26955 and 32404."
                ),
                "added_context_uids": sorted(ADDED_CONTEXT_UIDS),
                "derived_pre1913_exclusion_count": len(pre1913_exclusions),
                "retained_pre1913_context_uids": sorted(retained),
                "retained_post1912_exclusions": sorted(RETAINED_POST1912_EXCLUSIONS),
                "retained_context_token_mask": {
                    "filter_csv": str(TOKEN_MASK_CSV.relative_to(ROOT)),
                    "context_uids": sorted(TOKEN_MASK_CONTEXT_UIDS),
                    "token_match": TOKEN_MASK_TEXT,
                    "match_mode": "contains",
                    "action": "exclude_matching_token_rows_only",
                },
            },
        }
    )
    FILTER_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[filter] excluded context_uids={len(excluded_uids):,}; "
        f"added={len(missing):,}; removed={len(removed):,}"
    )
    return excluded_uids


def token_mask_series(tokens: pd.DataFrame) -> pd.Series:
    """Return rows whose 制憲-containing token is an official-title use to exclude."""
    context_uids = pd.to_numeric(tokens["context_uid"], errors="coerce")
    token_values = tokens["token"].fillna("").astype(str)
    return context_uids.isin(TOKEN_MASK_CONTEXT_UIDS) & token_values.str.contains(
        TOKEN_MASK_TEXT, regex=False, na=False
    )


def write_token_mask() -> Path:
    rows = [
        {
            "context_uid": context_uid,
            "token_match": TOKEN_MASK_TEXT,
            "match_mode": "contains",
            "exclude_reason": "official_title_not_constitution_making",
        }
        for context_uid in sorted(TOKEN_MASK_CONTEXT_UIDS)
    ]
    write_csv_rows(
        TOKEN_MASK_CSV,
        ["context_uid", "token_match", "match_mode", "exclude_reason"],
        rows,
    )

    matched_by_profile: Dict[str, List[Dict[str, object]]] = {}
    for profile in PROFILES:
        tokens = read_selected_tokens(profile, context_uids=TOKEN_MASK_CONTEXT_UIDS)
        matched = tokens.loc[token_mask_series(tokens), ["context_uid", "token"]].copy()
        matched["context_uid"] = pd.to_numeric(matched["context_uid"], errors="raise").astype(int)
        matched_by_profile[PROFILE_LABEL[profile]] = (
            matched.value_counts(["context_uid", "token"])
            .rename("matched_token_count")
            .reset_index()
            .sort_values(["context_uid", "token"], kind="mergesort")
            .to_dict("records")
        )

    metadata = {
        "filter_name": TOKEN_MASK_CSV.stem,
        "filter_basis": "token_in_retained_context",
        "action": "exclude_matching_token_rows_only",
        "rule": (
            "In retained pre-1913 contexts 9499, 10143, and 10904, exclude tokens "
            "containing 制憲 because they denote an official title rather than constitution-making. "
            "Keep the contexts and all other tokens."
        ),
        "filter_csv": str(TOKEN_MASK_CSV.relative_to(ROOT)),
        "generator_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "context_uids": sorted(TOKEN_MASK_CONTEXT_UIDS),
        "matched_tokens_by_profile": matched_by_profile,
    }
    TOKEN_MASK_METADATA.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_MASK_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[token mask] {TOKEN_MASK_CSV}")
    return TOKEN_MASK_CSV


def write_period_counts(excluded_uids: Set[int]) -> Path:
    tokens_path = TOKENS_ROOT / "regex_only" / "tokens.parquet"
    periods_path = TOKENS_ROOT / "regex_only" / "periods" / "periods.parquet"
    tokens = pd.read_parquet(
        tokens_path,
        columns=["date", "article_uid", "context_uid", "token"],
    )
    tokens["date"] = tokens["date"].astype(str)
    tokens = tokens[~tokens["context_uid"].astype(int).isin(excluded_uids)].copy()
    tokens = tokens.loc[~token_mask_series(tokens)].copy()
    periods = pd.read_parquet(periods_path)
    periods = periods[periods["period_set_id"].astype(str).eq("long_period_manual")].copy()
    periods = periods.sort_values(["sort_order", "period_id"], kind="mergesort")

    rows: List[Dict[str, object]] = []
    for period in periods.itertuples(index=False):
        subset = tokens[
            tokens["date"].between(str(period.start_date), str(period.end_date), inclusive="both")
        ]
        row: Dict[str, object] = {
            "period_id": str(period.period_id),
            "start_date": str(period.start_date),
            "end_date": str(period.end_date),
            "article_count": int(subset["article_uid"].nunique()),
            "context_count": int(subset["context_uid"].nunique()),
        }
        exact = subset[subset["token"].astype(str).isin(CORE_TOKENS)]
        counts = exact["token"].astype(str).value_counts().to_dict()
        for token in CORE_TOKENS:
            row[token] = int(counts.get(token, 0))
        rows.append(row)

    total = {
        "period_id": "total",
        "start_date": "",
        "end_date": "",
        "article_count": int(tokens["article_uid"].nunique()),
        "context_count": int(tokens["context_uid"].nunique()),
    }
    exact_all = tokens[tokens["token"].astype(str).isin(CORE_TOKENS)]["token"].astype(str).value_counts()
    for token in CORE_TOKENS:
        total[token] = int(exact_all.get(token, 0))
    rows.append(total)

    output = REPORT_DIR / "long_period_manual_counts_after_revision.csv"
    write_csv_rows(
        output,
        ["period_id", "start_date", "end_date", "article_count", "context_count", *CORE_TOKENS],
        rows,
    )
    print(f"[counts] {output}")
    return output


def read_selected_tokens(profile: str, *, context_uids: Set[int] | None = None, article_uids: Set[int] | None = None) -> pd.DataFrame:
    filters = None
    if context_uids is not None:
        filters = [("context_uid", "in", sorted(context_uids))]
    elif article_uids is not None:
        filters = [("article_uid", "in", sorted(article_uids))]
    return pd.read_parquet(
        TOKENS_ROOT / profile / "tokens.parquet",
        columns=["context_uid", "article_uid", "token_order_in_context", "token", "pos", "is_punctuation"],
        filters=filters,
    )


def build_event_stats(tokens: pd.DataFrame, window: int) -> Dict[str, object]:
    pair_count: Counter[PairKey] = Counter()
    center_count: Counter[TokenKey] = Counter()
    neighbor_count: Counter[TokenKey] = Counter()
    event_total: Counter[str] = Counter()
    distance_sum: Counter[PairKey] = Counter()
    pair_contexts: DefaultDict[PairKey, Set[int]] = defaultdict(set)
    pair_articles: DefaultDict[PairKey, Set[int]] = defaultdict(set)
    pair_pos: DefaultDict[PairKey, Counter[str]] = defaultdict(Counter)

    ordered = tokens.sort_values(["context_uid", "token_order_in_context"], kind="mergesort")
    for context_uid, frame in ordered.groupby("context_uid", sort=False):
        frame = frame[~frame["is_punctuation"].astype(bool)]
        records = list(frame[["article_uid", "token", "pos"]].itertuples(index=False, name=None))
        if not records:
            continue
        article_uid = int(records[0][0])
        for left_index, (_article, left_token, left_pos) in enumerate(records):
            right_limit = min(len(records), left_index + window + 1)
            for right_index in range(left_index + 1, right_limit):
                _right_article, right_token, right_pos = records[right_index]
                distance = right_index - left_index
                pos_pair = f"{left_pos}|{right_pos}"
                for direction, center_token, neighbor_token in (
                    ("R", str(left_token), str(right_token)),
                    ("L", str(right_token), str(left_token)),
                ):
                    pair = (direction, center_token, neighbor_token)
                    pair_count[pair] += 1
                    center_count[(direction, center_token)] += 1
                    neighbor_count[(direction, neighbor_token)] += 1
                    event_total[direction] += 1
                    distance_sum[pair] += distance
                    pair_contexts[pair].add(int(context_uid))
                    pair_articles[pair].add(article_uid)
                    pair_pos[pair][pos_pair] += 1
    return {
        "pair_count": pair_count,
        "center_count": center_count,
        "neighbor_count": neighbor_count,
        "event_total": event_total,
        "distance_sum": distance_sum,
        "pair_contexts": pair_contexts,
        "pair_articles": pair_articles,
        "pair_pos": pair_pos,
    }


def build_masked_event_stats(tokens: pd.DataFrame, window: int) -> Dict[str, object]:
    """Build removal statistics only for co-occurrences touching a masked token row."""
    pair_count: Counter[PairKey] = Counter()
    center_count: Counter[TokenKey] = Counter()
    neighbor_count: Counter[TokenKey] = Counter()
    event_total: Counter[str] = Counter()
    distance_sum: Counter[PairKey] = Counter()
    pair_contexts: DefaultDict[PairKey, Set[int]] = defaultdict(set)
    pair_articles: DefaultDict[PairKey, Set[int]] = defaultdict(set)
    pair_pos: DefaultDict[PairKey, Counter[str]] = defaultdict(Counter)

    ordered = tokens.sort_values(["context_uid", "token_order_in_context"], kind="mergesort").copy()
    ordered["is_token_masked"] = token_mask_series(ordered)
    for context_uid, frame in ordered.groupby("context_uid", sort=False):
        frame = frame[~frame["is_punctuation"].astype(bool)]
        records = list(
            frame[["article_uid", "token", "pos", "is_token_masked"]].itertuples(
                index=False, name=None
            )
        )
        if not records:
            continue
        article_uid = int(records[0][0])
        for left_index, (_article, left_token, left_pos, left_masked) in enumerate(records):
            right_limit = min(len(records), left_index + window + 1)
            for right_index in range(left_index + 1, right_limit):
                _right_article, right_token, right_pos, right_masked = records[right_index]
                if not (bool(left_masked) or bool(right_masked)):
                    continue
                distance = right_index - left_index
                pos_pair = f"{left_pos}|{right_pos}"
                for direction, center_token, neighbor_token in (
                    ("R", str(left_token), str(right_token)),
                    ("L", str(right_token), str(left_token)),
                ):
                    pair = (direction, center_token, neighbor_token)
                    pair_count[pair] += 1
                    center_count[(direction, center_token)] += 1
                    neighbor_count[(direction, neighbor_token)] += 1
                    event_total[direction] += 1
                    distance_sum[pair] += distance
                    pair_contexts[pair].add(int(context_uid))
                    pair_articles[pair].add(article_uid)
                    pair_pos[pair][pos_pair] += 1
    return {
        "pair_count": pair_count,
        "center_count": center_count,
        "neighbor_count": neighbor_count,
        "event_total": event_total,
        "distance_sum": distance_sum,
        "pair_contexts": pair_contexts,
        "pair_articles": pair_articles,
        "pair_pos": pair_pos,
    }


def merge_event_stats(*items: Dict[str, object]) -> Dict[str, object]:
    merged: Dict[str, object] = {
        "pair_count": Counter(),
        "center_count": Counter(),
        "neighbor_count": Counter(),
        "event_total": Counter(),
        "distance_sum": Counter(),
        "pair_contexts": defaultdict(set),
        "pair_articles": defaultdict(set),
        "pair_pos": defaultdict(Counter),
    }
    for stats in items:
        for key in ("pair_count", "center_count", "neighbor_count", "event_total", "distance_sum"):
            merged[key].update(stats[key])  # type: ignore[union-attr]
        for key in ("pair_contexts", "pair_articles"):
            target = merged[key]
            for pair, values in stats[key].items():  # type: ignore[union-attr]
                target[pair].update(values)  # type: ignore[index]
        target_pos = merged["pair_pos"]
        for pair, values in stats["pair_pos"].items():  # type: ignore[union-attr]
            target_pos[pair].update(values)  # type: ignore[index]
    return merged


def subtract_pos_json(raw: object, removed: Counter[str]) -> str:
    try:
        current = Counter({str(key): int(value) for key, value in json.loads(str(raw or "{}")).items()})
    except (json.JSONDecodeError, TypeError, ValueError):
        current = Counter()
    current.subtract(removed)
    cleaned = {key: value for key, value in current.items() if value > 0}
    return json.dumps(dict(sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))), ensure_ascii=False)


def parse_profile_window_threshold(path: Path) -> Tuple[str, int, int]:
    profile_match = re.search(r"_applied_(regex-only|strict|full)_", path.name)
    window_match = re.search(r"_w(\d+)_", path.name)
    threshold_match = re.search(r"_joint(\d+)up_", path.name)
    if not profile_match or not window_match or not threshold_match:
        raise ValueError(f"Cannot parse network condition: {path.name}")
    profile = "regex_only" if profile_match.group(1) == "regex-only" else profile_match.group(1)
    return profile, int(window_match.group(1)), int(threshold_match.group(1))


def patch_network_csv(
    path: Path,
    stats: Dict[str, object],
    retained_article_pairs: DefaultDict[PairKey, Set[int]],
    threshold: int,
) -> Dict[str, object]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    before_rows = len(frame)
    keys: List[PairKey] = list(
        zip(
            frame["direction"].astype(str),
            frame["center_token"].astype(str),
            frame["neighbor_token"].astype(str),
        )
    )
    pair_count: Counter[PairKey] = stats["pair_count"]  # type: ignore[assignment]
    center_count: Counter[TokenKey] = stats["center_count"]  # type: ignore[assignment]
    neighbor_count: Counter[TokenKey] = stats["neighbor_count"]  # type: ignore[assignment]
    event_total: Counter[str] = stats["event_total"]  # type: ignore[assignment]
    distance_sum: Counter[PairKey] = stats["distance_sum"]  # type: ignore[assignment]
    pair_contexts: DefaultDict[PairKey, Set[int]] = stats["pair_contexts"]  # type: ignore[assignment]
    pair_articles: DefaultDict[PairKey, Set[int]] = stats["pair_articles"]  # type: ignore[assignment]
    pair_pos: DefaultDict[PairKey, Counter[str]] = stats["pair_pos"]  # type: ignore[assignment]

    removed_pair = pd.Series([pair_count.get(key, 0) for key in keys], index=frame.index, dtype="int64")
    old_raw = pd.to_numeric(frame["raw_joint_event_count"], errors="raise").astype("int64")
    old_joint = pd.to_numeric(frame["joint_count"], errors="raise").astype(float)
    new_raw = old_raw - removed_pair
    new_joint = old_joint - removed_pair.astype(float)

    directions = frame["direction"].astype(str)
    centers = frame["center_token"].astype(str)
    neighbors = frame["neighbor_token"].astype(str)
    center_removed = pd.Series(
        [center_count.get((direction, token), 0) for direction, token in zip(directions, centers)],
        index=frame.index,
    )
    neighbor_removed = pd.Series(
        [neighbor_count.get((direction, token), 0) for direction, token in zip(directions, neighbors)],
        index=frame.index,
    )
    total_removed = directions.map(event_total).fillna(0).astype(int)

    frame["joint_count"] = new_joint
    frame["raw_joint_event_count"] = new_raw
    frame["center_marginal_count"] = pd.to_numeric(frame["center_marginal_count"], errors="raise") - center_removed
    frame["neighbor_marginal_count"] = pd.to_numeric(frame["neighbor_marginal_count"], errors="raise") - neighbor_removed
    frame["event_total"] = pd.to_numeric(frame["event_total"], errors="raise") - total_removed
    frame["raw_event_total"] = pd.to_numeric(frame["raw_event_total"], errors="raise").astype("int64") - total_removed
    frame["center_raw_count"] = pd.to_numeric(frame["center_raw_count"], errors="raise").astype("int64") - center_removed
    frame["neighbor_raw_count"] = pd.to_numeric(frame["neighbor_raw_count"], errors="raise").astype("int64") - neighbor_removed

    context_removed = pd.Series([len(pair_contexts.get(key, set())) for key in keys], index=frame.index)
    article_removed = pd.Series(
        [len(pair_articles.get(key, set()) - retained_article_pairs.get(key, set())) for key in keys],
        index=frame.index,
    )
    frame["distinct_context_count"] = (
        pd.to_numeric(frame["distinct_context_count"], errors="coerce").fillna(0).astype("int64") - context_removed
    )
    frame["distinct_article_count"] = (
        pd.to_numeric(frame["distinct_article_count"], errors="coerce").fillna(0).astype("int64") - article_removed
    )

    changed_indexes = frame.index[removed_pair.gt(0)]
    for index in changed_indexes:
        key = keys[index]
        if new_raw.at[index] > 0:
            old_distance_sum = float(frame.at[index, "avg_distance"]) * int(old_raw.at[index])
            frame.at[index, "avg_distance"] = (
                old_distance_sum - float(distance_sum.get(key, 0))
            ) / float(new_raw.at[index])
        frame.at[index, "pos_pair_json"] = subtract_pos_json(frame.at[index, "pos_pair_json"], pair_pos[key])

    valid = (
        frame["joint_count"].ge(float(threshold))
        & frame["center_marginal_count"].gt(0)
        & frame["neighbor_marginal_count"].gt(0)
        & frame["event_total"].gt(0)
    )
    frame = frame.loc[valid].copy()
    frame["pmi"] = frame.apply(
        lambda row: math.log2(
            (float(row["joint_count"]) * float(row["event_total"]))
            / (float(row["center_marginal_count"]) * float(row["neighbor_marginal_count"]))
        ),
        axis=1,
    )
    frame["ppmi"] = frame["pmi"].clip(lower=0.0)

    temp_path = path.with_suffix(".revision_tmp.csv")
    frame.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)
    return {
        "file": str(path),
        "rows_before": before_rows,
        "rows_after": len(frame),
        "rows_with_removed_events": int(removed_pair.gt(0).sum()),
        "rows_dropped_below_threshold": int(before_rows - len(frame)),
    }


def selected_directed_files() -> List[Path]:
    directories = [
        NETWORK_ROOT / "stopfiltered_global_filtered_pre_zhixian_context",
        NETWORK_ROOT / "stopfiltered_long_period_manual_p001_filtered_pre_zhixian_context",
    ]
    files: List[Path] = []
    for directory in directories:
        files.extend(
            sorted(
                path
                for path in directory.glob("network_constitutional_applied_*_all-tokens_*.csv")
                if path.is_file() and ".revision_tmp." not in path.name
            )
        )
    return files


def restore_directed_files_from_jointup() -> None:
    targets = [
        (
            NETWORK_ROOT / "jointup" / "global_filtered_pre_zhixian_context",
            NETWORK_ROOT / "stopfiltered_global_filtered_pre_zhixian_context",
        ),
        (
            NETWORK_ROOT / "jointup" / "long_period_manual_p001_filtered_pre_zhixian_context",
            NETWORK_ROOT / "stopfiltered_long_period_manual_p001_filtered_pre_zhixian_context",
        ),
    ]
    for jointup_dir, output_dir in targets:
        inputs = sorted(
            path
            for path in jointup_dir.glob("network_constitutional_applied_*_all-tokens_*.csv")
            if path.is_file()
        )
        if not inputs:
            raise FileNotFoundError(f"No recoverable joint-filtered files under {jointup_dir}")
        print(f"[network restore] {jointup_dir.name}: {len(inputs)} files", flush=True)
        for index, input_path in enumerate(inputs, start=1):
            command = [
                sys.executable,
                str(STOPWORD_SCRIPT),
                "--input-csv",
                str(input_path),
                "--runtime-rule-dir",
                str(RUNTIME_RULE_DIR),
                "--stopword-profile",
                "always",
                "--output-dir",
                str(output_dir),
                "--summary-json",
                "true",
            ]
            subprocess.run(command, cwd=str(ROOT), check=True)
            print(f"[network restore {index}/{len(inputs)}] {input_path.name}", flush=True)


def refresh_p001_legacy_aliases() -> List[Path]:
    directory = NETWORK_ROOT / "stopfiltered_long_period_manual_p001_filtered_pre_zhixian_context"
    aliases: List[Path] = []
    pattern = re.compile(
        r"^network_constitutional_applied_(regex-only|strict|full)_all-tokens_"
        r"w(\d+)_split-lr_none_raw-freq_long_period_manual_p001_"
        r"filtered_pre_zhixian_context_joint(\d+)up_stopv5alwaysfiltered\.csv$"
    )
    for source in sorted(directory.glob("network_constitutional_applied_*_all-tokens_*.csv")):
        match = pattern.fullmatch(source.name)
        if not match:
            continue
        profile, window, threshold = match.groups()
        alias = directory / (
            f"network_constitutional_applied_{profile}_w{window}_long_period_manual_p001_"
            f"filtered_joint{threshold}up_stopv5always.csv"
        )
        shutil.copy2(source, alias)
        aliases.append(alias)
    print(f"[network aliases] refreshed={len(aliases)}", flush=True)
    return aliases


def patch_networks(excluded_uids: Set[int]) -> None:
    if NETWORK_MARKER.exists():
        try:
            existing_report = json.loads(NETWORK_MARKER.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            existing_report = {}
        if existing_report.get("network_patch_version") == NETWORK_PATCH_VERSION:
            print(f"[network] already patched: {NETWORK_MARKER}")
            return
        superseded = REPORT_DIR / "network_patch_report.before_token_mask.json"
        if not superseded.exists():
            shutil.copy2(NETWORK_MARKER, superseded)

    restore_directed_files_from_jointup()

    reports: List[Dict[str, object]] = []
    stats_by_condition: Dict[Tuple[str, int], Dict[str, object]] = {}
    retained_pairs_by_condition: Dict[Tuple[str, int], DefaultDict[PairKey, Set[int]]] = {}

    for profile in PROFILES:
        removed_tokens = read_selected_tokens(profile, context_uids=ADDED_CONTEXT_UIDS)
        masked_context_tokens = read_selected_tokens(profile, context_uids=TOKEN_MASK_CONTEXT_UIDS)
        article_uids = set(pd.to_numeric(removed_tokens["article_uid"], errors="raise").astype(int))
        article_uids.update(
            pd.to_numeric(masked_context_tokens["article_uid"], errors="raise").astype(int)
        )
        same_article_tokens = read_selected_tokens(profile, article_uids=article_uids)
        retained_tokens = same_article_tokens[
            ~same_article_tokens["context_uid"].astype(int).isin(excluded_uids)
        ].copy()
        retained_tokens = retained_tokens.loc[~token_mask_series(retained_tokens)].copy()
        for window in WINDOWS:
            stats = merge_event_stats(
                build_event_stats(removed_tokens, window),
                build_masked_event_stats(masked_context_tokens, window),
            )
            retained_stats = build_event_stats(retained_tokens, window)
            stats_by_condition[(profile, window)] = stats
            retained_pairs_by_condition[(profile, window)] = retained_stats["pair_articles"]  # type: ignore[assignment]

    directed_files = selected_directed_files()
    for index, path in enumerate(directed_files, start=1):
        profile, window, threshold = parse_profile_window_threshold(path)
        report = patch_network_csv(
            path,
            stats_by_condition[(profile, window)],
            retained_pairs_by_condition[(profile, window)],
            threshold,
        )
        reports.append(report)
        print(f"[network {index}/{len(directed_files)}] {path.name}", flush=True)

    legacy_aliases = refresh_p001_legacy_aliases()

    undirected_reports: List[Dict[str, object]] = []
    for directory in {
        path.parent for path in directed_files
    }:
        undirected_dir = directory / "undirected"
        if not undirected_dir.is_dir():
            continue
        for undirected_path in sorted(undirected_dir.glob("network_constitutional_applied_*.csv")):
            directed_path = directory / undirected_path.name
            if not directed_path.is_file():
                continue
            kept_rows, output_rows = convert_undirected_file(directed_path)
            undirected_reports.append(
                {
                    "file": str(undirected_path),
                    "directed_r_rows": kept_rows,
                    "undirected_rows": output_rows,
                }
            )
            print(f"[undirected] {undirected_path.name}")

    payload = {
        "applied_at": datetime.now().isoformat(timespec="seconds"),
        "network_patch_version": NETWORK_PATCH_VERSION,
        "added_context_uids": sorted(ADDED_CONTEXT_UIDS),
        "token_mask_csv": str(TOKEN_MASK_CSV),
        "token_mask_context_uids": sorted(TOKEN_MASK_CONTEXT_UIDS),
        "token_mask_text": TOKEN_MASK_TEXT,
        "directed_files": reports,
        "p001_legacy_aliases": [str(path) for path in legacy_aliases],
        "undirected_files": undirected_reports,
    }
    NETWORK_MARKER.parent.mkdir(parents=True, exist_ok=True)
    NETWORK_MARKER.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[network] report: {NETWORK_MARKER}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-counts", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_uids = apply_filter_revision()
    write_token_mask()
    if len(excluded_uids) != 1371:
        raise ValueError(f"Unexpected revised filter size: {len(excluded_uids)} (expected 1371)")
    if not RETAINED_POST1912_EXCLUSIONS.issubset(excluded_uids):
        raise ValueError("Required post-1912 manual exclusions are missing.")
    if not args.skip_counts:
        write_period_counts(excluded_uids)
    if not args.skip_network:
        patch_networks(excluded_uids)


if __name__ == "__main__":
    main()
