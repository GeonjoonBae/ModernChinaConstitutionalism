#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run CKIP WS/POS/NER on Shenbao constitutional keyword-centered contexts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SEED = 1
DATASET_LABEL = "constitutional"
KEYWORDS = ["憲法", "立憲", "憲政", "制憲"]
SPECIAL_AD_VALUE = "分類廣告"
WINDOW_CHARS = 50
SEGMENT_CHAR_LIMIT = 400
MODEL_KEY = "ckip_bert_base_chinese"

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "shenbao" / "shenbao_textdata" / "preprocess" / "shenbao_textdata_stage3_preprocessed_articles_constitutional.csv"
OUTPUT_DIR = BASE_DIR / "shenbao" / "shenbao_tokenize" / "test"
OUTPUT_BASE = "shenbao_nlp_ckip_bert-base-chinese_context_constitutional"
OUTPUT_JSONL = OUTPUT_DIR / f"{OUTPUT_BASE}.jsonl"
OUTPUT_METADATA = OUTPUT_DIR / f"{OUTPUT_BASE}.metadata.json"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{OUTPUT_BASE}.summary.csv"
EXECUTION_SCOPE = "non-ad"

SUMMARY_COLUMNS = [
    "nlp_index",
    "dataset_index",
    "source_labels",
    "article_id",
    "date",
    "publish_variant",
    "issue_page",
    "context_index",
    "context_start",
    "context_end",
    "context_char_len",
    "keyword_occurrence_count",
    "matched_keywords",
    "keyword_occurrences_brief",
    "token_count",
    "ner_count",
    "ner_summary",
    "ws_joined",
    "status",
    "error_message",
]


def parse_keyword_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(KEYWORDS)
    keywords: List[str] = []
    for value in values:
        for item in str(value).split(","):
            keyword = item.strip()
            if keyword:
                keywords.append(keyword)
    if not keywords:
        raise ValueError("--keywords must contain at least one non-empty keyword.")
    return keywords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CKIP bert-base Chinese WS/POS/NER on Shenbao keyword-centered contexts."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument(
        "--scope",
        choices=["non-ad", "classified-ad", "sp-ad", "ads", "all"],
        default=EXECUTION_SCOPE,
        help=(
            "Article execution scope. non-ad preserves the ver0_2 behavior; "
            "classified-ad keeps only rows with special_column equal to classified ads."
        ),
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Context-center keywords. Pass space-separated values or comma-separated groups.",
    )
    parser.add_argument("--window-chars", type=int, default=WINDOW_CHARS)
    parser.add_argument("--segment-char-limit", type=int, default=SEGMENT_CHAR_LIMIT)
    parser.add_argument("--context-batch", type=int, default=None)
    parser.add_argument("--driver-batch", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output JSONL, summary CSV, and metadata JSON before running.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global INPUT_CSV
    global OUTPUT_DIR
    global OUTPUT_BASE
    global OUTPUT_JSONL
    global OUTPUT_METADATA
    global OUTPUT_SUMMARY
    global EXECUTION_SCOPE
    global KEYWORDS
    global WINDOW_CHARS
    global SEGMENT_CHAR_LIMIT

    INPUT_CSV = args.input_csv.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_BASE = args.output_base
    OUTPUT_JSONL = OUTPUT_DIR / f"{OUTPUT_BASE}.jsonl"
    OUTPUT_METADATA = OUTPUT_DIR / f"{OUTPUT_BASE}.metadata.json"
    OUTPUT_SUMMARY = OUTPUT_DIR / f"{OUTPUT_BASE}.summary.csv"
    EXECUTION_SCOPE = args.scope
    KEYWORDS = parse_keyword_args(args.keywords)
    WINDOW_CHARS = args.window_chars
    SEGMENT_CHAR_LIMIT = args.segment_char_limit


def delete_existing_outputs() -> None:
    for path in (OUTPUT_JSONL, OUTPUT_METADATA, OUTPUT_SUMMARY):
        if path.exists():
            path.unlink()


def row_matches_execution_scope(row: Dict[str, Any]) -> bool:
    qrynewstype = (row.get("qrynewstype") or "").strip()
    special_column = (row.get("special_column") or "").strip()
    is_sp_ad = qrynewstype == "SP_AD"
    is_classified_ad = special_column == SPECIAL_AD_VALUE

    if EXECUTION_SCOPE == "non-ad":
        return not is_sp_ad and not is_classified_ad
    if EXECUTION_SCOPE == "classified-ad":
        return is_classified_ad
    if EXECUTION_SCOPE == "sp-ad":
        return is_sp_ad
    if EXECUTION_SCOPE == "ads":
        return is_sp_ad or is_classified_ad
    if EXECUTION_SCOPE == "all":
        return True
    raise ValueError(f"Unknown execution scope: {EXECUTION_SCOPE}")


def describe_scope_filters() -> Dict[str, Any]:
    common = {
        "collect_error_must_be_null_or_empty": True,
        "collision_must_equal": "F",
        "analysis_text_must_contain_any": KEYWORDS,
    }
    if EXECUTION_SCOPE == "non-ad":
        common.update({"qrynewstype_exclude": ["SP_AD"], "special_column_exclude": [SPECIAL_AD_VALUE]})
    elif EXECUTION_SCOPE == "classified-ad":
        common.update({"special_column_must_equal": SPECIAL_AD_VALUE})
    elif EXECUTION_SCOPE == "sp-ad":
        common.update({"qrynewstype_must_equal": "SP_AD"})
    elif EXECUTION_SCOPE == "ads":
        common.update({"must_match_any": [{"qrynewstype": "SP_AD"}, {"special_column": SPECIAL_AD_VALUE}]})
    elif EXECUTION_SCOPE == "all":
        common.update({"article_type_filter": None})
    return common


def set_csv_field_size_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10


def detect_csv_encoding(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
    return "utf-8-sig" if first_line.startswith("\ufeff") else "utf-8"


def is_blank(value: Optional[str]) -> bool:
    return value is None or value.strip() == ""


def safe_int(value: Any) -> Any:
    try:
        if value is None or str(value).strip() == "":
            return value
        return int(str(value))
    except (TypeError, ValueError):
        return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def find_keyword_occurrences(text: str) -> List[Dict[str, Any]]:
    occurrences: List[Dict[str, Any]] = []
    for keyword in KEYWORDS:
        start = 0
        while True:
            index = text.find(keyword, start)
            if index < 0:
                break
            occurrences.append(
                {
                    "keyword": keyword,
                    "start": index,
                    "end": index + len(keyword),
                }
            )
            start = index + 1
    return sorted(occurrences, key=lambda item: (item["start"], item["end"], item["keyword"]))


def merge_windows(text_len: int, occurrences: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    windows = sorted(
        (
            max(0, occurrence["start"] - WINDOW_CHARS),
            min(text_len, occurrence["end"] + WINDOW_CHARS),
        )
        for occurrence in occurrences
    )
    merged: List[List[int]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def make_segments(text_len: int) -> List[Dict[str, int]]:
    segments = []
    for segment_index, start in enumerate(range(0, text_len, SEGMENT_CHAR_LIMIT)):
        end = min(text_len, start + SEGMENT_CHAR_LIMIT)
        segments.append(
            {
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "char_len": end - start,
            }
        )
    if not segments:
        segments.append({"segment_index": 0, "start": 0, "end": 0, "char_len": 0})
    return segments


def iter_context_records(path: Path, encoding: str) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for physical_index, row in enumerate(reader):
            text = row.get("analysis_text") or ""
            if not row_matches_execution_scope(row):
                continue
            if not is_blank(row.get("collect_error")):
                continue
            if row.get("collision") != "F":
                continue

            occurrences = find_keyword_occurrences(text)
            if not occurrences:
                continue

            context_spans = merge_windows(len(text), occurrences)
            for context_index, (context_start, context_end) in enumerate(context_spans):
                context_text = text[context_start:context_end]
                context_occurrences = []
                for occurrence in occurrences:
                    if context_start <= occurrence["start"] and occurrence["end"] <= context_end:
                        context_occurrences.append(
                            {
                                "keyword": occurrence["keyword"],
                                "start": occurrence["start"],
                                "end": occurrence["end"],
                                "relative_start": occurrence["start"] - context_start,
                                "relative_end": occurrence["end"] - context_start,
                            }
                        )

                yield {
                    "physical_index": physical_index,
                    "dataset_index": safe_int(row.get("dataset_index")),
                    "source_labels": row.get("source_labels") or "",
                    "article_id": row.get("article_id") or "",
                    "publish_variant": row.get("publish_variant") or "",
                    "date": row.get("date") or "",
                    "issue_page": row.get("issue_page") or "",
                    "context_index": context_index,
                    "context_start": context_start,
                    "context_end": context_end,
                    "context_char_len": len(context_text),
                    "context_text": context_text,
                    "keyword_occurrence_count": len(context_occurrences),
                    "keyword_occurrences": context_occurrences,
                    "segments": make_segments(len(context_text)),
                }


def align_token_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if token == "":
            offsets.append((cursor, cursor))
            continue
        if text.startswith(token, cursor):
            start = cursor
        else:
            found = text.find(token, cursor)
            start = found if found >= 0 else cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def normalize_ner_item(item: Any, segment_start: int, context_start: int) -> Dict[str, Any]:
    text = getattr(item, "word", None)
    ner_type = getattr(item, "ner", None)
    idx = getattr(item, "idx", None)
    if text is None and isinstance(item, (list, tuple)) and len(item) >= 1:
        text = item[0]
    if ner_type is None and isinstance(item, (list, tuple)) and len(item) >= 2:
        ner_type = item[1]
    if idx is None and isinstance(item, (list, tuple)) and len(item) >= 3:
        idx = item[2]
    if idx is None:
        idx = (0, len(text or ""))

    start = segment_start + int(idx[0])
    end = segment_start + int(idx[1])
    return {
        "text": text or "",
        "type": ner_type or "",
        "start": start,
        "end": end,
        "global_start": context_start + start,
        "global_end": context_start + end,
    }


def overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def run_drivers(
    ws_driver: Any,
    pos_driver: Any,
    ner_driver: Any,
    segment_texts: Sequence[str],
    batch_size: int,
) -> Tuple[List[List[str]], List[List[str]], List[List[Any]]]:
    ws_output = ws_driver(
        list(segment_texts),
        use_delim=False,
        batch_size=batch_size,
        show_progress=False,
    )
    pos_output = pos_driver(
        ws_output,
        use_delim=False,
        batch_size=batch_size,
        show_progress=False,
    )
    ner_output = ner_driver(
        list(segment_texts),
        use_delim=False,
        batch_size=batch_size,
        show_progress=False,
    )
    return ws_output, pos_output, ner_output


def build_success_result(
    nlp_index: int,
    record: Dict[str, Any],
    ws_by_segment: Sequence[Sequence[str]],
    pos_by_segment: Sequence[Sequence[str]],
    ner_by_segment: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    context_start = record["context_start"]
    segment_metadata = []
    all_ws: List[str] = []
    all_pos: List[str] = []
    all_tokens: List[Dict[str, Any]] = []
    all_ner: List[Dict[str, Any]] = []

    for segment, ws_items, pos_items, ner_items in zip(
        record["segments"],
        ws_by_segment,
        pos_by_segment,
        ner_by_segment,
    ):
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = record["context_text"][segment_start:segment_end]
        segment_metadata.append(
            {
                "segment_index": segment["segment_index"],
                "start": segment_start,
                "end": segment_end,
                "global_start": context_start + segment_start,
                "global_end": context_start + segment_end,
                "char_len": segment["char_len"],
            }
        )

        local_offsets = align_token_offsets(segment_text, ws_items)
        segment_ner = [
            normalize_ner_item(item, segment_start, context_start) for item in ner_items
        ]
        all_ner.extend(segment_ner)

        for raw_index, token in enumerate(ws_items):
            pos = pos_items[raw_index] if raw_index < len(pos_items) else ""
            local_start, local_end = local_offsets[raw_index]
            start = segment_start + local_start
            end = segment_start + local_end
            if token.strip() == "":
                continue
            token_ner = [
                entity
                for entity in segment_ner
                if overlaps(start, end, entity["start"], entity["end"])
            ]
            token_index = len(all_tokens)
            all_ws.append(token)
            all_pos.append(pos)
            all_tokens.append(
                {
                    "token_index": token_index,
                    "token": token,
                    "pos": pos,
                    "start": start,
                    "end": end,
                    "global_start": context_start + start,
                    "global_end": context_start + end,
                    "ner": token_ner,
                }
            )

    result = {
        "nlp_index": nlp_index,
        "dataset_index": record["dataset_index"],
        "source_labels": record["source_labels"],
        "article_id": record["article_id"],
        "publish_variant": record["publish_variant"],
        "date": record["date"],
        "issue_page": record["issue_page"],
        "context_index": record["context_index"],
        "context_start": record["context_start"],
        "context_end": record["context_end"],
        "context_char_len": record["context_char_len"],
        "keyword_occurrence_count": record["keyword_occurrence_count"],
        "keyword_occurrences": record["keyword_occurrences"],
        "context_text": record["context_text"],
        MODEL_KEY: {
            "ws": all_ws,
            "pos": all_pos,
            "token_count": len(all_tokens),
            "tokens": all_tokens,
            "ner": all_ner,
            "segments": segment_metadata,
        },
        "status": "success",
        "error_message": "",
    }
    return result


def build_error_result(nlp_index: int, record: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return {
        "nlp_index": nlp_index,
        "dataset_index": record["dataset_index"],
        "source_labels": record["source_labels"],
        "article_id": record["article_id"],
        "publish_variant": record["publish_variant"],
        "date": record["date"],
        "issue_page": record["issue_page"],
        "context_index": record["context_index"],
        "context_start": record["context_start"],
        "context_end": record["context_end"],
        "context_char_len": record["context_char_len"],
        "keyword_occurrence_count": record["keyword_occurrence_count"],
        "keyword_occurrences": record["keyword_occurrences"],
        "context_text": record["context_text"],
        MODEL_KEY: {
            "ws": [],
            "pos": [],
            "token_count": 0,
            "tokens": [],
            "ner": [],
            "segments": [
                {
                    "segment_index": segment["segment_index"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "global_start": record["context_start"] + segment["start"],
                    "global_end": record["context_start"] + segment["end"],
                    "char_len": segment["char_len"],
                }
                for segment in record["segments"]
            ],
        },
        "status": "error",
        "error_message": str(error),
    }


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def make_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    ckip = result[MODEL_KEY]
    keyword_occurrences = result["keyword_occurrences"]
    matched_keywords = ";".join(
        unique_in_order(item["keyword"] for item in keyword_occurrences)
    )
    keyword_occurrences_brief = ";".join(
        f"{item['keyword']}@{item['relative_start']}-{item['relative_end']}"
        for item in keyword_occurrences
    )
    ner_summary = ";".join(
        f"{item['text']}[{item['type']}]" for item in ckip["ner"]
    )
    return {
        "nlp_index": result["nlp_index"],
        "dataset_index": result["dataset_index"],
        "source_labels": result["source_labels"],
        "article_id": result["article_id"],
        "date": result["date"],
        "publish_variant": result["publish_variant"],
        "issue_page": result["issue_page"],
        "context_index": result["context_index"],
        "context_start": result["context_start"],
        "context_end": result["context_end"],
        "context_char_len": result["context_char_len"],
        "keyword_occurrence_count": result["keyword_occurrence_count"],
        "matched_keywords": matched_keywords,
        "keyword_occurrences_brief": keyword_occurrences_brief,
        "token_count": ckip["token_count"],
        "ner_count": len(ckip["ner"]),
        "ner_summary": ner_summary,
        "ws_joined": "|".join(ckip["ws"]),
        "status": result["status"],
        "error_message": result["error_message"],
    }


def make_resume_key(obj_or_record: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(obj_or_record.get("dataset_index")),
        str(obj_or_record.get("context_index")),
        str(obj_or_record.get("context_start")),
        str(obj_or_record.get("context_end")),
    )


def prepare_resume_jsonl(jsonl_path: Path) -> Tuple[set[Tuple[str, str, str, str]], int]:
    """Prepare JSONL for automatic resume.

    Keeps valid success objects only, drops all error objects for retry, and removes
    the last valid success object so the next run reprocesses it. This protects
    against a partially written final JSONL line or a final item interrupted around
    write/flush time.
    """
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return set(), 0

    valid_objects: List[Dict[str, Any]] = []
    malformed_count = 0
    error_count = 0
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if obj.get("status") == "success":
                valid_objects.append(obj)
            else:
                error_count += 1

    dropped_last = 0
    if valid_objects:
        valid_objects = valid_objects[:-1]
        dropped_last = 1

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for obj in valid_objects:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    done_keys = {make_resume_key(obj) for obj in valid_objects}
    print(
        "Resume prepared: "
        f"kept_success={len(valid_objects)}, "
        f"dropped_last_success={dropped_last}, "
        f"dropped_error={error_count}, "
        f"dropped_malformed={malformed_count}",
        flush=True,
    )
    return done_keys, len(valid_objects)


def rebuild_summary_from_jsonl(jsonl_path: Path, summary_path: Path) -> None:
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as jsonl_handle, summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as summary_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_COLUMNS)
        summary_writer.writeheader()
        for line in jsonl_handle:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary_writer.writerow(make_summary_row(result))


def compute_stats_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "context_count": 0,
        "success_count": 0,
        "error_count": 0,
        "token_count": 0,
        "ner_count": 0,
        "keyword_occurrence_count": 0,
        "contexts_over_400_chars": 0,
        "segment_count": 0,
        "matched_keyword_counts": Counter(),
    }
    if not jsonl_path.exists():
        stats["matched_keyword_counts"] = {}
        return stats

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            ckip = result.get(MODEL_KEY, {})
            stats["context_count"] += 1
            stats["success_count"] += 1 if result.get("status") == "success" else 0
            stats["error_count"] += 1 if result.get("status") == "error" else 0
            stats["token_count"] += int(ckip.get("token_count") or 0)
            stats["ner_count"] += len(ckip.get("ner", []))
            stats["keyword_occurrence_count"] += int(result.get("keyword_occurrence_count") or 0)
            stats["contexts_over_400_chars"] += 1 if int(result.get("context_char_len") or 0) > SEGMENT_CHAR_LIMIT else 0
            stats["segment_count"] += len(ckip.get("segments", []))
            for item in result.get("keyword_occurrences", []):
                stats["matched_keyword_counts"][item.get("keyword", "")] += 1

    stats["matched_keyword_counts"] = dict(stats["matched_keyword_counts"])
    return stats


def load_ckip_drivers() -> Tuple[Any, Any, Any]:
    from ckip_transformers.nlp import CkipNerChunker, CkipPosTagger, CkipWordSegmenter

    print("Loading CKIP WS model: ckiplab/bert-base-chinese-ws", flush=True)
    ws_driver = CkipWordSegmenter(model="bert-base", device=-1)
    print("Loaded CKIP WS model", flush=True)
    print("Loading CKIP POS model: ckiplab/bert-base-chinese-pos", flush=True)
    pos_driver = CkipPosTagger(model="bert-base", device=-1)
    print("Loaded CKIP POS model", flush=True)
    print("Loading CKIP NER model: ckiplab/bert-base-chinese-ner", flush=True)
    ner_driver = CkipNerChunker(model="bert-base", device=-1)
    print("Loaded CKIP NER model", flush=True)
    return ws_driver, pos_driver, ner_driver


def process_batch(
    records: Sequence[Dict[str, Any]],
    start_nlp_index: int,
    ws_driver: Any,
    pos_driver: Any,
    ner_driver: Any,
    batch_size: int,
) -> List[Dict[str, Any]]:
    segment_texts = []
    segment_counts = []
    for record in records:
        segment_counts.append(len(record["segments"]))
        for segment in record["segments"]:
            segment_texts.append(record["context_text"][segment["start"] : segment["end"]])

    ws_output, pos_output, ner_output = run_drivers(
        ws_driver,
        pos_driver,
        ner_driver,
        segment_texts,
        batch_size=batch_size,
    )

    results = []
    cursor = 0
    for offset, record in enumerate(records):
        count = segment_counts[offset]
        result = build_success_result(
            start_nlp_index + offset,
            record,
            ws_output[cursor : cursor + count],
            pos_output[cursor : cursor + count],
            ner_output[cursor : cursor + count],
        )
        results.append(result)
        cursor += count
    return results


def write_metadata(
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    csv_encoding: str,
    csv_field_limit: int,
    stats: Dict[str, Any],
) -> None:
    metadata = {
        "dataset_label": DATASET_LABEL,
        "input_csv": str(INPUT_CSV),
        "outputs": {
            "jsonl": str(OUTPUT_JSONL),
            "metadata_json": str(OUTPUT_METADATA),
            "summary_csv": str(OUTPUT_SUMMARY),
        },
        "output_base": OUTPUT_BASE,
        "execution_scope": EXECUTION_SCOPE,
        "filters": describe_scope_filters(),
        "random_seed": SEED,
        "keywords": KEYWORDS,
        "window_chars_each_side": WINDOW_CHARS,
        "segment_char_limit": SEGMENT_CHAR_LIMIT,
        "models": {
            "ws": {
                "driver": "CkipWordSegmenter",
                "model": "bert-base",
                "model_name": "ckiplab/bert-base-chinese-ws",
                "device": -1,
            },
            "pos": {
                "driver": "CkipPosTagger",
                "model": "bert-base",
                "model_name": "ckiplab/bert-base-chinese-pos",
                "device": -1,
            },
            "ner": {
                "driver": "CkipNerChunker",
                "model": "bert-base",
                "model_name": "ckiplab/bert-base-chinese-ner",
                "device": -1,
            },
        },
        "ckip_call_options": {
            "use_delim": False,
            "show_progress": False,
        },
        "output_format": {
            "jsonl": {
                "encoding": "utf-8",
                "ensure_ascii": False,
                "unit": "one merged keyword-centered context per line",
            },
            "summary_csv": {
                "encoding": "utf-8",
                "unit": "one row per JSONL object",
            },
        },
        "csv_reading": {
            "encoding": csv_encoding,
            "field_size_limit": csv_field_limit,
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stats": stats,
    }
    with OUTPUT_METADATA.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seed_everything(SEED)
    started_at = datetime.now().astimezone().isoformat()
    start_time = time.time()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        delete_existing_outputs()

    csv_field_limit = set_csv_field_size_limit()
    csv_encoding = detect_csv_encoding(INPUT_CSV)

    print(f"Input CSV: {INPUT_CSV}", flush=True)
    print(f"Output JSONL: {OUTPUT_JSONL}", flush=True)
    print(f"Output metadata: {OUTPUT_METADATA}", flush=True)
    print(f"Output summary: {OUTPUT_SUMMARY}", flush=True)
    print(f"Execution scope: {EXECUTION_SCOPE}", flush=True)
    print(f"Keywords: {', '.join(KEYWORDS)}", flush=True)
    print(f"Window chars each side: {WINDOW_CHARS}", flush=True)
    print(f"Segment char limit: {SEGMENT_CHAR_LIMIT}", flush=True)
    print(f"CSV encoding: {csv_encoding}", flush=True)

    all_records = list(iter_context_records(INPUT_CSV, csv_encoding))
    print(f"Prepared contexts: {len(all_records)}", flush=True)

    done_keys, next_nlp_index = prepare_resume_jsonl(OUTPUT_JSONL)
    records = [record for record in all_records if make_resume_key(record) not in done_keys]
    print(f"Resume completed contexts kept: {len(done_keys)}", flush=True)
    print(f"Remaining contexts to process: {len(records)}", flush=True)

    batch_context_count = args.context_batch or int(os.environ.get("SHENBAO_CKIP_CONTEXT_BATCH", "128"))
    driver_batch_size = args.driver_batch or int(os.environ.get("SHENBAO_CKIP_DRIVER_BATCH", "64"))

    if records:
        ws_driver, pos_driver, ner_driver = load_ckip_drivers()
        with OUTPUT_JSONL.open("a", encoding="utf-8", newline="\n") as jsonl_handle:
            nlp_index = next_nlp_index
            for batch_start in range(0, len(records), batch_context_count):
                batch_records = records[batch_start : batch_start + batch_context_count]
                batch_segment_count = sum(len(record["segments"]) for record in batch_records)
                print(
                    f"Running remaining batch {batch_start}-{batch_start + len(batch_records) - 1} "
                    f"as nlp_index {nlp_index}-{nlp_index + len(batch_records) - 1} "
                    f"with {batch_segment_count} segment(s)",
                    flush=True,
                )
                try:
                    results = process_batch(
                        batch_records,
                        nlp_index,
                        ws_driver,
                        pos_driver,
                        ner_driver,
                        driver_batch_size,
                    )
                except Exception as batch_error:
                    print(
                        f"Batch failed at remaining context {batch_start}; retrying one context at a time: {batch_error}",
                        flush=True,
                    )
                    results = []
                    for offset, record in enumerate(batch_records):
                        try:
                            results.extend(
                                process_batch(
                                    [record],
                                    nlp_index + offset,
                                    ws_driver,
                                    pos_driver,
                                    ner_driver,
                                    1,
                                )
                            )
                        except Exception as context_error:
                            results.append(build_error_result(nlp_index + offset, record, context_error))

                for result in results:
                    jsonl_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

                jsonl_handle.flush()
                nlp_index += len(batch_records)
                if nlp_index % 1024 == 0 or nlp_index == len(all_records):
                    print(f"Processed contexts: {nlp_index}/{len(all_records)}", flush=True)
    else:
        print("No remaining contexts to process after resume preparation.", flush=True)

    rebuild_summary_from_jsonl(OUTPUT_JSONL, OUTPUT_SUMMARY)
    stats = compute_stats_from_jsonl(OUTPUT_JSONL)
    finished_at = datetime.now().astimezone().isoformat()
    elapsed_seconds = time.time() - start_time
    write_metadata(
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        csv_encoding=csv_encoding,
        csv_field_limit=csv_field_limit,
        stats=stats,
    )
    print("Done", flush=True)
    print(f"JSONL: {OUTPUT_JSONL}", flush=True)
    print(f"Metadata: {OUTPUT_METADATA}", flush=True)
    print(f"Summary CSV: {OUTPUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
