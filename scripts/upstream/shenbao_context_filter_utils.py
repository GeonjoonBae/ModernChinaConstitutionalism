#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_FILTER_ROOT = ROOT / "shenbao" / "shenbao_filters"
DEFAULT_CONTEXT_FILTER_NAME = "filter_context_pre_zhixian"


@dataclass(frozen=True)
class ContextFilterInfo:
    name: str
    path: Path
    basis: str
    reason: str
    filename_suffix: str
    excluded_context_uids: Set[str]


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sanitize_slug(value: object) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "value"


def parse_filter_name(filter_name: str) -> Tuple[str, str]:
    stem = Path(filter_name).stem
    if not stem.startswith("filter_"):
        raise ValueError(f"Context filter name must start with filter_: {filter_name}")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Context filter name must follow filter_<basis>_<reason>.csv: {filter_name}")
    basis = parts[1]
    reason = "_".join(parts[2:])
    return sanitize_slug(basis), sanitize_slug(reason)


def resolve_filter_path(filter_name: str, filter_root: Path = DEFAULT_FILTER_ROOT) -> Path:
    raw = filter_name.strip()
    if not raw:
        raise ValueError("Empty context filter name.")
    path = Path(raw).expanduser()
    if path.suffix.lower() == ".csv" or path.parent != Path("."):
        return path.resolve()
    basis, _reason = parse_filter_name(raw)
    return (filter_root / basis / f"{Path(raw).stem}.csv").resolve()


def load_context_filter(
    filter_name: Optional[str],
    filter_root: Path = DEFAULT_FILTER_ROOT,
) -> Optional[ContextFilterInfo]:
    if not filter_name or filter_name.strip().lower() in {"none", "no", "false", "0"}:
        return None

    path = resolve_filter_path(filter_name, filter_root)
    if not path.exists():
        raise FileNotFoundError(f"Context filter CSV not found: {path}")

    basis, reason = parse_filter_name(path.stem)
    raise_csv_field_limit()
    excluded: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "context_uid" not in reader.fieldnames:
            raise ValueError(f"Context filter CSV must include context_uid column: {path}")
        for row in reader:
            context_uid = str(row.get("context_uid", "")).strip()
            if context_uid:
                excluded.add(context_uid)

    return ContextFilterInfo(
        name=path.stem,
        path=path,
        basis=basis,
        reason=reason,
        filename_suffix=f"filtered_{reason}_{basis}",
        excluded_context_uids=excluded,
    )


def apply_context_filter_to_df(
    df: pd.DataFrame,
    filter_info: Optional[ContextFilterInfo],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if filter_info is None:
        return df, {
            "context_filter": None,
            "excluded_context_uid_count": 0,
            "rows_before_context_filter": int(len(df)),
            "rows_after_context_filter": int(len(df)),
            "rows_removed_by_context_filter": 0,
        }
    if "context_uid" not in df.columns:
        raise ValueError("Cannot apply context filter: dataframe has no context_uid column.")

    before_rows = int(len(df))
    before_contexts = int(df["context_uid"].nunique())
    keep = ~df["context_uid"].astype(str).isin(filter_info.excluded_context_uids)
    out = df[keep].copy()
    after_rows = int(len(out))
    after_contexts = int(out["context_uid"].nunique())
    return out, {
        "context_filter": filter_info.name,
        "context_filter_csv": str(filter_info.path),
        "context_filter_suffix": filter_info.filename_suffix,
        "excluded_context_uid_count": int(len(filter_info.excluded_context_uids)),
        "rows_before_context_filter": before_rows,
        "rows_after_context_filter": after_rows,
        "rows_removed_by_context_filter": before_rows - after_rows,
        "contexts_before_context_filter": before_contexts,
        "contexts_after_context_filter": after_contexts,
        "contexts_removed_by_context_filter": before_contexts - after_contexts,
    }


def context_filter_stem_part(filter_info: Optional[ContextFilterInfo]) -> str:
    return f"_{filter_info.filename_suffix}" if filter_info else ""
