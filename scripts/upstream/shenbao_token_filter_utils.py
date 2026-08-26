from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_TOKEN_FILTER = (
    ROOT
    / "shenbao"
    / "shenbao_filters"
    / "token"
    / "filter_token_pre_zhixian_retained_official_title.csv"
)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_token_filter(value: str | Path | None) -> List[Dict[str, str]]:
    if value is None or str(value).strip().lower() in {"", "none", "null", "off"}:
        return []
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Token filter CSV not found: {path}")
    _raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"context_uid", "token_match", "match_mode"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Token filter must contain {sorted(required)}: {path}")
    return rows


def apply_token_filter_to_df(
    frame: pd.DataFrame,
    rows: List[Dict[str, str]],
    *,
    context_column: str = "context_uid",
    token_column: str = "token",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not rows:
        return frame, {"token_filter_rows": 0, "excluded_token_rows": 0}
    if context_column not in frame.columns or token_column not in frame.columns:
        raise ValueError(
            f"Token filtering requires columns {context_column!r} and {token_column!r}."
        )

    context_values = pd.to_numeric(frame[context_column], errors="coerce")
    token_values = frame[token_column].fillna("").astype(str)
    excluded = pd.Series(False, index=frame.index)
    for row in rows:
        context_uid = int(str(row["context_uid"]).strip())
        token_match = str(row["token_match"])
        match_mode = str(row["match_mode"]).strip().lower()
        if match_mode == "exact":
            token_hit = token_values.eq(token_match)
        elif match_mode == "contains":
            token_hit = token_values.str.contains(token_match, regex=False, na=False)
        else:
            raise ValueError(f"Unsupported token filter match_mode: {match_mode!r}")
        excluded |= context_values.eq(context_uid) & token_hit

    return (
        frame.loc[~excluded].copy(),
        {
            "token_filter_rows": len(rows),
            "excluded_token_rows": int(excluded.sum()),
        },
    )
