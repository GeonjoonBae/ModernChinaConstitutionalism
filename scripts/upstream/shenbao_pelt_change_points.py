from __future__ import annotations

import argparse
import csv
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from shenbao_context_filter_utils import (
    DEFAULT_CONTEXT_FILTER_NAME,
    DEFAULT_FILTER_ROOT,
    apply_context_filter_to_df,
    context_filter_stem_part,
    load_context_filter,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    ROOT_DIR / "shenbao" / "shenbao_textdata" / "count" / "shenbao_textdata_count_daily.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "shenbao" / "shenbao_pelt"
DEFAULT_APPLIED_TOKENS_ROOT = (
    ROOT_DIR / "shenbao" / "shenbao_network" / "applied_tokens"
)
DEFAULT_DATE_COL = "date"
DEFAULT_VALUE_COL = "total_dedup"
DEFAULT_SOURCE = "auto"
DEFAULT_TOKEN_PROFILE = "regex_only"
DEFAULT_COUNT_UNIT = "article"
DEFAULT_MODEL = "l2"
DEFAULT_MIN_SIZE = 15
DEFAULT_PENALTIES = (5.0, 10.0, 20.0, 50.0)
DEFAULT_JUMP = 1
DEFAULT_ROLLING_WINDOW = 30
DATE_FORMAT = "%Y-%m-%d"
KEYWORD_TOKEN_MAP = {
    "zhixian": "制憲",
    "制憲": "制憲",
    "lixian": "立憲",
    "立憲": "立憲",
    "xianzheng": "憲政",
    "憲政": "憲政",
    "헌정": "憲政",
    "xianfa": "憲法",
    "憲法": "憲法",
}
DISPLAY_NAMES = {
    "zhixian": "制憲",
    "lixian": "立憲",
    "xianzheng": "憲政",
    "xianfa": "憲法",
    "total_dedup": "total_dedup",
    "sum": "sum",
}
SERIES_COLORS = {
    "zhixian": "#ffd400",
    "制憲": "#ffd400",
    "lixian": "#00a651",
    "立憲": "#00a651",
    "xianzheng": "#0057ff",
    "憲政": "#0057ff",
    "헌정": "#0057ff",
    "xianfa": "#ef0000",
    "憲法": "#ef0000",
}
CJK_FONT_CANDIDATES = (
    "Microsoft JhengHei",
    "Noto Sans TC",
    "Microsoft YaHei",
    "MingLiU",
    "SimHei",
    "SimSun",
)


@dataclass(frozen=True)
class DailySeries:
    dates: list[datetime]
    values: list[float]


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def configure_plot_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def display_name(value_col: str) -> str:
    return DISPLAY_NAMES.get(value_col, KEYWORD_TOKEN_MAP.get(value_col, value_col))


def series_color(value_col: str) -> str:
    return SERIES_COLORS.get(value_col, SERIES_COLORS.get(display_name(value_col), "#1f77b4"))


@contextmanager
def open_dict_reader(path: Path) -> Iterator[csv.DictReader]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_header = reader.fieldnames[0] if reader.fieldnames else ""
        if not first_header.startswith("\ufeff"):
            yield reader
            return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield csv.DictReader(handle)


def parse_penalties(values: Sequence[str]) -> list[float]:
    penalties: list[float] = []
    seen: set[float] = set()

    for value in values:
        for token in value.replace(",", " ").split():
            try:
                penalty = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid penalty value: {token!r}") from exc
            if penalty <= 0:
                raise ValueError(f"Penalty value must be positive: {penalty:g}")
            if penalty in seen:
                continue
            seen.add(penalty)
            penalties.append(penalty)

    if not penalties:
        raise ValueError("At least one penalty value is required.")
    return penalties


def parse_value_columns(values: Sequence[Sequence[str]] | None) -> list[str]:
    if not values:
        return [DEFAULT_VALUE_COL]

    columns: list[str] = []
    seen: set[str] = set()
    for group in values:
        for value in group:
            for token in value.replace(",", " ").split():
                column = token.strip()
                if not column or column in seen:
                    continue
                seen.add(column)
                columns.append(column)

    if not columns:
        raise ValueError("At least one --value-col value is required.")
    return columns


def parse_optional_date(value: str | None, option_name: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must use YYYY-MM-DD format: {value!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PELT change point detection on shenbao_textdata_count_daily.csv. "
            "Outputs penalty-specific CSV and PNG files under shenbao/shenbao_pelt by default."
        )
    )
    parser.add_argument(
        "--source",
        choices=("auto", "daily-csv", "applied-tokens"),
        default=DEFAULT_SOURCE,
        help=(
            "Input source. auto uses filtered applied tokens when a context filter is active "
            "and the selected columns can be built from tokens. Default: auto."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Input daily count CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--applied-tokens-root",
        default=str(DEFAULT_APPLIED_TOKENS_ROOT),
        help=f"Applied tokens root directory. Default: {DEFAULT_APPLIED_TOKENS_ROOT}",
    )
    parser.add_argument(
        "--token-profile",
        default=DEFAULT_TOKEN_PROFILE,
        help=f"Applied token profile to use. Default: {DEFAULT_TOKEN_PROFILE}",
    )
    parser.add_argument(
        "--token-parquet",
        help="Optional direct applied tokens parquet path. Overrides --applied-tokens-root/profile.",
    )
    parser.add_argument(
        "--token-match-column",
        choices=("token", "dict_lv1", "dict_lv2"),
        default="token",
        help="Column used to match keyword value columns in applied-tokens mode. Default: token.",
    )
    parser.add_argument(
        "--target-match",
        choices=("contains", "exact"),
        default="contains",
        help="How keyword value columns match tokens in applied-tokens mode. Default: contains.",
    )
    parser.add_argument(
        "--count-unit",
        choices=("article", "context", "token"),
        default=DEFAULT_COUNT_UNIT,
        help=f"Count unit in applied-tokens mode. Default: {DEFAULT_COUNT_UNIT}.",
    )
    parser.add_argument(
        "--calendar-input",
        default=str(DEFAULT_INPUT_PATH),
        help="Calendar CSV used to fill missing dates in applied-tokens mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
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
        "--date-col",
        default=DEFAULT_DATE_COL,
        help=f"Date column name. Default: {DEFAULT_DATE_COL}",
    )
    parser.add_argument(
        "--value-col",
        action="append",
        nargs="+",
        help=(
            "Numeric column(s) to analyze. May be repeated or comma-separated, "
            f"for example: --value-col total_dedup zhixian. Default: {DEFAULT_VALUE_COL}"
        ),
    )
    parser.add_argument(
        "--pens",
        nargs="+",
        help="Penalty values to compare, for example: --pens 5 10 or --pens 5,10,20,50.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help=f"Minimum segment length in observations. Default: {DEFAULT_MIN_SIZE}",
    )
    parser.add_argument(
        "--model",
        choices=("l1", "l2", "rbf", "normal"),
        default=DEFAULT_MODEL,
        help=f"ruptures PELT cost model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--jump",
        type=int,
        default=DEFAULT_JUMP,
        help=f"Subsample step for candidate change points. Default: {DEFAULT_JUMP}",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help=f"Trailing rolling average window in days for plots. Default: {DEFAULT_ROLLING_WINDOW}",
    )
    parser.add_argument(
        "--start-date",
        help="Optional first date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional last date to include, in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    if args.min_size <= 0:
        raise ValueError(f"--min-size must be positive: {args.min_size}")
    if args.jump <= 0:
        raise ValueError(f"--jump must be positive: {args.jump}")
    if args.rolling_window <= 0:
        raise ValueError(f"--rolling-window must be positive: {args.rolling_window}")

    args.start_date = parse_optional_date(args.start_date, "--start-date")
    args.end_date = parse_optional_date(args.end_date, "--end-date")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")

    args.pens = (
        parse_penalties(args.pens)
        if args.pens
        else list(DEFAULT_PENALTIES)
    )
    args.value_cols = parse_value_columns(args.value_col)
    args.input = Path(args.input)
    args.applied_tokens_root = Path(args.applied_tokens_root)
    args.token_parquet = Path(args.token_parquet) if args.token_parquet else None
    args.calendar_input = Path(args.calendar_input)
    args.output_dir = Path(args.output_dir)
    args.context_filter_root = Path(args.context_filter_root)
    return args


def sanitize_filename_token(value: str) -> str:
    token = re.sub(r'[<>:"/\\\\|?*]', "_", value.strip())
    token = re.sub(r"\s+", "_", token)
    token = token.strip("._")
    if not token:
        raise ValueError("Filename token must not be empty.")
    return token


def penalty_token(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def format_number(value: float) -> str:
    return f"{value:.6f}"


def read_daily_series(path: Path, date_col: str, value_col: str) -> DailySeries:
    records: list[tuple[datetime, float]] = []
    seen_dates: set[datetime] = set()

    with open_dict_reader(path) as reader:
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {date_col, value_col} - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Missing required column(s) in {path}: {missing}. "
                f"Available columns: {available}"
            )

        for row_number, row in enumerate(reader, start=1):
            raw_date = (row.get(date_col) or "").strip()
            try:
                date_value = datetime.strptime(raw_date, DATE_FORMAT)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid date in {path} at data row {row_number}: {raw_date!r}"
                ) from exc

            if date_value in seen_dates:
                raise ValueError(
                    f"Duplicate date in {path} at data row {row_number}: {raw_date!r}"
                )
            seen_dates.add(date_value)

            raw_value = (row.get(value_col) or "").strip()
            try:
                count_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {path} at data row {row_number}, "
                    f"column {value_col!r}: {raw_value!r}"
                ) from exc

            records.append((date_value, count_value))

    if not records:
        raise ValueError(f"No data rows found in {path}")

    records.sort(key=lambda item: item[0])
    dates = [record[0] for record in records]
    values = [record[1] for record in records]
    return DailySeries(dates=dates, values=values)


def read_calendar_frame(path: Path, date_col: str) -> pd.DataFrame:
    with open_dict_reader(path) as reader:
        if date_col not in (reader.fieldnames or []):
            raise ValueError(f"Missing date column in calendar input: {path}")
        dates: list[datetime] = []
        for row_number, row in enumerate(reader, start=1):
            raw_date = (row.get(date_col) or "").strip()
            try:
                dates.append(datetime.strptime(raw_date, DATE_FORMAT))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid date in {path} at data row {row_number}: {raw_date!r}"
                ) from exc
    if not dates:
        raise ValueError(f"No calendar rows found in {path}")
    out = pd.DataFrame({"date": sorted(set(dates))})
    return out.reset_index(drop=True)


def applied_tokens_path(args: argparse.Namespace) -> Path:
    if args.token_parquet:
        return args.token_parquet
    return args.applied_tokens_root / args.token_profile / "tokens.parquet"


def can_build_applied_column(value_col: str) -> bool:
    return value_col in {"total_dedup", "sum"} or value_col in KEYWORD_TOKEN_MAP


def resolve_source(args: argparse.Namespace) -> str:
    if args.source != "auto":
        return args.source
    if args.context_filter_info and all(can_build_applied_column(column) for column in args.value_cols):
        return "applied-tokens"
    return "daily-csv"


def count_by_unit(frame: pd.DataFrame, count_unit: str) -> pd.Series:
    if count_unit == "article":
        return frame[["date", "article_uid"]].drop_duplicates().groupby("date").size()
    if count_unit == "context":
        return frame[["date", "context_uid"]].drop_duplicates().groupby("date").size()
    return frame.groupby("date").size()


def keyword_mask(values: pd.Series, target: str, match_mode: str) -> pd.Series:
    if match_mode == "exact":
        return values == target
    return values.str.contains(target, regex=False, na=False)


def read_applied_count_frame(args: argparse.Namespace) -> pd.DataFrame:
    token_path = applied_tokens_path(args)
    if not token_path.exists():
        raise FileNotFoundError(f"Applied tokens parquet not found: {token_path}")

    unsupported = [column for column in args.value_cols if not can_build_applied_column(column)]
    if unsupported:
        raise ValueError(
            "Cannot build these --value-col values from applied tokens: "
            f"{', '.join(unsupported)}. Use --source daily-csv --context-filter none, "
            "or select one of total_dedup, sum, zhixian, lixian, xianzheng, xianfa."
        )

    required_columns = ["date", args.token_match_column, "article_uid", "context_uid"]
    tokens_df = pd.read_parquet(token_path, columns=required_columns)
    tokens_df["date"] = pd.to_datetime(tokens_df["date"], format=DATE_FORMAT, errors="raise")
    tokens_df, args.context_filter_summary = apply_context_filter_to_df(tokens_df, args.context_filter_info)

    calendar_df = read_calendar_frame(args.calendar_input, args.date_col)
    output_df = calendar_df.copy()
    match_values = tokens_df[args.token_match_column].fillna("").astype(str)
    keyword_counts: dict[str, pd.Series] = {}

    for value_col in args.value_cols:
        if value_col == "total_dedup":
            counts = count_by_unit(tokens_df, args.count_unit)
        elif value_col == "sum":
            if not keyword_counts:
                for keyword_col, target in {
                    "zhixian": "制憲",
                    "lixian": "立憲",
                    "xianzheng": "憲政",
                    "xianfa": "憲法",
                }.items():
                    matched = tokens_df.loc[keyword_mask(match_values, target, args.target_match)]
                    keyword_counts[keyword_col] = count_by_unit(matched, args.count_unit)
            counts = sum(keyword_counts.values(), pd.Series(dtype=float))
        else:
            target = KEYWORD_TOKEN_MAP[value_col]
            matched = tokens_df.loc[keyword_mask(match_values, target, args.target_match)]
            counts = count_by_unit(matched, args.count_unit)
        output_df[value_col] = output_df["date"].map(counts).fillna(0).astype(int)

    return output_df


def daily_series_from_frame(frame: pd.DataFrame, value_col: str) -> DailySeries:
    if value_col not in frame.columns:
        raise ValueError(f"Missing value column in generated count frame: {value_col}")
    return DailySeries(
        dates=[value.to_pydatetime() for value in frame["date"]],
        values=[float(value) for value in frame[value_col]],
    )


def read_series(args: argparse.Namespace, value_col: str) -> DailySeries:
    if args.resolved_source == "daily-csv":
        return read_daily_series(args.input, args.date_col, value_col)
    if not hasattr(args, "applied_count_frame"):
        args.applied_count_frame = read_applied_count_frame(args)
    return daily_series_from_frame(args.applied_count_frame, value_col)


def filter_date_range(
    series: DailySeries,
    start_date: datetime | None,
    end_date: datetime | None,
) -> DailySeries:
    indexes = [
        index
        for index, date_value in enumerate(series.dates)
        if (start_date is None or date_value >= start_date)
        and (end_date is None or date_value <= end_date)
    ]

    if not indexes:
        available_start = series.dates[0].date().isoformat()
        available_end = series.dates[-1].date().isoformat()
        raise ValueError(
            "No rows match the requested date range. "
            f"Available range: {available_start} - {available_end}"
        )

    return DailySeries(
        dates=[series.dates[index] for index in indexes],
        values=[series.values[index] for index in indexes],
    )


def import_ruptures():
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing required package 'ruptures'. Install it with: "
            "python -m pip install ruptures"
        ) from exc
    return rpt


def normalize_breakpoints(raw_breakpoints: Sequence[int], length: int) -> list[int]:
    breakpoints = sorted({int(point) for point in raw_breakpoints if 0 < int(point) <= length})
    if not breakpoints or breakpoints[-1] != length:
        breakpoints.append(length)
    return breakpoints


def write_change_points(
    path: Path,
    dates: list[datetime],
    breakpoints: list[int],
    value_col: str,
    model: str,
    penalty: float,
    min_size: int,
    jump: int,
) -> None:
    fieldnames = [
        "series",
        "model",
        "penalty",
        "min_size",
        "jump",
        "change_point_number",
        "change_point_index",
        "change_point_date",
        "previous_segment_end_index",
        "previous_segment_end_date",
        "next_segment_start_index",
        "next_segment_start_date",
    ]
    change_points = [point for point in breakpoints if point < len(dates)]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for number, point in enumerate(change_points, start=1):
            writer.writerow(
                {
                    "series": value_col,
                    "model": model,
                    "penalty": f"{penalty:g}",
                    "min_size": min_size,
                    "jump": jump,
                    "change_point_number": number,
                    "change_point_index": point,
                    "change_point_date": dates[point].date().isoformat(),
                    "previous_segment_end_index": point - 1,
                    "previous_segment_end_date": dates[point - 1].date().isoformat(),
                    "next_segment_start_index": point,
                    "next_segment_start_date": dates[point].date().isoformat(),
                }
            )


def write_segments(
    path: Path,
    dates: list[datetime],
    values: list[float],
    breakpoints: list[int],
    value_col: str,
    model: str,
    penalty: float,
    min_size: int,
    jump: int,
) -> None:
    fieldnames = [
        "series",
        "model",
        "penalty",
        "min_size",
        "jump",
        "segment_number",
        "start_index",
        "end_index",
        "start_date",
        "end_date",
        "days",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "sum",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        start = 0
        for number, end in enumerate(breakpoints, start=1):
            segment_values = np.asarray(values[start:end], dtype=float)
            writer.writerow(
                {
                    "series": value_col,
                    "model": model,
                    "penalty": f"{penalty:g}",
                    "min_size": min_size,
                    "jump": jump,
                    "segment_number": number,
                    "start_index": start,
                    "end_index": end - 1,
                    "start_date": dates[start].date().isoformat(),
                    "end_date": dates[end - 1].date().isoformat(),
                    "days": end - start,
                    "mean": format_number(float(np.mean(segment_values))),
                    "median": format_number(float(np.median(segment_values))),
                    "std": format_number(float(np.std(segment_values))),
                    "min": format_number(float(np.min(segment_values))),
                    "max": format_number(float(np.max(segment_values))),
                    "sum": format_number(float(np.sum(segment_values))),
                }
            )
            start = end


def year_locator_base(dates: list[datetime]) -> int:
    year_span = dates[-1].year - dates[0].year
    if year_span > 80:
        return 10
    if year_span > 35:
        return 5
    return 1


def rolling_average(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    rolling = np.full(len(array), np.nan, dtype=float)
    if window <= 1:
        return array
    if len(array) < window:
        return rolling

    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    rolling[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return rolling


def monthly_totals(
    dates: list[datetime],
    values: list[float],
) -> tuple[list[datetime], list[float]]:
    month_dates: list[datetime] = []
    month_values: list[float] = []
    current_month: tuple[int, int] | None = None

    for date_value, count_value in zip(dates, values):
        month_key = (date_value.year, date_value.month)
        if month_key != current_month:
            current_month = month_key
            month_dates.append(datetime(date_value.year, date_value.month, 1))
            month_values.append(float(count_value))
        else:
            month_values[-1] += float(count_value)

    return month_dates, month_values


def plot_change_points(
    path: Path,
    dates: list[datetime],
    values: list[float],
    breakpoints: list[int],
    value_col: str,
    model: str,
    penalty: float,
    min_size: int,
    jump: int,
    rolling_window: int,
) -> None:
    display = display_name(value_col)
    color = series_color(value_col)
    fig, (daily_axis, monthly_axis) = plt.subplots(
        2,
        1,
        figsize=(20, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    change_points = [point for point in breakpoints if point < len(dates)]
    for point_number, point in enumerate(change_points):
        label = "PELT change point" if point_number == 0 else None
        daily_axis.axvline(
            dates[point],
            color="#111827",
            linewidth=1.0,
            alpha=0.7,
            label=label,
        )
        monthly_axis.axvline(
            dates[point],
            color="#111827",
            linewidth=1.0,
            alpha=0.7,
        )

    daily_axis.vlines(
        dates,
        0,
        values,
        linewidth=0.35,
        color=color,
        alpha=0.45,
        label=f"Daily count: {display}",
    )

    daily_axis.plot(
        dates,
        rolling_average(values, rolling_window),
        linewidth=1.7,
        color=color,
        alpha=0.95,
        label=f"{rolling_window}-day rolling average",
    )

    start = 0
    for end in breakpoints:
        segment_values = values[start:end]
        segment_mean = float(np.mean(segment_values))
        label = "PELT segment mean" if start == 0 else None
        daily_axis.hlines(
            segment_mean,
            dates[start],
            dates[end - 1],
            colors="#4b5563",
            linewidth=1.7,
            alpha=0.85,
            label=label,
        )
        start = end

    month_dates, month_values = monthly_totals(dates, values)
    monthly_axis.plot(
        month_dates,
        month_values,
        linewidth=1.4,
        color=color,
        label="Monthly total",
    )

    daily_axis.set_title(
        (
            f"PELT Change Points: {display}, pen={penalty:g}, "
            f"model={model}, min_size={min_size}, jump={jump}, "
            f"rolling={rolling_window}"
        ),
        pad=12,
    )
    daily_axis.set_ylabel(f"Daily {display}")
    monthly_axis.set_ylabel(f"Monthly {display}")
    monthly_axis.set_xlabel("Date")

    for axis in (daily_axis, monthly_axis):
        axis.set_xlim(dates[0], dates[-1])
        axis.set_ylim(bottom=0)
        axis.grid(True, which="major", axis="both", alpha=0.25)
        axis.legend(loc="upper left", framealpha=0.95)

    monthly_axis.xaxis.set_major_locator(mdates.YearLocator(base=year_locator_base(dates)))
    monthly_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.autofmt_xdate(rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def output_paths(
    output_dir: Path,
    value_col: str,
    min_size: int,
    penalty: float,
    model: str,
    start_date: datetime | None,
    end_date: datetime | None,
    context_filter_info,
) -> tuple[Path, Path, Path]:
    series_token = sanitize_filename_token(value_col)
    pen_token = penalty_token(penalty)
    model_token = sanitize_filename_token(model)
    prefix = f"pelt_{series_token}"
    if start_date:
        prefix += f"_from{start_date:%Y%m%d}"
    if end_date:
        prefix += f"_to{end_date:%Y%m%d}"
    prefix += context_filter_stem_part(context_filter_info)
    prefix += f"_min{min_size}_pen{pen_token}_{model_token}"
    return (
        output_dir / f"{prefix}_change_points.csv",
        output_dir / f"{prefix}_segments.csv",
        output_dir / f"{prefix}_plot.png",
    )


def run_series(args: argparse.Namespace, value_col: str, rpt) -> None:
    series = read_series(args, value_col)
    series = filter_date_range(series, args.start_date, args.end_date)
    if len(series.values) < args.min_size:
        raise ValueError(
            f"Not enough observations for --min-size {args.min_size}: "
            f"{len(series.values)} rows"
        )

    print(
        "Analysis range: "
        f"{series.dates[0].date().isoformat()} - {series.dates[-1].date().isoformat()} "
        f"({len(series.values)} rows)"
    )

    signal = np.asarray(series.values, dtype=float).reshape(-1, 1)
    algorithm = rpt.Pelt(model=args.model, min_size=args.min_size, jump=args.jump).fit(signal)

    for penalty in args.pens:
        print(f"Running PELT: series={value_col}, model={args.model}, penalty={penalty:g}")
        breakpoints = normalize_breakpoints(
            algorithm.predict(pen=penalty),
            len(series.values),
        )
        change_points_path, segments_path, plot_path = output_paths(
            args.output_dir,
            value_col,
            args.min_size,
            penalty,
            args.model,
            args.start_date,
            args.end_date,
            args.context_filter_info,
        )

        write_change_points(
            change_points_path,
            series.dates,
            breakpoints,
            value_col,
            args.model,
            penalty,
            args.min_size,
            args.jump,
        )
        write_segments(
            segments_path,
            series.dates,
            series.values,
            breakpoints,
            value_col,
            args.model,
            penalty,
            args.min_size,
            args.jump,
        )
        plot_change_points(
            plot_path,
            series.dates,
            series.values,
            breakpoints,
            value_col,
            args.model,
            penalty,
            args.min_size,
            args.jump,
            args.rolling_window,
        )

        print(f"Penalty: {penalty:g}")
        print(f"Change points: {max(len(breakpoints) - 1, 0)}")
        print(f"Wrote {change_points_path}")
        print(f"Wrote {segments_path}")
        print(f"Wrote {plot_path}")


def main() -> None:
    set_csv_field_size_limit()
    configure_plot_fonts()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.context_filter_info = load_context_filter(args.context_filter, args.context_filter_root)
    args.context_filter_summary = None
    args.resolved_source = resolve_source(args)
    if args.resolved_source == "daily-csv" and args.context_filter_info:
        print("Context filter skipped: daily-csv source has no context_uid.")
    elif args.context_filter_info:
        print(
            "Context filter: "
            f"{args.context_filter_info.name} "
            f"({len(args.context_filter_info.excluded_context_uids):,} excluded context_uid)"
        )
    else:
        print("Context filter: none")
    print(f"Input source: {args.resolved_source}")

    rpt = import_ruptures()
    for value_col in args.value_cols:
        run_series(args, value_col, rpt)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None
