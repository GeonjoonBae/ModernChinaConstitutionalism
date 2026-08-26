from __future__ import annotations

import argparse
import csv
import re
import sys
from contextlib import contextmanager
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
from shenbao_token_filter_utils import (
    DEFAULT_TOKEN_FILTER,
    apply_token_filter_to_df,
    load_token_filter,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    ROOT_DIR / "shenbao" / "shenbao_textdata" / "count" / "shenbao_textdata_count_daily.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "shenbao" / "shenbao_pelt"
DEFAULT_APPLIED_TOKENS_ROOT = (
    ROOT_DIR / "shenbao" / "shenbao_network" / "applied_tokens"
)
DEFAULT_PERIODS_PARQUET = (
    DEFAULT_APPLIED_TOKENS_ROOT / "regex_only" / "periods" / "periods.parquet"
)
DEFAULT_DATE_COL = "date"
DEFAULT_COLUMNS = ("lixian", "xianzheng", "xianfa", "zhixian")
DEFAULT_SOURCE = "auto"
DEFAULT_TOKEN_PROFILE = "regex_only"
DEFAULT_COUNT_UNIT = "article"
DEFAULT_ROLLING_WINDOW = 30
ROLLING_LINE_WIDTH = 1.5
LEGEND_FONT_SIZE = 20
LEGEND_HANDLE_LENGTH = 4.0
LEGEND_BORDER_PAD = 1.6
LEGEND_LABEL_SPACING = 1.0
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


def display_name(column: str) -> str:
    return DISPLAY_NAMES.get(column, KEYWORD_TOKEN_MAP.get(column, column))


def series_color(column: str) -> str:
    return SERIES_COLORS.get(column, SERIES_COLORS.get(display_name(column), "#1f77b4"))


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


def parse_columns(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_COLUMNS)

    columns: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in value.replace(",", " ").split():
            column = token.strip()
            if not column or column in seen:
                continue
            seen.add(column)
            columns.append(column)

    if not columns:
        raise ValueError("At least one column is required.")
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
            "Plot trailing rolling averages for selected Shenbao daily count columns "
            "on one line chart."
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
        help="Column used to match keyword columns in applied-tokens mode. Default: token.",
    )
    parser.add_argument(
        "--target-match",
        choices=("contains", "exact"),
        default="contains",
        help="How keyword columns match tokens in applied-tokens mode. Default: contains.",
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
        "--token-filter",
        default=str(DEFAULT_TOKEN_FILTER),
        help="Token-level filter CSV. Use 'none' to disable.",
    )
    parser.add_argument(
        "--date-col",
        default=DEFAULT_DATE_COL,
        help=f"Date column name. Default: {DEFAULT_DATE_COL}",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Columns to plot. May be space- or comma-separated. "
            "Default: lixian xianzheng xianfa zhixian."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help=f"Trailing rolling average window in days. Default: {DEFAULT_ROLLING_WINDOW}",
    )
    parser.add_argument(
        "--start-date",
        help="Optional first date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional last date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--period-set-id",
        default="",
        help="Optional period set whose boundaries are drawn on the plot.",
    )
    parser.add_argument(
        "--periods-parquet",
        default=str(DEFAULT_PERIODS_PARQUET),
        help=f"Periods parquet used with --period-set-id. Default: {DEFAULT_PERIODS_PARQUET}",
    )
    args = parser.parse_args()

    if args.rolling_window <= 0:
        raise ValueError(f"--rolling-window must be positive: {args.rolling_window}")

    args.start_date = parse_optional_date(args.start_date, "--start-date")
    args.end_date = parse_optional_date(args.end_date, "--end-date")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")

    args.input = Path(args.input)
    args.applied_tokens_root = Path(args.applied_tokens_root)
    args.token_parquet = Path(args.token_parquet) if args.token_parquet else None
    args.calendar_input = Path(args.calendar_input)
    args.output_dir = Path(args.output_dir)
    args.context_filter_root = Path(args.context_filter_root)
    args.token_filter = None if str(args.token_filter).lower() == "none" else Path(args.token_filter)
    args.periods_parquet = Path(args.periods_parquet)
    args.columns = parse_columns(args.columns)
    return args


def sanitize_filename_token(value: str) -> str:
    token = re.sub(r'[<>:"/\\\\|?*]', "_", value.strip())
    token = re.sub(r"\s+", "_", token)
    token = token.strip("._")
    if not token:
        raise ValueError("Filename token must not be empty.")
    return token


def read_daily_counts(
    path: Path,
    date_col: str,
    columns: Sequence[str],
) -> tuple[list[datetime], dict[str, list[float]]]:
    dates: list[datetime] = []
    values = {column: [] for column in columns}

    with open_dict_reader(path) as reader:
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {date_col, *columns} - fieldnames
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

            dates.append(date_value)
            for column in columns:
                raw_value = (row.get(column) or "").strip()
                try:
                    values[column].append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value in {path} at data row {row_number}, "
                        f"column {column!r}: {raw_value!r}"
                    ) from exc

    if not dates:
        raise ValueError(f"No data rows found in {path}")

    order = np.argsort(np.asarray(dates, dtype="datetime64[us]"))
    sorted_dates = [dates[index] for index in order]
    sorted_values = {
        column: [series[index] for index in order]
        for column, series in values.items()
    }
    return sorted_dates, sorted_values


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
    return pd.DataFrame({"date": sorted(set(dates))}).reset_index(drop=True)


def applied_tokens_path(args: argparse.Namespace) -> Path:
    if args.token_parquet:
        return args.token_parquet
    return args.applied_tokens_root / args.token_profile / "tokens.parquet"


def can_build_applied_column(column: str) -> bool:
    return column in {"total_dedup", "sum"} or column in KEYWORD_TOKEN_MAP


def resolve_source(args: argparse.Namespace) -> str:
    if args.source != "auto":
        return args.source
    if args.context_filter_info and all(can_build_applied_column(column) for column in args.columns):
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


def read_applied_counts(args: argparse.Namespace) -> tuple[list[datetime], dict[str, list[float]]]:
    token_path = applied_tokens_path(args)
    if not token_path.exists():
        raise FileNotFoundError(f"Applied tokens parquet not found: {token_path}")

    unsupported = [column for column in args.columns if not can_build_applied_column(column)]
    if unsupported:
        raise ValueError(
            "Cannot build these --columns values from applied tokens: "
            f"{', '.join(unsupported)}. Use --source daily-csv --context-filter none, "
            "or select one of total_dedup, sum, zhixian, lixian, xianzheng, xianfa."
        )

    required_columns = list(
        dict.fromkeys(["date", args.token_match_column, "token", "article_uid", "context_uid"])
    )
    tokens_df = pd.read_parquet(token_path, columns=required_columns)
    tokens_df["date"] = pd.to_datetime(tokens_df["date"], format=DATE_FORMAT, errors="raise")
    tokens_df, args.context_filter_summary = apply_context_filter_to_df(tokens_df, args.context_filter_info)
    tokens_df, args.token_filter_summary = apply_token_filter_to_df(tokens_df, args.token_filter_rows)

    output_df = read_calendar_frame(args.calendar_input, args.date_col)
    match_values = tokens_df[args.token_match_column].fillna("").astype(str)
    keyword_counts: dict[str, pd.Series] = {}

    for column in args.columns:
        if column == "total_dedup":
            counts = count_by_unit(tokens_df, args.count_unit)
        elif column == "sum":
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
            target = KEYWORD_TOKEN_MAP[column]
            matched = tokens_df.loc[keyword_mask(match_values, target, args.target_match)]
            counts = count_by_unit(matched, args.count_unit)
        output_df[column] = output_df["date"].map(counts).fillna(0).astype(int)

    return (
        [value.to_pydatetime() for value in output_df["date"]],
        {column: [float(value) for value in output_df[column]] for column in args.columns},
    )


def filter_date_range(
    dates: Sequence[datetime],
    values: dict[str, list[float]],
    start_date: datetime | None,
    end_date: datetime | None,
) -> tuple[list[datetime], dict[str, list[float]]]:
    indexes = [
        index
        for index, date_value in enumerate(dates)
        if (start_date is None or date_value >= start_date)
        and (end_date is None or date_value <= end_date)
    ]

    if not indexes:
        available_start = dates[0].date().isoformat()
        available_end = dates[-1].date().isoformat()
        raise ValueError(
            "No rows match the requested date range. "
            f"Available range: {available_start} - {available_end}"
        )

    filtered_dates = [dates[index] for index in indexes]
    filtered_values = {
        column: [series[index] for index in indexes]
        for column, series in values.items()
    }
    return filtered_dates, filtered_values


def rolling_average(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    rolling = np.full(len(array), np.nan, dtype=float)
    if window <= 1:
        return array
    if len(array) < window:
        return rolling

    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    rolling[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return rolling


def year_locator_base(dates: Sequence[datetime]) -> int:
    year_span = dates[-1].year - dates[0].year
    if year_span > 80:
        return 10
    if year_span > 35:
        return 5
    return 1


def output_paths(
    output_dir: Path,
    columns: Sequence[str],
    rolling_window: int,
    start_date: datetime | None,
    end_date: datetime | None,
    context_filter_info,
    period_set_id: str,
) -> tuple[Path, Path]:
    column_token = "_".join(sanitize_filename_token(column) for column in columns)
    prefix = f"rolling{rolling_window}_{column_token}"
    if start_date:
        prefix += f"_from{start_date:%Y%m%d}"
    if end_date:
        prefix += f"_to{end_date:%Y%m%d}"
    prefix += context_filter_stem_part(context_filter_info)
    if period_set_id:
        prefix += "_with_periods"
    return (
        output_dir / f"{prefix}_values.csv",
        output_dir / f"{prefix}_line_plot.png",
    )


def write_rolling_values(
    path: Path,
    dates: Sequence[datetime],
    rolling_values: dict[str, np.ndarray],
    rolling_window: int,
) -> None:
    fieldnames = ["date"] + [
        f"{column}_rolling{rolling_window}" for column in rolling_values
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, date_value in enumerate(dates):
            row = {"date": date_value.date().isoformat()}
            for column, values in rolling_values.items():
                value = values[index]
                row[f"{column}_rolling{rolling_window}"] = (
                    "" if np.isnan(value) else f"{float(value):.6f}"
                )
            writer.writerow(row)


def plot_rolling_values(
    path: Path,
    dates: Sequence[datetime],
    rolling_values: dict[str, np.ndarray],
    rolling_window: int,
    period_boundaries: Sequence[tuple[datetime, str]],
) -> None:
    fig, axis = plt.subplots(figsize=(20, 8))

    for column, values in rolling_values.items():
        axis.plot(
            dates,
            values,
            linewidth=ROLLING_LINE_WIDTH,
            color=series_color(column),
            label=display_name(column),
        )

    axis.set_title(
        f"{rolling_window}-Day Rolling Average of Shenbao Daily Counts",
        pad=12,
    )
    axis.set_xlabel("Date")
    axis.set_ylabel(f"{rolling_window}-day rolling average")
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(bottom=0)
    axis.grid(True, which="major", axis="both", alpha=0.25)
    axis.xaxis.set_major_locator(mdates.YearLocator(base=year_locator_base(dates)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for boundary_date, boundary_label in period_boundaries:
        if boundary_date < dates[0] or boundary_date > dates[-1]:
            continue
        axis.axvline(boundary_date, color="#4b5563", linewidth=1.0, linestyle="--", alpha=0.7)
        axis.text(
            boundary_date,
            0.96,
            boundary_label,
            rotation=90,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            color="#374151",
            fontsize=9,
        )
    legend = axis.legend(
        loc="upper left",
        framealpha=0.95,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=LEGEND_BORDER_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    legend_handles = (
        legend.legend_handles
        if hasattr(legend, "legend_handles")
        else legend.legendHandles
    )
    for handle in legend_handles:
        handle.set_linewidth(ROLLING_LINE_WIDTH * 2)

    fig.autofmt_xdate(rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def load_period_boundaries(path: Path, period_set_id: str) -> list[tuple[datetime, str]]:
    if not period_set_id:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Periods parquet not found: {path}")
    periods = pd.read_parquet(
        path,
        columns=["period_set_id", "period_id", "sort_order", "start_date"],
    )
    periods = periods[periods["period_set_id"].astype(str).eq(period_set_id)].copy()
    if periods.empty:
        raise ValueError(f"No periods found for period_set_id={period_set_id}")
    periods = periods.sort_values(["sort_order", "period_id"], kind="mergesort")
    return [
        (
            datetime.strptime(str(row.start_date), DATE_FORMAT),
            str(row.start_date).replace("-", "."),
        )
        for row in periods.itertuples(index=False)
    ]


def main() -> None:
    set_csv_field_size_limit()
    configure_plot_fonts()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.context_filter_info = load_context_filter(args.context_filter, args.context_filter_root)
    args.context_filter_summary = None
    args.token_filter_rows = load_token_filter(args.token_filter)
    args.token_filter_summary = None
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
    if args.token_filter_rows:
        print(f"Token filter: {args.token_filter} ({len(args.token_filter_rows):,} rules)")
    else:
        print("Token filter: none")
    print(f"Input source: {args.resolved_source}")

    if args.resolved_source == "applied-tokens":
        dates, daily_values = read_applied_counts(args)
    else:
        dates, daily_values = read_daily_counts(args.input, args.date_col, args.columns)
    dates, daily_values = filter_date_range(
        dates,
        daily_values,
        args.start_date,
        args.end_date,
    )
    rolling_values = {
        column: rolling_average(values, args.rolling_window)
        for column, values in daily_values.items()
    }
    values_path, plot_path = output_paths(
        args.output_dir,
        args.columns,
        args.rolling_window,
        args.start_date,
        args.end_date,
        args.context_filter_info,
        args.period_set_id,
    )

    period_boundaries = load_period_boundaries(args.periods_parquet, args.period_set_id)

    write_rolling_values(values_path, dates, rolling_values, args.rolling_window)
    plot_rolling_values(
        plot_path,
        dates,
        rolling_values,
        args.rolling_window,
        period_boundaries,
    )

    print(f"Wrote {values_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None
