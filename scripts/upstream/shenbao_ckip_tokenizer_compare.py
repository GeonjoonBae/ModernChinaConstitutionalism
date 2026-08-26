# -*- coding: utf-8 -*-
"""Run CKIP word segmentation on Shenbao textdata CSV files.

This script adds CKIP Transformers word-segmentation outputs to the existing
tokenizer comparison workflow. It can run against the sample CSVs in this
workflow repository or the full local Shenbao textdata folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


CSV_FIELD_LIMIT = 2_000_000
DEFAULT_MODELS = ["bert-base", "albert-base", "bert-tiny", "albert-tiny"]
MODEL_CHOICES = {
    "bert-base",
    "albert-base",
    "bert-tiny",
    "albert-tiny",
}
LABEL_KEYWORDS = {
    "lixian": "立憲",
    "xianfa": "憲法",
    "xianzheng": "憲政",
    "zhixian": "制憲",
}
ERROR_PREFIX = "[ERROR]"


OUTPUT_COLUMNS = [
    "tokenizer",
    "source_file",
    "label",
    "page",
    "item_index",
    "article_id",
    "keyword",
    "text_source",
    "text",
    "token",
    "token_count",
    "exact",
    "in_long_token",
    "splited",
    "skipped",
    "skip_reason",
]
SUMMARY_COLUMNS = [
    "tokenizer",
    "model",
    "status",
    "rows",
    "processed_rows",
    "skipped_rows",
    "exact_T",
    "exact_F",
    "in_long_token_T",
    "splited_T",
    "load_seconds",
    "tokenize_seconds",
    "rows_per_second",
    "error",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_full_input = script_dir.parent / "shenbao" / "shenbao_textdata"
    default_sample_input = script_dir / "shenbao_textdata"

    parser = argparse.ArgumentParser(
        description="Tokenize Shenbao textdata with CKIP Transformers word segmenters."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Input directory. Defaults to ../shenbao/shenbao_textdata, or to "
            "./shenbao_textdata when --sample is used."
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use workflow sample files matching '(sample)shenbao_textdata_*.csv'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/tokenize_ckip.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "CKIP WordSegmenter model names. Use any of: "
            + ", ".join(sorted(MODEL_CHOICES))
            + ", or 'all'."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="CKIP device. Use -1 for CPU, 0 for GPU:0.",
    )
    parser.add_argument(
        "--text-source",
        choices=["text", "title_text"],
        default="text",
        help="Tokenize text only, or title + text joined by one space.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size passed to CkipWordSegmenter.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="max_length passed to CkipWordSegmenter.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20_000,
        help="Skip rows whose tokenization text is longer than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit input rows after reading. Use 0 for all rows.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["long", "per-model", "both"],
        default="long",
        help="Save one long-format CSV, per-model CSVs, or both.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check input files and dependency availability; do not tokenize.",
    )

    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = default_sample_input if args.sample else default_full_input
    if args.output_dir is None:
        args.output_dir = args.input_dir / "tokenize_ckip"
    return args


def require_ckip_transformers() -> None:
    if importlib.util.find_spec("ckip_transformers") is None:
        raise SystemExit(
            "ckip-transformers is not installed.\n"
            "Install it before running CKIP tokenization:\n"
            "  python -m pip install ckip-transformers\n"
            "The script does not download or vendor CKIP model weights into this repository."
        )


def resolve_models(requested: list[str]) -> list[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(DEFAULT_MODELS)
    invalid = [model for model in requested if model not in MODEL_CHOICES]
    if invalid:
        raise ValueError(
            "Unsupported model(s): "
            + ", ".join(invalid)
            + ". Supported models: "
            + ", ".join(sorted(MODEL_CHOICES))
        )
    return requested


def discover_input_files(input_dir: Path, sample: bool) -> list[Path]:
    pattern = "(sample)shenbao_textdata_*.csv" if sample else "shenbao_textdata_*.csv"
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input files matching {pattern!r} in {input_dir}")
    return files


def parse_article_id(detail_url: str) -> str:
    if not detail_url:
        return ""
    parsed = parse_qs(urlsplit(detail_url).query)
    article_id = parsed.get("id", [""])[0]
    if article_id:
        return article_id
    match = re.search(r"[?&]id=([^&]+)", detail_url)
    return match.group(1) if match else ""


def normalize_token_list(tokens: object) -> list[str]:
    if isinstance(tokens, str):
        raw = [tokens]
    else:
        raw = list(tokens or [])
    return [str(token).strip() for token in raw if str(token).strip()]


def make_tokenization_text(row: dict[str, str], text_source: str) -> str:
    text = row.get("text", "") or ""
    if text_source == "text":
        return text
    title = row.get("title", "") or ""
    return f"{title} {text}".strip()


def row_skip_reason(row: dict[str, str], text_to_tokenize: str, max_chars: int) -> str:
    label = row.get("label", "")
    if label not in LABEL_KEYWORDS:
        return "unknown_label"
    if not text_to_tokenize.strip():
        return "empty_text"
    if text_to_tokenize.lstrip().startswith(ERROR_PREFIX):
        return "error_text"
    if max_chars > 0 and len(text_to_tokenize) > max_chars:
        return "too_long"
    return ""


def read_rows(files: list[Path], text_source: str, max_chars: int, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = {"label", "page", "item_index", "detail_url", "title", "text"}

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            for row in reader:
                text_to_tokenize = make_tokenization_text(row, text_source)
                label = row.get("label", "")
                rows.append(
                    {
                        "source_file": path.name,
                        "label": label,
                        "page": row.get("page", ""),
                        "item_index": row.get("item_index", ""),
                        "article_id": parse_article_id(row.get("detail_url", "")),
                        "keyword": LABEL_KEYWORDS.get(label, ""),
                        "text_source": text_source,
                        "text": text_to_tokenize,
                        "skip_reason": row_skip_reason(row, text_to_tokenize, max_chars),
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    return rows
    return rows


def batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def classify_keyword(tokens: list[str], keyword: str) -> tuple[str, str, str]:
    if not keyword:
        return "", "", ""
    exact = keyword in tokens
    if exact:
        return "T", "F", "F"

    in_long_token = any(keyword in token and token != keyword for token in tokens)
    splited = False
    if len(keyword) >= 2:
        first = keyword[0]
        rest = keyword[1:]
        splited = any(
            left.endswith(first) and right.startswith(rest)
            for left, right in zip(tokens, tokens[1:])
        )
    return "F", "T" if in_long_token else "F", "T" if splited else "F"


def output_row(
    model_slug: str,
    source_row: dict[str, str],
    tokens: list[str] | None,
) -> dict[str, str]:
    skip_reason = source_row.get("skip_reason", "")
    if skip_reason:
        token_text = ""
        token_count = "0"
        exact = in_long = splited = ""
        skipped = "T"
    else:
        assert tokens is not None
        token_text = " ".join(tokens)
        token_count = str(len(tokens))
        exact, in_long, splited = classify_keyword(tokens, source_row["keyword"])
        skipped = "F"

    return {
        "tokenizer": model_slug,
        "source_file": source_row["source_file"],
        "label": source_row["label"],
        "page": source_row["page"],
        "item_index": source_row["item_index"],
        "article_id": source_row["article_id"],
        "keyword": source_row["keyword"],
        "text_source": source_row["text_source"],
        "text": source_row["text"],
        "token": token_text,
        "token_count": token_count,
        "exact": exact,
        "in_long_token": in_long,
        "splited": splited,
        "skipped": skipped,
        "skip_reason": skip_reason,
    }


def model_slug(model: str) -> str:
    return "ckip_" + model.replace("-", "_")


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, str]], columns: list[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def summarize_output_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        counter["rows"] += 1
        if row["skipped"] == "T":
            counter["skipped_rows"] += 1
            continue
        counter["processed_rows"] += 1
        if row["exact"] == "T":
            counter["exact_T"] += 1
        elif row["exact"] == "F":
            counter["exact_F"] += 1
        if row["in_long_token"] == "T":
            counter["in_long_token_T"] += 1
        if row["splited"] == "T":
            counter["splited_T"] += 1
    return dict(counter)


def run_model(
    model: str,
    rows: list[dict[str, str]],
    device: int,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    from ckip_transformers.nlp import CkipWordSegmenter

    slug = model_slug(model)
    summary: dict[str, object] = {
        "tokenizer": slug,
        "model": model,
        "status": "pending",
        "rows": len(rows),
        "processed_rows": 0,
        "skipped_rows": 0,
        "exact_T": 0,
        "exact_F": 0,
        "in_long_token_T": 0,
        "splited_T": 0,
        "load_seconds": 0,
        "tokenize_seconds": 0,
        "rows_per_second": 0,
        "error": "",
    }

    load_start = time.perf_counter()
    try:
        segmenter = CkipWordSegmenter(model=model, device=device)
    except Exception as exc:
        summary["status"] = "load_failed"
        summary["load_seconds"] = round(time.perf_counter() - load_start, 4)
        summary["error"] = repr(exc)
        return [], summary
    summary["load_seconds"] = round(time.perf_counter() - load_start, 4)

    output_rows: list[dict[str, str]] = []
    to_process = [row for row in rows if not row["skip_reason"]]
    skipped = [output_row(slug, row, None) for row in rows if row["skip_reason"]]

    tokenize_start = time.perf_counter()
    try:
        token_by_key: dict[tuple[str, str, str], list[str]] = {}
        for batch in batched(to_process, batch_size):
            texts = [row["text"] for row in batch]
            batch_output = segmenter(
                texts,
                use_delim=True,
                batch_size=batch_size,
                max_length=max_length,
            )
            if len(batch_output) != len(batch):
                raise ValueError(
                    f"Unexpected CKIP batch output length {len(batch_output)} "
                    f"for batch size {len(batch)}."
                )
            for row, tokens in zip(batch, batch_output):
                key = (row["source_file"], row["label"], row["item_index"])
                token_by_key[key] = normalize_token_list(tokens)

        for row in rows:
            if row["skip_reason"]:
                continue
            key = (row["source_file"], row["label"], row["item_index"])
            output_rows.append(output_row(slug, row, token_by_key[key]))
        output_rows.extend(skipped)
        output_rows.sort(
            key=lambda row: (
                row["source_file"],
                int(row["item_index"]) if row["item_index"].isdigit() else 0,
                row["tokenizer"],
            )
        )
    except Exception as exc:
        summary["status"] = "tokenize_failed"
        summary["tokenize_seconds"] = round(time.perf_counter() - tokenize_start, 4)
        summary["error"] = repr(exc)
        return output_rows, summary

    elapsed = time.perf_counter() - tokenize_start
    summary.update(summarize_output_rows(output_rows))
    summary["status"] = "ok"
    summary["tokenize_seconds"] = round(elapsed, 4)
    summary["rows_per_second"] = round(len(to_process) / elapsed, 4) if elapsed else 0
    return output_rows, summary


def print_input_summary(files: list[Path], rows: list[dict[str, str]]) -> None:
    print("Input files:")
    for path in files:
        print(f"- {path}")
    print(f"Rows: {len(rows)}")
    print("Rows by label:")
    for label, count in sorted(Counter(row["label"] for row in rows).items()):
        print(f"- {label}: {count}")
    skipped = Counter(row["skip_reason"] for row in rows if row["skip_reason"])
    if skipped:
        print("Rows skipped before tokenization:")
        for reason, count in sorted(skipped.items()):
            print(f"- {reason}: {count}")
    else:
        print("Rows skipped before tokenization: 0")


def main() -> None:
    csv.field_size_limit(CSV_FIELD_LIMIT)
    args = parse_args()
    models = resolve_models(args.models)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    files = discover_input_files(input_dir, args.sample)
    rows = read_rows(files, args.text_source, args.max_chars, args.limit)
    print_input_summary(files, rows)

    if args.check_only:
        if importlib.util.find_spec("ckip_transformers") is None:
            print("Dependency check: ckip-transformers is not installed.")
            print("Install command: python -m pip install ckip-transformers")
        else:
            print("Dependency check: ckip-transformers is installed.")
        return

    require_ckip_transformers()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "sample" if args.sample else "full"
    text_source = args.text_source
    long_path = output_dir / f"shenbao_ckip_tokenize_{suffix}_{text_source}_long.csv"
    summary_path = output_dir / f"shenbao_ckip_tokenize_{suffix}_{text_source}_summary.csv"
    if args.output_mode in {"long", "both"} and long_path.exists():
        long_path.unlink()

    summary_rows: list[dict[str, object]] = []
    for model in models:
        print(f"[load/tokenize] {model}")
        output_rows, summary = run_model(
            model=model,
            rows=rows,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        summary_rows.append(summary)

        slug = model_slug(model)
        if output_rows and args.output_mode in {"long", "both"}:
            append_csv(
                long_path,
                output_rows,
                OUTPUT_COLUMNS,
                write_header=not long_path.exists(),
            )
        if output_rows and args.output_mode in {"per-model", "both"}:
            model_path = output_dir / f"shenbao_ckip_tokenize_{suffix}_{text_source}_{slug}.csv"
            write_csv(model_path, output_rows, OUTPUT_COLUMNS)

        if summary["status"] == "ok":
            print(
                f"[done] {model}: processed={summary['processed_rows']} "
                f"skipped={summary['skipped_rows']} exact_T={summary['exact_T']} "
                f"in_long={summary['in_long_token_T']} splited={summary['splited_T']}"
            )
        else:
            print(f"[fail] {model}: {summary['status']} {summary['error']}")

    write_csv(summary_path, [dict(row) for row in summary_rows], SUMMARY_COLUMNS)
    print("Output:")
    if args.output_mode in {"long", "both"}:
        print(f"- {long_path}")
    if args.output_mode in {"per-model", "both"}:
        print(f"- per-model CSV files in {output_dir}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
