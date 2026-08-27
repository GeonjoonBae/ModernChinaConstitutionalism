#!/usr/bin/env python
"""Prepare the medium public release from the local Shenbao research workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PAPER_GROUPS = [
    ("long_period_manual_p001", "立憲-憲政"),
    ("long_period_manual_p001", "立憲-憲法"),
    ("long_period_manual_p002", "憲法-制憲"),
    ("long_period_manual_p002", "立憲-憲政"),
    ("long_period_manual_p005", "憲政-憲法"),
    ("long_period_manual_p005", "憲法-制憲"),
]

CONTEXT_INDEX = [
    ("1905-07-13", "日本改定法律沿革攷續", "A2013010391372", "1489", "1905-1913 立憲-憲法", "憲法;法;法律"),
    ("1905-08-01", "吾國宜硏究政法學說", "A2013010392861", "1506;1507", "1905-1913 立憲-憲法", "立憲;憲法;法;權"),
    ("1906-10-28", "對於立憲諭旨敬告書業", "A2013010429295", "2103;2104;2105;2106", "1905-1913 立憲-憲法", "立憲;憲政;憲法;法律"),
    ("1917-08-23", "召集臨時參議院近訊", "A2013040094154", "16341", "1913-1923 憲法-制憲", "舊;解決;憲法;制憲"),
    ("1917-01-14", "審議會出席制限從嚴", "A2013040070647", "15228;15229", "1913-1923 憲法-制憲", "進行;憲法;制憲"),
    ("1946-12-17", "印臨時政府考慮 三日內宣布獨立", "A2013060787569", "34709;34710", "1945-1949 憲法-制憲", "印度;憲法;制憲"),
    ("1946-11-08", "社論 從美國選舉看我國憲法", "A2013060779036", "34119;34120;34121;34122;34123;34124", "1945-1949 憲法-制憲", "美國;憲法;制憲"),
    ("1946-11-26", "王寵惠談新憲章 係根據政協原則擬成者", "A2013060782604", "34376;34377", "1945-1949 憲法-制憲", "美國;憲法;制憲"),
    ("1946-10-11", "方治發表時局意見 國大爲憲政階梯", "A2013060772757", "33991;33992;33993", "1945-1949 憲政-憲法", "民主;人民;保障;精神"),
    ("1946-10-03", "張君勱對時局意見", "A2013060771010", "33965;33966;33967;33968", "1945-1949 憲政-憲法", "統一;憲法;國民大會"),
    ("1946-03-31", "參政會通過敎育文化提案", "A2013060739315", "33512", "1945-1949 憲政-憲法", "敎育;民智;憲政"),
    ("1947-02-02", "新的階段新的規劃 全國敎育會議", "A2013060795770", "35273", "1945-1949 憲政-憲法", "敎育;民主;憲政;憲法"),
]

UPSTREAM_SCRIPTS = [
    "shenbao_nlp_ckip_bert-base-chinese_context_constitutional_ver0_3.py",
    "shenbao_ckip_tokenizer_compare.py",
    "shenbao_dictionary_runtime_split_v2.py",
    "shenbao_applied_token_build.py",
    "shenbao_applied_counts_periods_build.py",
    "shenbao_context_filter_utils.py",
    "shenbao_token_filter_utils.py",
    "shenbao_pre_zhixian_filter_revision.py",
    "shenbao_network_applied.py",
    "shenbao_network_by_period_onestop.py",
    "shenbao_stopword_postfilter_v3.py",
    "shenbao_network_undirected_pmi.py",
    "shenbao_gephi_dynamic_merge.py",
    "shenbao_rolling_average_compare_plot.py",
    "shenbao_pelt_change_points.py",
    "shenbao_focus_cta_dashboard_build.py",
    "shenbao_html_controls.py",
    "shenbao_region_analysis_utils.py",
    "shenbao_region_network_html_graph.py",
    "shenbao_capture_focus_ego_for_paper.py",
    "shenbao_network_overlap_metrics.py",
    "shenbao_network_overlap_metrics_batch.py",
    "shenbao_network_overlap_metrics_dashboard.py",
    "shenbao_pair_conditioned_keyness.py",
    "shenbao_pair_conditioned_keyness_dashboard_build.py",
]


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT.parent / "shenbao")
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"copy {destination.relative_to(REPO_ROOT)}")


def copy_rules(source_root: Path, repo_root: Path) -> None:
    dictionary_root = source_root / "shenbao_dictionary"
    names = [
        "dictionary_v12_master.csv",
        "stopword_v5_master.csv",
        "variant_char_normalization_rules_v2.csv",
    ]
    runtime_names = [
        "dictionary_annotation_crosswalk_v0.csv",
        "dictionary_exact_merge_optional_v12.csv",
        "dictionary_exact_merge_v12.csv",
        "dictionary_regex_merge_v12.csv",
        "stopword_exact_always_v5.csv",
        "stopword_exact_optional_v5.csv",
        "stopword_regex_always_v5.csv",
        "variant_char_normalization_runtime_v2.csv",
    ]
    for name in names:
        copy_file(dictionary_root / name, repo_root / "config" / "dictionaries" / name)
    for name in runtime_names:
        copy_file(dictionary_root / "runtime_rule_splits" / name, repo_root / "config" / "dictionaries" / "runtime" / name)


def copy_filters(source_root: Path, repo_root: Path) -> None:
    context_root = source_root / "shenbao_filters" / "context"
    token_root = source_root / "shenbao_filters" / "token"
    copy_file(context_root / "filter_context_pre_zhixian_metadata.json", repo_root / "config" / "filters" / "filter_context_pre_zhixian_metadata.json")
    copy_file(token_root / "filter_token_pre_zhixian_retained_official_title.csv", repo_root / "config" / "filters" / "filter_token_pre_zhixian_retained_official_title.csv")
    copy_file(token_root / "filter_token_pre_zhixian_retained_official_title_metadata.json", repo_root / "config" / "filters" / "filter_token_pre_zhixian_retained_official_title_metadata.json")
    copy_file(context_root / "pre_zhixian_revision_20260817" / "token_mask_verification_summary.json", repo_root / "config" / "filters" / "token_mask_verification_summary.json")

    source = context_root / "filter_context_pre_zhixian.csv"
    destination = repo_root / "config" / "filters" / "filter_context_pre_zhixian_uids.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as src, destination.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = ["context_uid", "date", "exclude_reason"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row.get(field, "") for field in fields})


def copy_periodization(source_root: Path, repo_root: Path) -> None:
    revision = source_root / "shenbao_filters" / "context" / "pre_zhixian_revision_20260817"
    counts_source = revision / "long_period_manual_counts_after_revision.csv"
    counts_target = repo_root / "data" / "periodization" / "long_period_manual_counts.csv"
    copy_file(counts_source, counts_target)

    periods_target = repo_root / "config" / "long_period_manual.csv"
    periods_target.parent.mkdir(parents=True, exist_ok=True)
    with counts_source.open("r", encoding="utf-8-sig", newline="") as src, periods_target.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = ["period_id", "start_date", "end_date"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            if row.get("period_id") != "total":
                writer.writerow({field: row.get(field, "") for field in fields})

    pelt_root = source_root / "shenbao_pelt"
    copy_file(
        pelt_root / "rolling30_lixian_xianzheng_xianfa_zhixian_filtered_pre_zhixian_context_values.csv",
        repo_root / "data" / "periodization" / "rolling30_values.csv",
    )
    example = pelt_root / "from19040501_to19150831"
    for suffix in ("change_points.csv", "segments.csv"):
        name = f"pelt_xianfa_from19040501_to19150831_min15_pen5_l2_{suffix}"
        copy_file(example / name, repo_root / "data" / "periodization" / "pelt_example" / name)


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {directory / pattern}, found {len(matches)}")
    return matches[0]


def copy_edges(source_root: Path, repo_root: Path) -> None:
    network_root = source_root / "shenbao_network" / "network_applied"
    global_dir = network_root / "stopfiltered_global_filtered_pre_zhixian_context" / "undirected"
    for window in (1, 5, 10, 20):
        source = find_one(global_dir, f"*strict_all-tokens_w{window}_*.csv")
        copy_file(source, repo_root / "data" / "network" / "edges" / "strict" / "global" / source.name)
    for period in range(6):
        period_id = f"p{period:03d}"
        directory = network_root / f"stopfiltered_long_period_manual_{period_id}_filtered_pre_zhixian_context" / "undirected"
        source = find_one(directory, "*strict_w10*.csv")
        copy_file(source, repo_root / "data" / "network" / "edges" / "strict" / "periods" / period_id / source.name)
    source = find_one(global_dir, "*full_all-tokens_w10_*.csv")
    copy_file(source, repo_root / "data" / "network" / "edges" / "full" / "global" / source.name)
    dynamic_dir = network_root / "for_gephi" / "dynamic" / "long_period_manual"
    source = find_one(dynamic_dir, "for_gephi_dynamic_long_period_manual_full_all-tokens_w10_*_edges.csv")
    copy_file(source, repo_root / "data" / "network" / "edges" / "full" / "periods" / source.name)


def sanitize_csv(source: Path, destination: Path, drop_fields: set[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as src, destination.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = [field for field in (reader.fieldnames or []) if field not in drop_fields]
        writer = csv.DictWriter(dst, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reader)


def copy_network_outputs(source_root: Path, repo_root: Path) -> None:
    metrics_root = source_root / "shenbao_network" / "network_overlap_metrics"
    sanitize_csv(
        metrics_root / "network_core_overlap_metrics_combined.csv",
        repo_root / "data" / "network" / "metrics" / "network_core_overlap_metrics.csv",
        {"source_file"},
    )
    copy_file(metrics_root / "network_overlap_metrics_dashboard.html", repo_root / "dashboards" / "network_overlap_metrics_dashboard.html")
    interpretation = source_root / "shenbao_interpretation" / "focus_anchor_dashboard_ver2"
    copy_file(
        interpretation / "network_summary" / "pair_shared_tables" / "strict_w10_exact_core_only_global_top100_exact2_pair_pos_table.csv",
        repo_root / "data" / "network" / "tables" / "global_pair_shared_tokens.csv",
    )
    copy_file(
        source_root / "shenbao_pelt" / "rolling30_lixian_xianzheng_xianfa_zhixian_filtered_pre_zhixian_context_line_plot_with_periods.png",
        repo_root / "figures" / "figure1_rolling30_with_periods.png",
    )
    copy_file(
        interpretation / "paper_figure" / "focus_multi_ego_strict_w10_global_exact_core_top100.png",
        repo_root / "figures" / "figure2_multi_core_ego_network.png",
    )


def build_multi_core_public(source_root: Path, code_root: Path, repo_root: Path) -> None:
    interpretation = source_root / "shenbao_interpretation" / "focus_anchor_dashboard_ver2"
    input_csv = interpretation / "network_summary" / "region_network_top_neighbors.csv"
    output_html = repo_root / "dashboards" / "multi_core_ego_network_dashboard.html"
    prepared_csv = repo_root / "data" / "network" / "multi_core" / "multi_core_ego_neighbors.csv"
    summary_json = repo_root / "data" / "network" / "multi_core" / "multi_core_ego_dashboard_summary.json"
    output_html.parent.mkdir(parents=True, exist_ok=True)
    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(code_root / "shenbao_region_network_html_graph.py"),
            "--input-csv",
            str(input_csv),
            "--output-html",
            str(output_html),
            "--region-mode",
            "custom",
            "--region-norms",
            "憲政,立憲,憲法,制憲",
            "--profiles",
            "regex-only,strict,full",
            "--period-set-ids",
            "global,long_period_manual",
            "--windows",
            "1,5,10,20",
            "--max-neighbors-per-region",
            "100",
            "--default-topn-neighbors",
            "20",
            "--tokens-root",
            str(source_root / "shenbao_network" / "applied_tokens"),
            "--title",
            "Multi-core ego networks",
            "--ui-mode",
            "core",
            "--hide-data-scope",
            "true",
            "--public-release",
            "true",
            "--prepared-output-csv",
            str(prepared_csv),
            "--summary-json",
            str(summary_json),
        ],
        check=True,
        cwd=code_root,
    )


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return float("-inf")


def build_keyness_outputs(source_root: Path, repo_root: Path) -> None:
    source = source_root / "shenbao_interpretation" / "pair_conditioned_keyness" / "pair_conditioned_keyness_robust_candidates.csv"
    selected: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            key = (row.get("period_id", ""), row.get("pair_id", ""))
            if key not in PAPER_GROUPS:
                continue
            if row.get("token_profile") != "full" or row.get("network_window") != "10":
                continue
            if row.get("candidate_scope") != "shared" or row.get("neighbor_scope") != "top100":
                continue
            if row.get("pair_count_mode") != "min" or row.get("comparison_type") != "same_period_other_pairs":
                continue
            if number(row, "robust_score") < 4:
                continue
            selected[key].append(row)
    missing = [key for key in PAPER_GROUPS if not selected[key]]
    if missing:
        raise RuntimeError(f"No keyness candidates for: {missing}")
    ordered_rows: list[dict[str, str]] = []
    top_rows: list[dict[str, str]] = []
    for key in PAPER_GROUPS:
        rows = sorted(selected[key], key=lambda row: (-number(row, "log_likelihood"), -abs(number(row, "log_odds_z")), row.get("token", "")))
        ordered_rows.extend(rows)
        for rank, row in enumerate(rows[:20], 1):
            top_rows.append({"paper_rank": str(rank), **row})
    output_root = repo_root / "data" / "keyness"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "paper_robust_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)
    with (output_root / "paper_top20.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_rank", *fieldnames])
        writer.writeheader()
        writer.writerows(top_rows)


def write_context_index(repo_root: Path) -> None:
    destination = repo_root / "data" / "evidence" / "paper_context_index.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "title", "article_id", "context_uids", "paper_section", "evidence_terms"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(CONTEXT_INDEX)


def copy_scripts(code_root: Path, repo_root: Path) -> None:
    target = repo_root / "scripts" / "upstream"
    for name in UPSTREAM_SCRIPTS:
        copy_file(code_root / name, target / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(path: Path) -> str:
    first = path.parts[0] if path.parts else "root"
    return {"config": "configuration", "data": "data", "dashboards": "dashboard", "figures": "figure", "scripts": "code", "docs": "documentation", "archive": "archive"}.get(first, "repository")


def write_manifest(repo_root: Path) -> None:
    destination = repo_root / "MANIFEST.csv"
    paths = sorted(
        path for path in repo_root.rglob("*")
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path != destination
        )
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "category", "bytes", "sha256"])
        for path in paths:
            relative = path.relative_to(repo_root)
            writer.writerow([relative.as_posix(), category(relative), path.stat().st_size, sha256(path)])


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    source_root = args.source_root.resolve()
    code_root = args.code_root.resolve()
    repo_root = args.repo_root.resolve()
    if repo_root != REPO_ROOT.resolve():
        print(f"Preparing alternate repository root: {repo_root}")
    copy_rules(source_root, repo_root)
    copy_filters(source_root, repo_root)
    copy_periodization(source_root, repo_root)
    copy_edges(source_root, repo_root)
    copy_network_outputs(source_root, repo_root)
    build_multi_core_public(source_root, code_root, repo_root)
    build_keyness_outputs(source_root, repo_root)
    write_context_index(repo_root)
    copy_scripts(code_root, repo_root)
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "build_keyness_dashboard_lite.py")],
        check=True,
        cwd=repo_root,
    )
    write_manifest(repo_root)
    print(f"Prepared medium release at {repo_root}")


if __name__ == "__main__":
    main()
