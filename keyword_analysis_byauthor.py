# coding/author_keywords_pipeline.py
# -*- coding: utf-8 -*-

"""
저자별 주요 키워드 분석(빠른 버전)
- XML의 <keywords><keyword>만 사용하여 연도별 키워드 분포 구성
- 인접 연도 JSD로 변화점 탐지 → 자연 분절
- 세트별(통합 all, 近代史研究 jd, 抗日战争研究 kr)로
  1) 전체 키워드 빈도 CSV
  2) 분절 요약 CSV
  3) 분절 구간별 상위 키워드 CSV
- 논문 목록은 CSV 대신, 요청 형식의 TXT로 저장:
  "<author>，<author>： 《<title>》，《<journal>》，<year>年第<issue>期。"

입력(상대경로: 현재 작업 디렉토리=coding):
- freexml/articles_jdsyj_2025-08-18.xml
- freexml/articles_krzzyj_2025-08-18.xml

출력:
- freexml/articleanalysis_result/ 이하 CSV/TXT
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import argparse
import re
import sys

# ----------------------------
# 경로
# ----------------------------
CWD = Path(".").resolve()               # 현재 작업 디렉토리: coding
XML_DIR = CWD / "freexml"
OUT_DIR = XML_DIR / "articleanalysis_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 분석 대상 XML(두 파일로 한정)
XML_FILES = [
    XML_DIR / "articles_jdsyj_2025-08-18.xml",
    XML_DIR / "articles_krzzyj_2025-08-18.xml",
]

# ----------------------------
# 인자 + 터미널 입력
# ----------------------------
def norm_name(s: str) -> str:
    return (s or "").strip()

parser = argparse.ArgumentParser()
parser.add_argument("--author", type=str, default=None,
                    help="분석할 특정 저자 이름(예: 杨天石). 공저 포함.")
parser.add_argument("--sigma", type=float, default=1.0,
                    help="변화점 임계치: 평균 + sigma*표준편차 (기본 1.0)")
parser.add_argument("--min_gap_years", type=int, default=2,
                    help="경계 최소 연간격 (기본 2년)")
args, _ = parser.parse_known_args()

# VS Code에서 저자를 터미널로 입력받기(미지정 시)
if args.author is None:
    try:
        user_in = input("분석할 저자명을 입력하세요(빈칸이면 전체 분석): ").strip()
    except EOFError:
        user_in = ""
    args.author = user_in if user_in else None

TARGET_AUTHOR = norm_name(args.author) if args.author else None
SIGMA = args.sigma
MIN_GAP = args.min_gap_years

# ----------------------------
# 파일 존재 확인
# ----------------------------
for x in XML_FILES:
    if not x.exists():
        print(f"[ERROR] XML not found: {x}", file=sys.stderr)
        sys.exit(1)

# ----------------------------
# XML 파싱
# ----------------------------
def parse_xml_file(path: Path):
    rows = []
    tree = ET.parse(str(path))
    root = tree.getroot()
    for art in root.findall("article"):
        title = (art.findtext("title") or "").strip()
        journal = (art.findtext("journal") or "").strip() or None
        year_txt = (art.findtext("year") or "").strip()
        issue_txt = (art.findtext("issue") or "").strip()

        try:
            year = int(year_txt)
        except Exception:
            year = None
        try:
            issue = int(issue_txt)
        except Exception:
            issue = None

        authors = [a.text.strip() for a in art.findall("authors/author")
                   if a.text and a.text.strip()]
        # abstract는 이번 파이프라인에서 사용하지 않음
        keywords = [k.text.strip() for k in art.findall("keywords/keyword")
                    if k.text and k.text.strip()]

        rows.append({
            "title": title,
            "journal": journal,
            "year": year,
            "issue": issue,
            "authors": authors,     # 공저 리스트
            "keywords": keywords
        })
    return rows

records = []
for x in XML_FILES:
    records.extend(parse_xml_file(x))
df_all = pd.DataFrame(records)

# 완전 중복 제거(제목·저널·연·호 동일)
df_all = df_all.drop_duplicates(subset=["title", "journal", "year", "issue"]).reset_index(drop=True)

# ----------------------------
# 저자 필터(공저 포함) — 저자별 분석 틀
# ----------------------------
if TARGET_AUTHOR:
    df_all = df_all[df_all["authors"].apply(
        lambda lst: isinstance(lst, list) and any(norm_name(a) == TARGET_AUTHOR for a in lst)
    )].reset_index(drop=True)
    if df_all.empty:
        print(f"[INFO] 지정한 저자 '{TARGET_AUTHOR}' 에 해당하는 논문이 없습니다.", file=sys.stderr)

# ----------------------------
# 논문 목록 TXT 저장 (요청 형식)
# ----------------------------
def issue_sort_key(x):
    if x is None or pd.isna(x):
        return (1, 0)
    try:
        return (0, int(x))
    except Exception:
        return (0, 0)

def format_entry_txt(authors, title, journal, year, issue):
    # 저자: '杨天石' 또는 '杨天石，王学庄'
    author_str = "，".join(authors) if isinstance(authors, list) else str(authors or "")
    # 본문 형식: <authors>：《<title>》，《<journal>》，<year>年第<issue>期。
    # 결측값 처리
    parts = []
    parts.append(f"{author_str}：《{title}》")
    if journal:
        parts.append(f"《{journal}》")
    # 연·호
    if year is not None and not pd.isna(year):
        if issue is not None and not pd.isna(issue):
            parts.append(f"{int(year)}年第{int(issue)}期。")
        else:
            parts.append(f"{int(year)}年。")
    else:
        # year가 없으면 마침표만
        parts.append("。")
    # 쉼표/간격
    # 예시와 동일하게 쉼표(，)로 연결
    # "《제목》, 《저널》, 1995年第5期。" → 전각 쉼표 사용
    # 여기서는 이미 리스트에 넣었으니 join으로 결합
    # 제목과 저널 사이, 저널과 연호 사이에 전각 쉼표를 넣음
    if len(parts) >= 2:
        return "，".join(parts[:-1]) + parts[-1]  # 마지막은 문장부호 포함
    else:
        return parts[0]

def save_article_list_txt(df_subset: pd.DataFrame, name: str):
    # 연→호 오름차순(호 없음은 뒤)
    df_sorted = df_subset.sort_values(
        by=["year", "issue"],
        ascending=[True, True],
        key=lambda s: s.map(issue_sort_key) if s.name == "issue" else s
    )
    lines = []
    for _, r in df_sorted.iterrows():
        line = format_entry_txt(r["authors"], r["title"], r["journal"], r["year"], r["issue"])
        lines.append(line)
    p = OUT_DIR / f"{name}_articles_list.txt"
    with p.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    return p

# ----------------------------
# 연도별 "키워드" 분포 & JSD 기반 분절
# ----------------------------
def build_year_keyword_counts(df_subset: pd.DataFrame):
    year_kw = defaultdict(Counter)
    for _, r in df_subset.iterrows():
        y = r.get("year")
        if y is None or pd.isna(y):
            continue
        kws = r.get("keywords") or []
        # 빈/공백 키워드 제거
        clean = [k.strip() for k in kws if isinstance(k, str) and k.strip()]
        if clean:
            year_kw[y].update(clean)
    return year_kw

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5*(p+q)
    def kl(a, b): return float(np.sum(a * np.log(a/b)))
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def jsd_series(year_kw: dict):
    years = sorted(year_kw.keys())
    if not years:
        return years, []
    vocab = sorted(set(t for c in year_kw.values() for t in c.keys()))
    eps = 1e-12
    vec = {}
    for y in years:
        counts = np.array([year_kw[y].get(t, 0) for t in vocab], dtype=float)
        if counts.sum() == 0:
            p = np.ones_like(counts) / len(counts)
        else:
            p = counts / counts.sum()
        p = (p + eps) / (p.sum() + eps*len(p))
        vec[y] = p
    jsd_vals = []
    for i in range(1, len(years)):
        y1, y2 = years[i-1], years[i]
        jsd_vals.append((y1, y2, js_divergence(vec[y1], vec[y2])))
    return years, jsd_vals

def detect_boundaries(jsd_vals, min_gap_years: int = 2, sigma: float = 1.0):
    if not jsd_vals:
        return []
    vals = np.array([d for _, _, d in jsd_vals], dtype=float)
    thresh = float(vals.mean() + sigma*vals.std()) if len(vals) else float("inf")
    candidates = [y2 for (y1, y2, d) in jsd_vals if d >= thresh]
    if not candidates:
        return []
    candidates = sorted(set(candidates))
    filtered, last = [], None
    for b in candidates:
        if last is None or (b - last) >= min_gap_years:
            filtered.append(b)
            last = b
    return filtered

def build_segments(years, boundaries):
    if not years:
        return []
    segments = []
    start = years[0]
    for b in boundaries:
        idx_b = years.index(b)
        seg_years = years[years.index(start): idx_b]
        if seg_years:
            segments.append((seg_years[0], seg_years[-1], seg_years.copy()))
        start = b
    seg_years = years[years.index(start):]
    if seg_years:
        segments.append((seg_years[0], seg_years[-1], seg_years.copy()))
    return segments

def count_keywords_for_segment(year_kw, seg_years):
    c = Counter()
    for y in seg_years:
        c.update(year_kw[y])
    return c

# ----------------------------
# 저장 유틸
# ----------------------------
def save_overall_keywords(year_kw, name: str, topn: int = None):
    overall = Counter()
    for y in sorted(year_kw.keys()):
        overall.update(year_kw[y])
    df = pd.DataFrame(overall.most_common(topn), columns=["keyword", "count"])
    p = OUT_DIR / f"{name}_overall_keywords.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p

def save_segments_and_keywords(year_kw, segments, name: str, topn: int = 50):
    summary_rows, seg_paths = [], []
    for (ys, ye, seg_years) in segments:
        c = count_keywords_for_segment(year_kw, seg_years)
        seg_df = pd.DataFrame(c.most_common(topn), columns=["keyword", "count"])
        seg_path = OUT_DIR / f"{name}_keywords_segment_{ys}_{ye}.csv"
        seg_df.to_csv(seg_path, index=False, encoding="utf-8")
        seg_paths.append(seg_path)
        top_list = [f"{k}({v})" for k, v in c.most_common(15)]
        summary_rows.append({"period": f"{ys}-{ye}", "top_keywords": ", ".join(top_list)})
    sum_df = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / f"{name}_keywords_segments_summary.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8")
    return sum_path, seg_paths

# ----------------------------
# 파이프라인 실행 (키워드 기반)
# ----------------------------
def slug(s: str) -> str:
    if not s:
        return "allauthors"
    return re.sub(r"[^\w\-]+", "_", s)

def run_pipeline_keywords(df_input: pd.DataFrame, name: str, sigma: float, min_gap_years: int):
    # 논문 목록 TXT 저장
    articles_txt_path = save_article_list_txt(df_input, name)

    # 연도별 키워드 분포
    year_kw = build_year_keyword_counts(df_input)
    years, jsd_vals = jsd_series(year_kw)
    boundaries = detect_boundaries(jsd_vals, min_gap_years=min_gap_years, sigma=sigma)
    segments = build_segments(years, boundaries) if years else []
    if not segments and years:
        segments = [(years[0], years[-1], years.copy())]

    # 전체/분절 저장
    overall_path = save_overall_keywords(year_kw, name)
    summary_path, seg_paths = save_segments_and_keywords(year_kw, segments, name)

    return {
        "articles_txt": str(articles_txt_path),
        "overall_keywords_csv": str(overall_path),
        "segments_summary_csv": str(summary_path),
        "segment_keyword_csvs": [str(p) for p in seg_paths],
        "years": years,
        "boundaries": boundaries,
        "segments": [(ys, ye) for (ys, ye, _) in segments]
    }

# ----------------------------
# 세트 생성 및 실행
# ----------------------------
author_suffix = slug(TARGET_AUTHOR)
df_all_valid = df_all.copy()
df_jd = df_all_valid[df_all_valid["journal"] == "近代史研究"].copy()
df_kr = df_all_valid[df_all_valid["journal"] == "抗日战争研究"].copy()

report_all = run_pipeline_keywords(df_all_valid, name=f"{author_suffix}_all", sigma=SIGMA, min_gap_years=MIN_GAP)
report_jd  = run_pipeline_keywords(df_jd,       name=f"{author_suffix}_jds",  sigma=SIGMA, min_gap_years=MIN_GAP)
report_kr  = run_pipeline_keywords(df_kr,       name=f"{author_suffix}_krzz",  sigma=SIGMA, min_gap_years=MIN_GAP)

print("[ALL] segments:", report_all["segments"], "boundaries:", report_all["boundaries"])
print("[JD ] segments:", report_jd["segments"],  "boundaries:", report_jd["boundaries"])
print("[KR ] segments:", report_kr["segments"],  "boundaries:", report_kr["boundaries"])
print("Outputs saved under:", OUT_DIR)
