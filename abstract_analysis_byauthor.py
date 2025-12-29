# coding/abstract_analysis_byauthor.py
# -*- coding: utf-8 -*-

"""
- HanLP(FINE_ELECTRA_SMALL_ZH)로 초록(abstract) 토큰화
- 연도별 토큰 분포 → JSD 기반 변화점 탐지 → 자연 분절
- 분절 구간별/전체 빈출 토큰 CSV 저장
- 결과 세트: all(통합), jd(近代史研究), kr(抗日战争研究)
- --author "이름" 옵션 또는 실행 후 터미널에서 입력(공저 포함 매칭)
- dict_force / dict_combine 미지원(또는 None)일 때 안전 가드 처리

입력(상대경로: 현재 작업 디렉토리=coding):
- freexml/articles_jdsyj_2025-08-18.xml
- freexml/articles_krzzyj_2025-08-18.xml
- stopwords.txt
- dictionary_force.txt
- dictionary_combine.txt

출력:
- freexml/articleanalysis_result/ 이하 CSV들
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

# 분석 대상 XML을 두 파일로 한정
XML_FILES = [
    XML_DIR / "articles_jdsyj_2025-08-18.xml",
    XML_DIR / "articles_krzzyj_2025-08-18.xml",
]

STOPWORDS_PATH = CWD / "stopwords.txt"
FORCE_DICT_PATH = CWD / "dictionary_force.txt"
COMBINE_DICT_PATH = CWD / "dictionary_combine.txt"

# ----------------------------
# 인자 + 터미널 입력
# ----------------------------
def norm_name(s: str) -> str:
    return (s or "").strip()

parser = argparse.ArgumentParser()
parser.add_argument("--author", type=str, default=None,
                    help="분석할 특정 저자 이름(예: 杨天石). 공저로 포함된 논문도 포함.")
parser.add_argument("--sigma", type=float, default=1.0,
                    help="변화점 임계치: 평균 + sigma*표준편차 (기본 1.0)")
parser.add_argument("--min_gap_years", type=int, default=2,
                    help="경계 최소 연간격 (기본 2년)")
args, unknown = parser.parse_known_args()

# VS Code 터미널에서 실행 후 저자를 직접 입력하고 싶을 때:
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
# HanLP 로드
# ----------------------------
import hanlp
from hanlp.pretrained.tok import FINE_ELECTRA_SMALL_ZH

tok = hanlp.load(FINE_ELECTRA_SMALL_ZH)  # TransformerTokenizer

def _read_list_file(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

STOPWORDS = set(_read_list_file(STOPWORDS_PATH))
DICT_FORCE = set(_read_list_file(FORCE_DICT_PATH))
DICT_COMBINE = set(_read_list_file(COMBINE_DICT_PATH))

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
        abstract = (art.findtext("abstract") or "").strip()
        keywords = [k.text.strip() for k in art.findall("keywords/keyword")
                    if k.text and k.text.strip()]

        rows.append({
            "title": title,
            "journal": journal,
            "year": year,
            "issue": issue,
            "authors": authors,     # 공저 리스트
            "abstract": abstract,
            "keywords": keywords
        })
    return rows

records = []
for x in XML_FILES:
    records.extend(parse_xml_file(x))
df_all = pd.DataFrame(records)

# ----------------------------
# XML의 모든 키워드를 combine 사전에 추가 (안전 가드)
# ----------------------------
xml_keywords = set()
for kws in df_all["keywords"].dropna():
    for kw in kws:
        if kw:
            xml_keywords.add(kw.strip())

# ----------------------------
# ✅ 예제처럼 "간단 대입"으로 사용자 사전 설정
# ----------------------------
tok.dict_force = DICT_FORCE
tok.dict_combine = set(DICT_COMBINE) | xml_keywords

# ----------------------------
# 특정 저자 필터 (공저 포함)
# ----------------------------
df_all = df_all.drop_duplicates(subset=["title", "journal", "year", "issue"]).reset_index(drop=True)

if TARGET_AUTHOR:
    df_all = df_all[df_all["authors"].apply(
        lambda lst: isinstance(lst, list) and any(norm_name(a) == TARGET_AUTHOR for a in lst)
    )].reset_index(drop=True)
    if df_all.empty:
        print(f"[INFO] 지정한 저자 '{TARGET_AUTHOR}' 에 해당하는 논문이 없습니다.", file=sys.stderr)

# ----------------------------
# 토큰화
# ----------------------------
_LATIN = re.compile(r"[A-Za-z]{2,}")  # 라틴 토큰(2자 이상)

def tokenize(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    toks = tok(text)  # list[str]
    normed = []
    for w in toks:
        w = w.strip()
        if not w:
            continue
        if re.fullmatch(r"[\W_]+", w):
            continue
        if _LATIN.fullmatch(w):
            w = w.lower()
        if w in STOPWORDS:
            continue
        if len(w) == 1:
            continue
        normed.append(w)
    return normed

# ----------------------------
# 연도별 토큰 분포 & JSD 기반 분절
# ----------------------------
from math import log

def build_year_token_counts(df_subset: pd.DataFrame):
    year_tok = defaultdict(Counter)
    for _, r in df_subset.iterrows():
        y = r.get("year")
        if y is None or pd.isna(y):
            continue
        tokens = tokenize(r.get("abstract", ""))
        if tokens:
            year_tok[y].update(tokens)
    return year_tok

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5*(p+q)
    def kl(a, b): return float(np.sum(a * np.log(a/b)))
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def jsd_series(year_tok: dict):
    years = sorted(year_tok.keys())
    if not years:
        return years, []
    union = sorted(set(t for c in year_tok.values() for t in c.keys()))
    eps = 1e-12
    vec = {}
    for y in years:
        counts = np.array([year_tok[y].get(t, 0) for t in union], dtype=float)
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

def count_tokens_for_segment(year_tok, seg_years):
    c = Counter()
    for y in seg_years:
        c.update(year_tok[y])
    return c

# ----------------------------
# 저장 유틸
# ----------------------------
def issue_sort_key(x):
    if x is None or pd.isna(x):
        return (1, 0)
    try:
        return (0, int(x))
    except Exception:
        return (0, 0)

def save_articles(df_subset: pd.DataFrame, name: str):
    out = df_subset.copy()
    out["author"] = out["authors"].apply(lambda lst: "；".join(lst) if isinstance(lst, list) else "")
    out = out.sort_values(
        by=["year", "issue"],
        ascending=[True, True],
        key=lambda s: s.map(issue_sort_key) if s.name == "issue" else s
    )[["author", "title", "journal", "year", "issue", "abstract"]]
    p = OUT_DIR / f"{name}_articles.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    return p

def save_overall_tokens(year_tok, name: str, topn: int = None):
    overall = Counter()
    for y in sorted(year_tok.keys()):
        overall.update(year_tok[y])
    df = pd.DataFrame(overall.most_common(topn), columns=["token", "count"])
    p = OUT_DIR / f"{name}_abstract_overall_tokens.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p

def save_segments_and_keywords(year_tok, segments, name: str, topn: int = 50):
    summary_rows, seg_paths = [], []
    for (ys, ye, seg_years) in segments:
        c = count_tokens_for_segment(year_tok, seg_years)
        seg_df = pd.DataFrame(c.most_common(topn), columns=["token", "count"])
        seg_path = OUT_DIR / f"{name}_abstract_segment_tokens_{ys}_{ye}.csv"
        seg_df.to_csv(seg_path, index=False, encoding="utf-8")
        seg_paths.append(seg_path)
        top_list = [f"{k}({v})" for k, v in c.most_common(15)]
        summary_rows.append({"period": f"{ys}-{ye}", "top_tokens": ", ".join(top_list)})
    sum_df = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / f"{name}_abstract_segments_summary.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8")
    return sum_path, seg_paths

# ----------------------------
# 파이프라인 실행
# ----------------------------
def slug(s: str) -> str:
    if not s:
        return "allauthors"
    return re.sub(r"[^\w\-]+", "_", s)

def run_pipeline(df_input: pd.DataFrame, name: str, sigma: float, min_gap_years: int):
    articles_path = save_articles(df_input, name)

    year_tok = build_year_token_counts(df_input)
    years, jsd_vals = jsd_series(year_tok)
    boundaries = detect_boundaries(jsd_vals, min_gap_years=min_gap_years, sigma=sigma)
    segments = build_segments(years, boundaries) if years else []
    if not segments and years:
        segments = [(years[0], years[-1], years.copy())]

    overall_path = save_overall_tokens(year_tok, name)
    summary_path, seg_paths = save_segments_and_keywords(year_tok, segments, name)

    return {
        "articles_csv": str(articles_path),
        "overall_tokens_csv": str(overall_path),
        "segments_summary_csv": str(summary_path),
        "segment_token_csvs": [str(p) for p in seg_paths],
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

report_all = run_pipeline(df_all_valid, name=f"{author_suffix}_all", sigma=SIGMA, min_gap_years=MIN_GAP)
report_jd  = run_pipeline(df_jd,       name=f"{author_suffix}_jds",  sigma=SIGMA, min_gap_years=MIN_GAP)
report_kr  = run_pipeline(df_kr,       name=f"{author_suffix}_krzz",  sigma=SIGMA, min_gap_years=MIN_GAP)

print("[ALL] segments:", report_all["segments"], "boundaries:", report_all["boundaries"])
print("[JD ] segments:", report_jd["segments"],  "boundaries:", report_jd["boundaries"])
print("[KR ] segments:", report_kr["segments"],  "boundaries:", report_kr["boundaries"])
print("Outputs saved under:", OUT_DIR)
