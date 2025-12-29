# coding/abstract_analysis.py
# -*- coding: utf-8 -*-

"""
XML의 논문 초록(abstract)을 대상으로:
1) HanLP(FINE_ELECTRA_SMALL_ZH)로 토큰화(불용어/사용자사전 반영)
2) 연도별 토큰 분포 → JSD 기반 변화점 탐지 → 자연 분절
3) 각 분절 구간별 빈출 토큰/요약, 전체 빈출 토큰
4) 세 가지 세트로 출력: 전체(all), 近代史研究(jd), 抗日战争研究(kr)

입력 위치:
- coding/freexml/*.xml
- coding/stopwords.txt
- coding/dictionary_force.txt
- coding/dictionary_combine.txt

출력 위치:
- coding/freexml/articleanalysis_result/
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re

# ----------------------------
# 경로 설정
# ----------------------------
CWD = Path(".").resolve()  # 현재 작업 디렉토리: coding
XML_DIR = CWD / "freexml"
OUT_DIR = XML_DIR / "articleanalysis_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS_PATH = CWD / "stopwords.txt"
FORCE_DICT_PATH = CWD / "dictionary_force.txt"
COMBINE_DICT_PATH = CWD / "dictionary_combine.txt"

# ----------------------------
# HanLP 로드 (사전/불용어 구성)
# ----------------------------
import hanlp
from hanlp.pretrained.tok import FINE_ELECTRA_SMALL_ZH

tok = hanlp.load(FINE_ELECTRA_SMALL_ZH)  # TransformerTokenizer

def _read_list_file(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        # 빈 줄/주석 제거
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

STOPWORDS = set(_read_list_file(STOPWORDS_PATH))
dict_force = set(_read_list_file(FORCE_DICT_PATH))
dict_combine = set(_read_list_file(COMBINE_DICT_PATH))

# ----------------------------
# XML 파싱
# ----------------------------
def parse_xml_file(path: Path):
    """ freexml 폴더의 단일 XML에서 article 레코드를 추출 """
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
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords
        })
    return rows

# freexml 폴더 내 지정 XML만 불러오기
xml_files = [
    XML_DIR / "articles_jdsyj_2025-08-18.xml",
    XML_DIR / "articles_krzzyj_2025-08-18.xml"
]
for x in xml_files:
    if not x.exists():
        raise FileNotFoundError(f"{x} not found in {XML_DIR}")

records = []
for x in xml_files:
    records.extend(parse_xml_file(x))
df_all = pd.DataFrame(records)

# ----------------------------
# XML의 키워드들을 사용자 사전(combine)으로 추가
# ----------------------------
# 키워드는 'combine' 유형으로 묶어서 분절을 방지
all_keywords = set()
for kws in df_all["keywords"].dropna():
    for kw in kws:
        if kw:
            all_keywords.add(kw.strip())

# HanLP v2 토크나이저는 dict_force/dict_combine 속성을 제공
# - dict_force: 강제 단일 토큰으로 분리
# - dict_combine: 이미 분절된 토큰을 결합해 하나로 유지
# 문자열 집합을 그대로 update
if hasattr(tok, "dict_force"):
    tok.dict_force.update(dict_force)
if hasattr(tok, "dict_combine"):
    tok.dict_combine.update(dict_combine)
    tok.dict_combine.update(all_keywords)

# ----------------------------
# 토큰화 함수
# ----------------------------
_LATIN = re.compile(r"[A-Za-z]{2,}")  # 라틴 토큰 허용(2자 이상)

def tokenize(text: str):
    """HanLP로 토큰화 + 불용어/공백/기호 제거 + 라틴/한중일 혼합 처리"""
    if not isinstance(text, str) or not text.strip():
        return []
    # HanLP 분절
    toks = tok(text)
    normed = []
    for w in toks:
        w = w.strip()
        if not w:
            continue
        # 기호류 거르기
        if re.fullmatch(r"[\W_]+", w):
            continue
        # 라틴은 소문자화
        if _LATIN.fullmatch(w):
            w = w.lower()
        # 불용어 제거
        if w in STOPWORDS:
            continue
        if len(w) == 1:
            continue
        normed.append(w)
    return normed

# ----------------------------
# 유틸: 연도별 토큰 분포, JSD, 변화점, 분절
# ----------------------------
def build_year_token_counts(df_subset: pd.DataFrame):
    """연도별 토큰 카운트 (abstract 대상)"""
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
    """Jensen-Shannon Divergence with smoothing"""
    m = 0.5*(p+q)
    def kl(a, b):  # a * log(a/b)
        return float(np.sum(a * np.log(a/b)))
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def jsd_series(year_tok: dict):
    """연도 정렬 후 인접 연도간 JSD 값 시리즈 반환"""
    years = sorted(year_tok.keys())
    if not years:
        return years, []
    union = sorted(set(t for c in year_tok.values() for t in c.keys()))
    # 확률벡터 구성(라플라스 근사형 스무딩)
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
    # 인접연도 JSD
    jsd_vals = []
    for i in range(1, len(years)):
        y1, y2 = years[i-1], years[i]
        jsd_vals.append((y1, y2, js_divergence(vec[y1], vec[y2])))
    return years, jsd_vals

def detect_boundaries(jsd_vals, min_gap_years: int = 2, sigma: float = 1.0):
    """
    JSD 급변점 탐지:
    - 기준: 평균 + sigma*표준편차
    - 최소 연간격 제약: min_gap_years
    """
    boundaries = []
    if not jsd_vals:
        return boundaries
    vals = np.array([d for _, _, d in jsd_vals], dtype=float)
    thresh = float(vals.mean() + sigma*vals.std()) if len(vals) else float("inf")
    candidates = [y2 for (y1, y2, d) in jsd_vals if d >= thresh]
    if not candidates:
        return boundaries
    candidates = sorted(set(candidates))
    # 최소 간격 보장
    filtered = []
    last = None
    for b in candidates:
        if last is None or (b - last) >= min_gap_years:
            filtered.append(b)
            last = b
    return filtered

def build_segments(years, boundaries):
    """경계를 사용해 [start, end] 구간 리스트 생성"""
    if not years:
        return []
    segments = []
    start = years[0]
    for b in boundaries:
        end = years[years.index(b)-1] if b in years else None
        # start..end (b 직전까지)
        idx_b = years.index(b)
        seg_years = years[years.index(start): idx_b]
        if seg_years:
            segments.append((seg_years[0], seg_years[-1], seg_years.copy()))
        start = b
    # 마지막 구간
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
# 출력 유틸
# ----------------------------
def save_articles(df_subset: pd.DataFrame, name: str):
    """ author,title,journal,year,issue,abstract CSV 저장 (연→호 오름차순, issue None은 뒤) """
    def issue_key(x):
        # None을 뒤로
        return (x is None or pd.isna(x), int(x) if pd.notna(x) else 0)

    out = df_subset.copy()
    out["author"] = out["authors"].apply(lambda lst: "；".join(lst) if isinstance(lst, list) else "")
    out = out.sort_values(
        by=["year", "issue"],
        ascending=[True, True],
        key=lambda s: s.map(issue_key) if s.name == "issue" else s
    )[["author", "title", "journal", "year", "issue", "abstract"]]

    p = OUT_DIR / f"articles_{name}.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    return p

def save_overall_tokens(year_tok, name: str, topn: int = None):
    overall = Counter()
    for y in sorted(year_tok.keys()):
        overall.update(year_tok[y])
    df = pd.DataFrame(overall.most_common(topn), columns=["token", "count"])
    p = OUT_DIR / f"overall_tokens_{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p

def save_segments_and_keywords(year_tok, segments, name: str, topn: int = 50):
    """분절 요약(summary) + 각 구간 토큰 상위 목록 CSV 저장"""
    summary_rows = []
    seg_paths = []
    for (ys, ye, seg_years) in segments:
        c = count_tokens_for_segment(year_tok, seg_years)
        seg_df = pd.DataFrame(c.most_common(topn), columns=["token", "count"])
        seg_path = OUT_DIR / f"segment_tokens_{name}_{ys}_{ye}.csv"
        seg_df.to_csv(seg_path, index=False, encoding="utf-8")
        seg_paths.append(seg_path)
        # 요약(상위 15개 정도)
        top_list = [f"{k}({v})" for k, v in c.most_common(15)]
        summary_rows.append({
            "period": f"{ys}-{ye}",
            "top_tokens": ", ".join(top_list)
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / f"segments_summary_{name}.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8")
    return sum_path, seg_paths

# ----------------------------
# 세트별 분석 파이프라인
# ----------------------------
def run_pipeline(df_input: pd.DataFrame, name: str, sigma: float = 1.0, min_gap_years: int = 2):
    """
    name: 'all', 'jd', 'kr' 등
    sigma: 변화점 임계치(평균 + sigma*표준편차)
    min_gap_years: 경계 최소 간격 (연)
    """
    # 1) 기사 목록 저장
    articles_path = save_articles(df_input, name)

    # 2) 연도별 토큰 분포
    year_tok = build_year_token_counts(df_input)

    # 3) JSD 시리즈 & 변화점
    years, jsd_vals = jsd_series(year_tok)
    boundaries = detect_boundaries(jsd_vals, min_gap_years=min_gap_years, sigma=sigma)
    segments = build_segments(years, boundaries) if years else []

    # 변화점이 하나도 없으면 단일 구간으로
    if not segments and years:
        segments = [(years[0], years[-1], years.copy())]

    # 4) 전체 토큰 / 분절별 토큰 저장
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
# 전체(df_all), 저널별 필터링
# ----------------------------
# 완전 중복 제거(제목·저널·연·호 동일)
df_all = df_all.drop_duplicates(subset=["title", "journal", "year", "issue"]).reset_index(drop=True)

df_all_valid = df_all.copy()
df_jd = df_all_valid[df_all_valid["journal"] == "近代史研究"].copy()
df_kr = df_all_valid[df_all_valid["journal"] == "抗日战争研究"].copy()

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    report_all = run_pipeline(df_all_valid, name="all", sigma=1.0, min_gap_years=2)
    report_jd  = run_pipeline(df_jd,       name="jd",  sigma=1.0, min_gap_years=2)
    report_kr  = run_pipeline(df_kr,       name="kr",  sigma=1.0, min_gap_years=2)

    # 간단 리포트 출력
    print("[ALL] segments:", report_all["segments"], "boundaries:", report_all["boundaries"])
    print("[JD]  segments:", report_jd["segments"],  "boundaries:", report_jd["boundaries"])
    print("[KR]  segments:", report_kr["segments"],  "boundaries:", report_kr["boundaries"])
    print("Outputs saved under:", OUT_DIR)
