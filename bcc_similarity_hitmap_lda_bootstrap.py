import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hanlp
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from gensim import corpora, models

# 출력 폰트 설정
font_paths = [
    r"C:/Windows/Fonts/times.ttf",
    r"C:/Windows/Fonts/simhei.ttf",
    r"C:/Windows/Fonts/kaiu.ttf"
]
font_names = [font_manager.FontProperties(fname=fp).get_name() for fp in font_paths]
print("폰트 이름 리스트:", font_names)
plt.rcParams['font.family'] = font_names

# 입력
keyword_input = input("Enter keyword (simplified chinese): ").strip()
output_file_keyword = input("Enter output file keyword (english): ").strip()
start_year = input("Enter start year (yyyy): ").strip()
end_year = input("Enter end year (yyyy): ").strip()
sample_size = int(input("Enter sample size for each year: "))
bootstrap = int(input("Enter bootstrap repeat count: "))
use_stopwords = input("Use stopwords? (Y/N): ").strip().upper() == 'Y'
use_userdict = input("Use user dictionary? (Y/N): ").strip().upper() == 'Y'

xml_filepath = r"bcc_corpus\bcc_corpusdata_canzheng_1912-1945.xml"
output_directory = r"bcc_corpus\results_similarity"
stopwords_file = "stopwords.txt"
force_dict_file = 'dictionary_force.txt'
combine_dict_file = 'dictionary_combine.txt'

# 불용어 및 사용자사전
stop_words = set()
if use_stopwords and os.path.exists(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = set(line.strip() for line in f if line.strip())

force_dict = set()
if use_userdict and os.path.exists(force_dict_file):
    with open(force_dict_file, 'r', encoding='utf-8') as f:
        force_dict = set(line.strip() for line in f if line.strip())

combine_dict = set()
if use_userdict and os.path.exists(combine_dict_file):
    with open(combine_dict_file, 'r', encoding='utf-8') as f:
        combine_dict = set(line.strip() for line in f if line.strip())

tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
if use_userdict:
    tokenizer.dict_force = force_dict
    tokenizer.dict_combine = combine_dict

def tokenize(text):
    return [w for w in tokenizer(text) if (not stop_words or w not in stop_words) and len(w) > 1]

# 데이터 로드 및 연도별 entry 분류
tree = ET.parse(xml_filepath)
root = tree.getroot()
entries_by_year = {}
for entry in root.findall('entry'):
    year = entry.find('year').text.strip()
    if not (start_year <= year <= end_year):
        continue
    context = entry.find('context')
    kw = context.find('keyword').text.strip() if context is not None else ''
    if context is not None and kw == keyword_input:
        L = context.find('L').text.strip() if context.find('L') is not None else ""
        R = context.find('R').text.strip() if context.find('R') is not None else ""
        entry_text = L + kw + R
        entries_by_year.setdefault(year, []).append(entry_text)

# 최소 표본수 검증
min_sample = min(len(lst) for lst in entries_by_year.values())
if sample_size > min_sample:
    print(f"Warning: 표본수({sample_size})가 최소 entry({min_sample})보다 많음. {min_sample}로 자동 조정.")
    sample_size = min_sample

years = sorted(entries_by_year.keys())
n_topics = 10

# 부트스트랩 벡터 생성
all_sim_mats = []
for _ in range(bootstrap):
    year_topic_dist = []
    all_docs = []
    year_sampled_docs = []
    for year in years:
        docs = entries_by_year[year]
        sample = random.sample(docs, sample_size) if len(docs) > sample_size else docs
        # 한 연도마다 entry 별로 토큰화, 전체 리스트 저장
        doc_tokens = [tokenize(doc) for doc in sample]
        year_sampled_docs.append(doc_tokens)
        all_docs.extend(doc_tokens)
    # 전체 코퍼스 사전 생성
    dictionary = corpora.Dictionary(all_docs)
    corpus = [dictionary.doc2bow(doc) for doc in all_docs]
    # LDA 모델
    lda = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=15, iterations=400, random_state=0)
    # 연도별 토픽 분포 (연도 내 entry의 토픽 분포 평균)
    start = 0
    for doc_list in year_sampled_docs:
        ndoc = len(doc_list)
        year_corpus = [dictionary.doc2bow(doc) for doc in doc_list]
        topic_dist = np.zeros(n_topics)
        for doc_bow in year_corpus:
            topics = lda.get_document_topics(doc_bow, minimum_probability=0)
            for tid, prob in topics:
                topic_dist[tid] += prob
        topic_dist /= ndoc
        year_topic_dist.append(topic_dist)
        start += ndoc
    sim_mat = cosine_similarity(np.stack(year_topic_dist))
    all_sim_mats.append(sim_mat)

mean_sim_mat = np.mean(all_sim_mats, axis=0)
std_sim_mat = np.std(all_sim_mats, axis=0)

# 저장
result_df = pd.DataFrame(mean_sim_mat, index=years, columns=years)
std_df = pd.DataFrame(std_sim_mat, index=years, columns=years)
result_file = f"bcc_{start_year}-{end_year}_{output_file_keyword}_similarity_lda.csv"
result_df.to_csv(os.path.join(output_directory, result_file), encoding='utf-8-sig')

# 히트맵
plt.figure(figsize=(10, 8))
plt.imshow(mean_sim_mat, cmap='Blues', interpolation='nearest')
plt.xticks(ticks=range(len(years)), labels=years, rotation=45)
plt.yticks(ticks=range(len(years)), labels=years)
plt.colorbar(label='Cosine Similarity')
plt.title(f"Yearly Similarity (LDA, {keyword_input}, {start_year}-{end_year})")
plt.tight_layout()
img_file = f"bcc_{start_year}-{end_year}_{output_file_keyword}_similarity_lda.png"
plt.savefig(os.path.join(output_directory, img_file), dpi=200)
plt.close()
print(f"완료: {result_file}, {img_file}")

# 표준편차 결과 저장
std_file = f"bcc_{start_year}-{end_year}_{output_file_keyword}_similarity_lda_std.csv"
std_df.to_csv(os.path.join(output_directory, std_file), encoding='utf-8-sig')