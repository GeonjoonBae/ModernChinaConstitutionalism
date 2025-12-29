import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import hanlp

# 사용자 입력
xml_filename = input("Enter file name(with .xml): ").strip()

# 파일 경로
xml_filepath = os.path.join("freexml", xml_filename)
output_directory = r"freexml/results_tfidf"
stopwords_file = "stopwords.txt"
force_dict_file = "dictionary_force.txt"
combine_dict_file = "dictionary_combine.txt"

# 불용어 로드
if os.path.exists(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = set(line.strip() for line in f if line.strip())
else:
    stop_words = set()
    print("Warning: no stopwords.txt file.")

# dict_force 사용자 사전 로드
if os.path.exists(force_dict_file):
    with open(force_dict_file, 'r', encoding='utf-8') as f:
        force_dict = set(line.strip() for line in f if line.strip())
else:
    force_dict = set()
    print("Warning: no dictionary_force.txt file.")

# dict_combine 사용자 사전 로드
if os.path.exists(combine_dict_file):
    with open(combine_dict_file, 'r', encoding='utf-8') as f:
        combine_dict = set(line.strip() for line in f if line.strip())
else:
    combine_dict = set()
    print("Warning: no dictionary_combine.txt file.")

# HanLP 토크나이저 및 사용자 사전 적용
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tokenizer.dict_force = force_dict
tokenizer.dict_combine = combine_dict

# XML 파일 존재 여부 확인
if not os.path.exists(xml_filepath):
    print(f"XML file not found: {xml_filepath}")
else:
    tree = ET.parse(xml_filepath)
    root = tree.getroot()


    def tokenize(text):
        # 토큰화 + 불용어 제거 + 1자 이하 제외
        return [word for word in tokenizer(text) if word not in stop_words and len(word) > 1]

    # 모든 <entry> 요소에서 author와 content 추출
    documents = []
    for entry in root.findall('entry'):
        author_elem = entry.find('author')
        content_elem = entry.find('content')
        author_text = author_elem.text.strip() if author_elem is not None and author_elem.text else ""
        content_text = content_elem.text.strip() if content_elem is not None and content_elem.text else ""
        if author_text or content_text:
            documents.append(f"{author_text} {content_text}")

    if not documents:
        print("No documents found in the XML file.")
        exit(0)

# 데이터 전처리 및 토큰화
tokenized_texts = [' '.join(tokenize(doc)) for doc in documents]

# TF-IDF 및 Count 벡터라이저 설정
tfvectorizer = TfidfVectorizer(norm="l1")
counter = CountVectorizer()

tfidf_matrix = tfvectorizer.fit_transform(tokenized_texts)
count_matrix = counter.fit_transform(tokenized_texts)

terms = tfvectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
docf = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()
count = np.asarray(count_matrix.sum(axis=0)).flatten()
idf = tfvectorizer.idf_
mean_val = np.mean(tfidf_scores)
std_val = np.std(tfidf_scores)
normdis_values = norm.pdf(tfidf_scores, mean_val, std_val)

# 결과 DataFrame 생성
result_df = pd.DataFrame({
    "term": terms,
    "docf": docf,
    "count": count,
    "idf": idf,
    "tf-idf": tfidf_scores,
    "normdis": normdis_values
})
result_df = result_df.sort_values(by="tf-idf", ascending=False).reset_index(drop=True)
result_df["rank"] = result_df.index + 1

# 출력 파일 경로 및 저장
basename = os.path.splitext(os.path.basename(xml_filepath))[0]
output_filename = f"{basename}_tfidf_norml1.csv"
output_filepath = os.path.join(output_directory, output_filename)
result_df.to_csv(output_filepath, index=False, encoding="utf-8-sig")

print(f"TF-IDF analysis completed. Results saved to: {output_filepath}")