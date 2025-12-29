import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import hanlp

# 사용자 입력
keyword_input = input("Enter keyword (simplified chinese): ").strip()
input_filename = input("Enter filename (with .xml): ").strip()
output_file_keyword = input("Enter output file keyword (english): ").strip()

# 파일 경로
xml_filepath = os.path.join("bcc_corpus", input_filename)
output_directory = r"bcc_corpus\results_tfidf"
stopwords_file = "stopwords.txt"
force_dict_file = 'dictionary_force.txt'
combine_dict_file = 'dictionary_combine.txt'

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

    years = sorted(set(
        entry.find('year').text.strip()
        for entry in root.findall('entry')
        if entry.find('year') is not None
    ))

    found_any = False
    all_documents = []   # 전체 연도 데이터 저장용

    def tokenize(text):
        # 토큰화 + 불용어 제거 + 1자 이하 제외
        return [word for word in tokenizer(text) if word not in stop_words and len(word) > 1]

    for year in years:
        documents = []
        for entry in root.findall('entry'):
            entry_year = entry.find('year')
            context = entry.find('context')
            if entry_year is not None and context is not None:
                context_keyword = context.find('keyword')
                if context_keyword is not None:
                    # 연도 및 키워드로 필터링
                    if entry_year.text.strip() == year and context_keyword.text.strip() == keyword_input:
                        L_elem = context.find('L')
                        R_elem = context.find('R')
                        text_L = L_elem.text.strip() if L_elem is not None and L_elem.text else ""
                        text_keyword = context_keyword.text.strip()
                        text_R = R_elem.text.strip() if R_elem is not None and R_elem.text else ""
                        document = text_L + text_keyword + text_R
                        documents.append(document)
                        all_documents.append(document)  # 전체 데이터에도 추가

        if not documents:
            print(f"[{year}] No entries found for keyword '{keyword_input}'.")
            continue

        # 데이터프레임 생성 및 전처리
        df = pd.DataFrame({'title': documents})
        tokenized_text = [' '.join(tokenize(text)) for text in df["title"].tolist()]

        # TF-IDF 분석
        tfvectorizer = TfidfVectorizer(norm="l1")
        counter = CountVectorizer()
        tfidf_matrix = tfvectorizer.fit_transform(tokenized_text)
        count_matrix = counter.fit_transform(tokenized_text)
        terms = tfvectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        docf = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()
        count = np.asarray(count_matrix.sum(axis=0)).flatten()
        idf = tfvectorizer.idf_
        mean_val = np.mean(tfidf_scores)
        std_val = np.std(tfidf_scores)
        normdis_values = norm.pdf(tfidf_scores, mean_val, std_val)

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

        # 연도별 결과 파일명
        output_filename = f"bcc_{year}_{output_file_keyword}_tfidf_norml1.csv"
        output_filepath = os.path.join(output_directory, output_filename)
        result_df.to_csv(output_filepath, index=False, encoding="utf-8-sig")
        print(f"[{year}] TF-IDF analysis completed. Results saved to: {output_filepath}")
        found_any = True

    # 전체 연도 대상 분석
    if all_documents:
        df_all = pd.DataFrame({'title': all_documents})
        tokenized_text_all = [' '.join(tokenize(text)) for text in df_all["title"].tolist()]
        tfvectorizer = TfidfVectorizer(norm="l1")
        counter = CountVectorizer()
        tfidf_matrix_all = tfvectorizer.fit_transform(tokenized_text_all)
        count_matrix_all = counter.fit_transform(tokenized_text_all)
        terms_all = tfvectorizer.get_feature_names_out()
        tfidf_scores_all = tfidf_matrix_all.sum(axis=0).A1
        docf_all = np.asarray((tfidf_matrix_all > 0).sum(axis=0)).flatten()
        count_all = np.asarray(count_matrix_all.sum(axis=0)).flatten()
        idf_all = tfvectorizer.idf_
        mean_val_all = np.mean(tfidf_scores_all)
        std_val_all = np.std(tfidf_scores_all)
        normdis_values_all = norm.pdf(tfidf_scores_all, mean_val_all, std_val_all)

        result_df_all = pd.DataFrame({
            "term": terms_all,
            "docf": docf_all,
            "count": count_all,
            "idf": idf_all,
            "tf-idf": tfidf_scores_all,
            "normdis": normdis_values_all
        })
        result_df_all = result_df_all.sort_values(by="tf-idf", ascending=False).reset_index(drop=True)
        result_df_all["rank"] = result_df_all.index + 1

        # 전체 결과 파일명
        output_filename_all = f"bcc_{output_file_keyword}_tfidf_norml1.csv"
        output_filepath_all = os.path.join(output_directory, output_filename_all)
        result_df_all.to_csv(output_filepath_all, index=False, encoding="utf-8-sig")
        print(f"전체 연도 TF-IDF analysis completed. Results saved to: {output_filepath_all}")

    if not found_any:
        print("No data found for the keyword in any year.")