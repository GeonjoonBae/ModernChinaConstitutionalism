import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from opencc import OpenCC
import hanlp

# ---------------------------
# 사용자 입력 (필수)
# ---------------------------
year_input = input("Enter year (yyyy): ").strip()
keyword_input = input("Enter keyword (simplified chinese): ").strip()
output_file_keyword = input("Enter output file keyword (english): ").strip()

# 최종 파일명 생성
output_filename = f"bcc_{year_input}_{output_file_keyword}_tfidf.csv"
print("Final output file name:", output_filename)

# XML 파일 경로 (수정 필요 시 실제 경로로 변경)
#xml_filepath = r"bcc_corpus\bcc_corpusdata_constitution_1872-1945.xml"
xml_filepath = r"bcc_corpus\bcc_corpusdata_constitution_1946-1949.xml"

# 저장할 경로
output_directory = "results_tfidf"

# XML 파일 존재 여부 확인
if not os.path.exists(xml_filepath):
    print(f"XML file not found: {xml_filepath}")
else:
    # XML 파싱
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # 추출한 문서를 저장할 리스트
    documents = []

    # 각 entry를 순회하면서 조건에 맞는 항목의 <L>, <keyword>, <R> 텍스트 추출
    for entry in root.findall('entry'):
        entry_year = entry.find('year')
        context = entry.find('context')
        if entry_year is not None and context is not None:
            # <keyword>는 context 하위에 위치
            context_keyword = context.find('keyword')
            if context_keyword is not None:
                # 조건: 입력한 year와 keyword가 일치
                if entry_year.text.strip() == year_input and context_keyword.text.strip() == keyword_input:
                    # 추출: <L>, <keyword>, <R>의 텍스트
                    L_elem = context.find('L')
                    R_elem = context.find('R')
                    # 각 요소가 존재하면 텍스트 추출, 없으면 빈 문자열 처리
                    text_L = L_elem.text.strip() if L_elem is not None and L_elem.text else ""
                    text_keyword = context_keyword.text.strip()  # 이미 확인됨
                    text_R = R_elem.text.strip() if R_elem is not None and R_elem.text else ""
                    # 문서 생성: 세 문자열을 이어붙임
                    document = text_L + text_keyword + text_R
                    documents.append(document)

    if not documents:
        print(f"No entries found for year {year_input} with keyword '{keyword_input}'.")
    else:
        # 데이터프레임 생성 (한 행=하나의 문서)
        df = pd.DataFrame({'title': documents})
        
        # 번체→간체 변환
        cc = OpenCC('t2s')
        df["title"] = df["title"].astype(str).apply(lambda x: cc.convert(x))
        
        # HanLP의 FINE_ELECTRA_SMALL_ZH 토크나이저 모델 사용
        tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        # 각 문서에 대해 토큰화 후, 공백으로 결합하여 문자열로 변환
        tokenized_text = [' '.join(tokenizer(text)) for text in df["title"].tolist()]
        
        # TF-IDF 분석 수행
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_text)
        
        # 어휘 및 TF-IDF 점수 계산
        terms = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # 단어별 TF와 IDF 계산
        tf = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()
        idf = vectorizer.idf_
        
        # 정규분포 값 계산 (TF-IDF 점수에 대한 확률 밀도)
        mean_val = np.mean(tfidf_scores)
        std_val = np.std(tfidf_scores)
        normdis_values = norm.pdf(tfidf_scores, mean_val, std_val)
        
        # 결과 DataFrame 생성
        result_df = pd.DataFrame({
            "term": terms,
            "tf": tf,
            "idf": idf,
            "tf-idf": tfidf_scores,
            "normdis": normdis_values
        })
        
        # TF-IDF 값 기준 내림차순 정렬 후, 순위(rank) 열 추가
        result_df = result_df.sort_values(by="tf-idf", ascending=False).reset_index(drop=True)
        result_df["rank"] = result_df.index + 1
        
        # 결과 파일 경로
        output_filepath = os.path.join(output_directory, output_filename)
        
        # 결과 CSV 파일 저장 (utf-8-sig 인코딩)
        result_df.to_csv(output_filepath, index=False, encoding="utf-8-sig")
        print(f"TF-IDF analysis completed. Results saved to: {output_filepath}")
