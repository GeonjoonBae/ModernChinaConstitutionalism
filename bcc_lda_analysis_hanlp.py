import pandas as pd
# import jieba
import hanlp
from gensim import corpora, models
import os
import time
import csv

# 경로 설정
filename = input("Enter input filename (csv file): ")
folder = "bcc_corpus"
data_file = os.path.join(folder, filename)
output_dir = os.path.join(folder, "results_lda")
stopwords_file = "stopwords.txt"

# 불용어 로드
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stop_words = set(line.strip() for line in f)

# 사용자 사전(dictionary.txt) dict_force로 적용
force_dict_file = 'dictionary_force.txt'
if os.path.exists(force_dict_file):
    with open(force_dict_file, 'r', encoding='utf-8') as f:
        force_dict = set(line.strip() for line in f if line.strip())
else:
    user_dict = set()
    print("Warning: no force dictionary file.")

# 사용자 사전(dictionary.txt) dict_combine으로 적용
combine_dict_file = 'dictionary_combine.txt'
if os.path.exists(combine_dict_file):
    with open(combine_dict_file, 'r', encoding='utf-8') as f:
        combine_dict = set(line.strip() for line in f if line.strip())
else:
    user_dict = set()
    print("Warning: no force dictionary file.")

# HanLP 2 딥러닝 토크나이저 로드
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tokenizer.dict_force = force_dict  # 강제 사용자 사전 적용
tokenizer.dict_combine = combine_dict # 결합 사용자 사전 적용

# 데이터 불러오기
df = pd.read_csv(data_file)

# 키워드 리스트 생성 (유일한 keyword 자동 탐색)
keywords = df["keyword"].unique()

# 토큰화 함수 정의
def tokenize(text):
#    return [word for word in jieba.lcut(text) if word not in stop_words and len(word) > 1]
    return [word for word in tokenizer(text) if word not in stop_words and len(word) > 1]

# 전체 실행 시간 기록
total_start_time = time.time()

# 키워드별 실행
for keyword in keywords:
    keyword_start_time = time.time()
    print(f"Processing keyword: {keyword}")

    # 연도 구분 없이 실행
    df_keyword = df[df["keyword"] == keyword]
    text_data = df_keyword["text"].astype(str)

    # LDA 실행 함수
    def run_lda(text_data, keyword, year=None):
        if text_data.empty:
            print(f"{keyword} ({year if year else 'all years'}) - No data found.")
            return

        # 토큰화 및 불용어 제거
        tokenized_text = [tokenize(text) for text in text_data]

        # LDA 모델 생성
        dictionary = corpora.Dictionary(tokenized_text)
        corpus = [dictionary.doc2bow(text) for text in tokenized_text]
        lda_model = models.LdaModel(
            corpus,
            num_topics=15,
            id2word=dictionary,
            passes=20,
            iterations=500,
            random_state=42
        )

        # 결과 저장 파일명 설정
        if year:
            txt_result_file = os.path.join(output_dir, f"{keyword}_topic_terms_matrix_{year}.txt")
            csv_result_file = os.path.join(output_dir, f"{keyword}_text_topic_matrix_{year}.csv")
        else:
            txt_result_file = os.path.join(output_dir, f"{keyword}_topic_terms_matrix.txt")
            csv_result_file = os.path.join(output_dir, f"{keyword}_text_topic_matrix.csv")

        # 토픽별 주요 단어 저장 (TXT)
        with open(txt_result_file, "w", encoding="utf-8") as f:
            for idx, topic in lda_model.print_topics(-1):
                topic_words = ", ".join([f"{pair.split('*')[1]}, {pair.split('*')[0]}" for pair in topic.split(" + ")])
                f.write(f"Topic {idx}: {topic_words}\n")

        # 문서별 토픽 할당 저장 (CSV)
        csv_output = [["no.", "year", "keyword", "text"] + [f"topic {i}" for i in range(15)]]
        for idx, (text, doc_bow) in enumerate(zip(text_data, corpus)):
            doc_topics = lda_model.get_document_topics(doc_bow)
            topic_probs = [''] * 15
            for topic_num, prob in doc_topics:
                topic_probs[topic_num] = f"{prob:.4f}"
            csv_output.append([idx + 1, year if year else "all", keyword, text] + topic_probs)

        with open(csv_result_file, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerows(csv_output)

        print(f"{keyword} ({year if year else 'all years'}) - LDA 완료")

    # 연도 구분 없이 LDA 실행
    run_lda(text_data, keyword)

    # 연도별 실행
    years = df_keyword["year"].unique()
    for year in years:
        df_year = df_keyword[df_keyword["year"] == year]
        run_lda(df_year["text"].astype(str), keyword, year)

    print(f"Keyword {keyword} 완료. 소요 시간: {time.time() - keyword_start_time:.2f} 초")

# 전체 실행 시간 출력
print(f"전체 작업 완료. 총 소요 시간: {time.time() - total_start_time:.2f} 초")
