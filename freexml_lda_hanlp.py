import pandas as pd
import hanlp
from gensim import corpora, models
import os
import time
import csv
import xml.etree.ElementTree as ET

# 사용자 입력 및 경로 설정
xml_filename = input("Enter xml filename (with .xml): ").strip()
xml_filepath = os.path.join("freexml", xml_filename)
output_dir = os.path.join("freexml", "results_lda")
stopwords_file = "stopwords.txt"

# 불용어 로드
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stop_words = set(line.strip() for line in f if line.strip())

# 사용자 사전(dictionary.txt) dict_force로 적용
force_dict_file = "dictionary_force.txt"

force_dict = set()
if os.path.exists(force_dict_file):
    with open(force_dict_file, 'r', encoding='utf-8') as f:
        force_dict = set(line.strip() for line in f if line.strip())
else:
    print("Warning: no dictionary_force.txt file.")

# 사용자 사전(dictionary.txt) dict_combine으로 적용
combine_dict_file = "dictionary_combine.txt"
combine_dict = set()
if os.path.exists(combine_dict_file):
    with open(combine_dict_file, 'r', encoding='utf-8') as f:
        combine_dict = set(line.strip() for line in f if line.strip())
else:
    print("Warning: no dictionary_combine.txt file.")

# 3) HanLP 토크나이저 초기화
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tokenizer.dict_force = force_dict
tokenizer.dict_combine = combine_dict

def tokenize(text):
    return [word for word in tokenizer(text) if word not in stop_words and len(word) > 1]

# 4) XML 파싱 — <entry> 단위로 author + content 결합
if not os.path.exists(xml_filepath):
    print(f"XML file not found: {xml_filepath}")
    exit(1)

tree = ET.parse(xml_filepath)
root = tree.getroot()

documents = []
authors = []
for entry in root.findall('entry'):
    author_elem = entry.find('author')
    content_elem = entry.find('content')
    author_text = author_elem.text.strip() if author_elem is not None and author_elem.text else ""
    content_text = content_elem.text.strip() if content_elem is not None and content_elem.text else ""
    if author_text or content_text:
        authors.append(author_text)
        documents.append(content_text)

if not documents:
    print("No documents found in the XML file.")
    exit(0)

# 5) 토큰화
tokenized_texts = [tokenize(doc) for doc in documents]

# 6) LDA 모델링
num_topics = 7
start_time = time.time()

# 사전·코퍼스 생성
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# LDA 학습
lda_model = models.LdaModel(
    corpus,
    num_topics=num_topics,
    id2word=dictionary,
    passes=20,
    iterations=500,
    random_state=1938
)

# 7) 토픽별 단어 출력 (TXT)
basename = os.path.splitext(os.path.basename(xml_filepath))[0]
txt_result_file = os.path.join(output_dir, f"{basename}_topic_terms_{num_topics}.txt")
with open(txt_result_file, "w", encoding="utf-8") as f:
    for idx, topic in lda_model.print_topics(-1):
        topic_words = ", ".join([f"{pair.split('*')[1]}, {pair.split('*')[0]}" for pair in topic.split(" + ")])
        f.write(f"Topic {idx}: {topic_words}\n")

# 8) 문서별 토픽 확률 매트릭스 출력 (CSV)
csv_result_file = os.path.join(output_dir, f"{basename}_doc_topic_matrix_{num_topics}.csv")
csv_output = [["no.", "author", "text"] + [f"topic_{i}" for i in range(num_topics)]]
for idx, (auth, doc_bow, raw_text) in enumerate(zip(authors, corpus, documents)):
    doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
    probs = [f"{prob:.4f}" for _, prob in sorted(doc_topics, key=lambda x: x[0])]
    csv_output.append([idx + 1, auth, raw_text] + probs)
        # writer.writerow([idx, auth, raw_text] + probs)
with open(csv_result_file, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerows(csv_output)

print(f"LDA completed in {time.time() - start_time:.2f}s")
print(f"- topic terms: {txt_result_file}")
print(f"- doc–topic matrix: {csv_result_file}")