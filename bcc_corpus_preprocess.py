import os
import pandas as pd
import xml.etree.ElementTree as ET

def process_corpus_csv(input_filename):
    directory = r"bcc_corpus"
    input_filename = input_filename + ".csv" 
    input_path = os.path.join(directory, input_filename)
    output_filename = input_filename.replace("corpusraw", "corpusdata")
    output_path_csv = os.path.join(directory, output_filename)
    output_path_xml = os.path.join(directory, output_filename.replace(".csv", ".xml"))
    
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    # content 열의 이름을 kwic으로 변경
    df.rename(columns={'content': 'kwic'}, inplace=True)
    
    # source와 year 추출
    df[['source', 'year']] = df['kwic'].str.extract(r'<B>(.*?)\s(\d{4})</B>')
    df['kwic'] = df['kwic'].str.replace(r'<B>.*?</B>', '', regex=True).str.strip()
    
    # 새로운 열 추가
    df['text'] = df['kwic'].str.replace(r'<U>|</U>', '', regex=True)
    df['L-context'] = df['kwic'].str.extract(r'(.*)<U>')
    df['keyword'] = df['kwic'].str.extract(r'<U>(.*?)</U>')
    df['R-context'] = df['kwic'].str.extract(r'</U>(.*)')
    
    # 열 순서 조정
    df = df[['source', 'year', 'kwic', 'text', 'L-context', 'keyword', 'R-context']]
    
    # CSV 저장
    df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
    
    # XML 생성
    root = ET.Element("corpus")
    
    for index, row in df.iterrows():
        entry = ET.SubElement(root, "entry", id=f"{row['year']}_{index+1}")
        ET.SubElement(entry, "source").text = row['source']
        ET.SubElement(entry, "year").text = str(row['year'])
        context = ET.SubElement(entry, "context")
        ET.SubElement(context, "L").text = row['L-context'] if pd.notna(row['L-context']) else ""
        ET.SubElement(context, "keyword").text = row['keyword'] if pd.notna(row['keyword']) else ""
        ET.SubElement(context, "R").text = row['R-context'] if pd.notna(row['R-context']) else ""
    
    # XML 들여쓰기 적용
    ET.indent(root)
    
    # XML 저장
    tree = ET.ElementTree(root)
    tree.write(output_path_xml, encoding='utf-8', xml_declaration=True)

# 실행
input_filename = input("Enter input filename (csv file): ")
process_corpus_csv(input_filename)