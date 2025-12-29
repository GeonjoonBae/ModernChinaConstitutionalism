import os
import pandas as pd

def merge_text_files_to_csv(directory, keyword, start_year, end_year):
    files_to_merge = [f"bcc_corpus_{keyword}_{year}.txt" for year in range(start_year, end_year + 1)]
    output_file = f"bcc_corpusraw_{keyword}_{start_year}-{end_year}.csv"
    output_path = os.path.join(directory, output_file)
    
    data = []
    
    for filename in files_to_merge:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8-sig') as infile:
                content = infile.read().strip()
                data.append(content)  
        else:
            print(f"Warning: {filename} not found, skipping.")
    
    df = pd.DataFrame(data, columns=["content"])
    df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=3, escapechar='\\')

# 실행
keyword = input("Enter keyword: ")
start_year = int(input("Enter start year: "))
end_year = int(input("Enter end year: "))
directory = r"bcc_corpus"
merge_text_files_to_csv(directory, keyword, start_year, end_year)