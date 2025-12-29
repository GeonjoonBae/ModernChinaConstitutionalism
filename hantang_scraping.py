import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 경로 및 변수 설정
BASE_URL = "https://www.neohytung.com/"
CSV_PREFIX = "ht_xianzheng_text_"
SAVE_CYCLE = 10  # 루프 10번마다 저장

# Chrome 옵션 (백그라운드 실행하려면 --headless 추가)
chrome_options = Options()
# chrome_options.add_argument('--headless')

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)

driver.get(BASE_URL)
time.sleep(2)

# 1. #allcheck 부분 체크박스 클릭 (모두 선택)
allcheck = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#allcheck")))
allcheck.click()
time.sleep(1)

# 2. #TxtKeyword 에 "憲政" 입력
keyword_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#TxtKeyword")))
keyword_input.clear()
keyword_input.send_keys("憲政")
time.sleep(1)

# 3. #BtnExact 클릭
btn_exact = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#BtnExact")))
btn_exact.click()
time.sleep(2)

results = []
file_count = 1
page_loop = 0

while True:
    for i in range(1, 6):  # 1~5번째 tr 반복
        try:
            derivation = driver.find_element(By.CSS_SELECTOR, f"#DataList > tbody > tr:nth-child({i}) > td > div > div.derivation").text.strip()
            summary = driver.find_element(By.CSS_SELECTOR, f"#DataList > tbody > tr:nth-child({i}) > td > div > div.summary").text.strip()
            results.append({"source": derivation, "content": summary})
        except Exception as e:
            # 해당 tr이 없으면 무시
            continue

    page_loop += 1

    # 10번 루틴마다 저장
    if page_loop % SAVE_CYCLE == 0:
        df = pd.DataFrame(results)
        df.to_csv(f"{CSV_PREFIX}{file_count}.csv", index=False, encoding='utf-8-sig')
        print(f"Saved: {CSV_PREFIX}{file_count}.csv")
        file_count += 1
        results = []  # 누적된 결과 초기화

    # 다음 페이지 이동: #pagerFrm2 > ul > li:nth-child(15) > a
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "#pagerFrm2 > ul > li:nth-child(15) > a")
        driver.execute_script("arguments[0].scrollIntoView();", next_btn)
        next_btn.click()
        time.sleep(5)
    except Exception as e:
        print("No next page button found. Exiting.")
        break

# 남은 데이터 저장
if results:
    df = pd.DataFrame(results)
    df.to_csv(f"{CSV_PREFIX}{file_count}.csv", index=False, encoding='utf-8-sig')
    print(f"Saved: {CSV_PREFIX}{file_count}.csv")

driver.quit()
