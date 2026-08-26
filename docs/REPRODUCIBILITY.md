# 재현 절차

## 범위

이 저장소는 『申報』 원문을 재배포하지 않는다. 따라서 공개 데이터만으로 가능한 재현 범위는 다음과 같다.

- 논문 표 1-6의 수치와 정렬 검증
- 공개 edge CSV를 이용한 핵심어 네트워크 유사도 재계산
- 공개 edge CSV를 이용한 논문 조건의 pair-conditioned keyness 재계산
- 그림과 독립형 HTML 대시보드 열람

기사 수집, 문맥 추출, CKIP NLP부터의 완전한 재현에는 상하이도서관 데이터베이스 접근 권한과 비공개 중간 데이터가 필요하다.

## 환경

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 공개 데이터 묶음 재구성

이 저장소와 원 연구 작업 폴더가 같은 상위 디렉터리에 있을 때 다음 명령을 사용한다.

```powershell
python scripts/prepare_release.py
```

다른 위치에 있다면 경로를 지정한다.

```powershell
python scripts/prepare_release.py `
  --source-root C:\path\to\shenbao `
  --code-root C:\path\to\coding
```

## 경량 keyness 대시보드

```powershell
python scripts/build_keyness_dashboard_lite.py
```

출력은 `dashboards/pair_conditioned_keyness_dashboard_lite.html`이다.

## Network overlap 재계산

개별 edge CSV에는 다음 스크립트를 적용할 수 있다.

```powershell
python scripts/upstream/shenbao_network_overlap_metrics.py `
  --input-csv data/network/edges/strict/global/NETWORK_FILE.csv `
  --output-dir reproduced/network_metrics `
  --topn 100
```

정확한 옵션은 `python scripts/upstream/shenbao_network_overlap_metrics.py --help`에서 확인할 수 있다. `scripts/upstream/`의 파일은 원 연구 환경에서 사용한 소스 사본이므로, 일부 기본 경로는 원 작업 폴더를 가리킬 수 있다. 이 저장소에서 실행할 때에는 위 예시처럼 입력과 출력 경로를 명시한다. 논문에 제시된 통합 결과는 `data/network/metrics/network_core_overlap_metrics.csv`에 제공한다.

## Keyness 정렬 검증

`data/keyness/paper_robust_candidates.csv`에서 period-pair별로 다음 순서를 적용한다.

1. `robust_score >= 4`
2. `log_likelihood` 내림차순
3. 동률일 때 `abs(log_odds_z)` 내림차순
4. 상위 20개 선택

그 결과는 `data/keyness/paper_top20.csv`와 일치해야 한다.

## 무결성 확인

`MANIFEST.csv`는 각 공개 파일의 SHA-256을 기록한다. release 준비 스크립트를 다시 실행하면 체크섬도 갱신된다.
