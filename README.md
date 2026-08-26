# 전산 텍스트 분석을 통해 본 중국 근대 헌정-헌법 논의

## Code and Data for a Computational Text Analysis of Constitutionalism and Constitution-Making in Modern China

이 저장소는 배건준의 논문 「전산 텍스트 분석을 통해 본 중국 근대 헌정-헌법 논의: 『申報』 텍스트에 대한 디지털 역사학적 분석」에서 사용한 집계 데이터, 분석 코드, 표·그림 및 재현 문서를 제공한다. 분석은 『申報』 기사에서 `立憲`, `憲政`, `憲法`, `制憲`을 중심으로 추출한 문맥을 대상으로 한다.

원문 기사와 토큰화된 문맥 전체는 상하이도서관 데이터베이스의 이용 조건을 고려하여 이 저장소에 포함하지 않는다. 대신 논문의 수치와 표를 집계 네트워크 데이터부터 재현할 수 있는 중간 수준의 공개 데이터를 제공한다.

## 논문 정보

- 국문 제목: 전산 텍스트 분석을 통해 본 중국 근대 헌정-헌법 논의: 『申報』 텍스트에 대한 디지털 역사학적 분석
- 영문 제목: Computational Text Analysis of Constitutionalism and Constitution-Making in Modern China: A Digital Historical Analysis of the *Shenbao*
- 저자: 배건준 (Geonjoon Bae)
- 출판 정보: 출판 확정 후 갱신 예정

## 연구 질문

이 연구는 장기간에 걸쳐 『申報』 지면에 등장한 헌정·헌법 논의가 어떠한 어휘적 관계 구조를 형성했는지를 검토한다. 특히 `立憲-憲政`과 `憲法-制憲`의 내부 관계, 두 계열을 연결하는 `憲政-憲法`의 관계, 그리고 이 관계 구조의 시기별 변화를 분석한다.

## 분석 자료

- 수집 기사: 네 핵심어 중 하나 이상을 포함하는 기사 33,513건
- 광고 제외 후 분석 원자료: 26,787건
- 최초 추출 문맥: 38,294개
- `制憲`의 총독 별칭 용례 필터링 후 문맥: 36,923개
- 기본 문맥 범위: 핵심어 2글자와 좌우 50글자를 합한 102자
- NLP 도구: CKIP `bert-base-chinese`
- 토큰 프로필: `regex-only`, `strict`, `full`

기사 수집과 기사 단위 전처리 절차는 별도 저장소 [shlib-shenbao-dataset-workflow](https://github.com/GeonjoonBae/shlib-shenbao-dataset-workflow)에 설명되어 있다.

## 분석 흐름

```text
상하이도서관 『申報』 DB
  -> 네 핵심어 기사 수집·중복 제거
  -> 핵심어 중심 102자 문맥 추출·중첩 문맥 병합
  -> CKIP 단어 분절·품사 태깅·개체명 인식
  -> 이체자 통일·사용자 사전 병합·불용어 제거
  -> 制憲 총독 별칭 용례 필터링
  -> 핵심어 일별 출현량과 long_period_manual 시기 구분
  -> 공기어 네트워크와 핵심어 쌍 유사도
  -> 공유 주변어 pair-conditioned keyness
  -> 중요 문맥의 정성적 독해
```

상세한 처리 단계와 입력·출력 관계는 [DATA_FLOW.md](docs/DATA_FLOW.md)를 참고한다.

## 저장소 구성

```text
ModernChinaConstitutionalism/
├─ README.md
├─ CITATION.cff
├─ MANIFEST.csv
├─ config/
│  ├─ dictionaries/
│  └─ filters/
├─ data/
│  ├─ periodization/
│  ├─ network/
│  ├─ keyness/
│  └─ evidence/
├─ dashboards/
├─ figures/
├─ docs/
├─ scripts/
│  └─ upstream/
└─ archive/
   └─ legacy_scripts/
```

`archive/legacy_scripts/`에는 저장소 개설 초기에 업로드되었으나 이번 논문에는 사용되지 않은 BCC·LDA·TF-IDF 관련 스크립트를 보존한다.

## 논문과 데이터의 대응

| 논문 항목 | 주요 공개 파일 |
| --- | --- |
| 표 1: period별 기사·문맥·핵심어 수 | `data/periodization/long_period_manual_counts.csv` |
| 그림 1: 30일 이동평균과 시기 구분 | `data/periodization/rolling30_values.csv`, `figures/figure1_rolling30_with_periods.png` |
| 그림 2: 전체 시기 핵심어 자아망 | `figures/figure2_multi_core_ego_network.png` |
| 표 2: 핵심어 쌍 공유 주변어 | `data/network/tables/global_pair_shared_tokens.csv` |
| 표 3-5: Jaccard·가중 Jaccard | `data/network/metrics/network_core_overlap_metrics.csv` |
| 표 6: 핵심어 쌍 Keyness Top20 | `data/keyness/paper_top20.csv` |
| 정성 독해 인용문 | `data/evidence/paper_context_index.csv` |

## 주요 분석 조건

네트워크 유사도는 `core exact · strict · top 100`을 기준으로 하며, 전체 시기 비교에는 window 1, 5, 10, 20을, 시기별 비교에는 window 10을 사용한다. 공유 주변어 keyness는 다음 조건을 사용한다.

```text
core exact · full · window 10 · top 100
pair count: min
comparison: same_period_other_pairs
robust score: 4 이상
정렬: log-likelihood 내림차순
동률: |log odds z| 내림차순
```

계산식과 해석 범위는 [METHODS.md](docs/METHODS.md), 열 정의는 [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)에 정리되어 있다.

## 대시보드

- [Network Overlap Metrics Dashboard](dashboards/network_overlap_metrics_dashboard.html)
- [Pair-Conditioned Keyness Dashboard - Paper Lite](dashboards/pair_conditioned_keyness_dashboard_lite.html)

두 파일은 외부 서버 없이 브라우저에서 직접 열 수 있는 독립형 HTML이다. 경량 keyness 대시보드는 논문에서 사용한 여섯 개 period-pair 조건과 robust score 4 이상 후보만 포함한다.

## 재현 방법

저장소 작성에 사용한 로컬 원자료가 있는 경우 다음 명령으로 공개 데이터 묶음을 다시 구성할 수 있다.

```powershell
python scripts/prepare_release.py
```

공개된 집계 데이터만으로 표와 대시보드를 검증하는 절차는 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)를 참고한다.

## 관련 저장소

- [shlib-shenbao-dataset-workflow](https://github.com/GeonjoonBae/shlib-shenbao-dataset-workflow): 상하이도서관 『申報』 텍스트 수집 및 기사 단위 전처리
- [3rdForceNetwork](https://github.com/GeonjoonBae/3rdForceNetwork): 중국근현대사 네트워크 분석 데이터 공개의 선행 사례

## 데이터 이용과 인용

이 저장소는 『申報』 원문이나 전체 토큰열을 재배포하지 않는다. 공개 파일은 연구자가 생성한 집계값, 규칙 사전, 필터 메타데이터, 분석 코드와 시각화 결과로 구성된다. 자세한 범위는 [DATA_ACCESS.md](docs/DATA_ACCESS.md)를 참고한다.

저장소를 이용할 때에는 다음과 같이 인용할 수 있다.

> Bae, Geonjoon. *Code and Data for a Computational Text Analysis of Constitutionalism and Constitution-Making in Modern China*. GitHub repository, 2026. https://github.com/GeonjoonBae/ModernChinaConstitutionalism
