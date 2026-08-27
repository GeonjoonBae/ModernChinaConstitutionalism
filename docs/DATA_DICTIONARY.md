# 공개 데이터 사전

## Periodization

### `long_period_manual_counts.csv`

- `period_id`: 시기 식별자
- `start_date`, `end_date`: 시기 시작일과 종료일
- `article_count`: 필터 적용 후 해당 시기에 남은 고유 기사 수
- `context_count`: 필터 적용 후 문맥 수
- `立憲`, `憲政`, `憲法`, `制憲`: regex-only 프로필에서 핵심어와 정확히 일치하는 토큰의 출현 횟수

### `rolling30_values.csv`

- `date`: 날짜
- 핵심어별 count 열: 일별 exact token 출현량
- 핵심어별 rolling 열: 30일 이동평균

## Network edges

- `Source`, `Target`: 연결된 두 토큰
- `joint_count`: 설정 window 안에서 함께 출현한 횟수
- `period_id` 또는 동적 기간 열: 해당 edge가 속하는 시기
- 기타 PMI 계열 열: 네트워크 생성 과정에서 함께 산출된 보조 통계

파일명은 token profile, window, period, 최소 joint count와 stopword rule 버전을 기록한다.

### `multi_core_ego_neighbors.csv`

- `region_norm`, `alter_token`: 핵심어와 주변어
- `token_profile`, `network_window`, `center_mode`: 토큰·윈도우·중심어 조건
- `period_set_id`, `period_id`: 전체 시기 또는 long period 식별자
- `edge_weight`: 핵심어와 주변어의 연결량
- `neighbor_rank`, `neighbor_share_within_region`: 핵심어 내부 주변어 순위와 연결 비중
- `dominant_pos`, `dominant_pos_share`: 주변어의 최다 품사와 그 비중
- `pos_group`, `pos_group_label`: 대시보드 표시용 품사군
- `region_dominant_pos` 계열: 핵심어 노드의 dominant POS 정보

이 파일은 집계 네트워크의 공개용 대시보드 입력이며 기사 원문, 문맥 문자열, 기사·문맥 식별자를 포함하지 않는다.

## Network overlap metrics

- `focus_a_label`, `focus_b_label`: 비교 핵심어
- `topn`: 비교에 사용한 상위 주변어 수, `0`은 전체 주변어
- `jaccard`: 주변어 집합의 교집합/합집합 비율
- `weighted_jaccard`: 정규화된 주변어 가중치 분포의 겹침
- `cosine`: 주변어 가중치 벡터의 방향 유사도
- `direct_strength`: 두 핵심어 사이 직접 edge의 정규화 강도
- `shared_neighbor_count`: 공유 주변어 수
- `support_status`: 자료량 안정성 진단

공개본에서는 로컬 절대경로가 포함된 `source_file` 열을 제거했다.

## Pair-conditioned keyness

- `token`: 공유 주변어
- `period_id`, `pair_id`: 분석 시기와 핵심어 쌍
- `pair_count_mode`: pair 연결량 결합 방식, 논문은 `min`
- `count_period`, `count_ref`: target pair와 비교군에서의 연결량
- `log_odds_z`, `log_likelihood`, `log_ratio`, `chi_square`, `tfidf`: keyness 지표
- `robust_score`: 다섯 지표의 상위 후보 목록에 포함된 횟수
- `included_metrics`: 해당 후보를 지지한 지표
- `edge_a`, `edge_b`: 주변어와 두 핵심어 각각의 edge count
- `shared_strength`: 두 핵심어 전체 연결량으로 정규화된 공유 강도

`paper_top20.csv`의 `paper_rank`는 각 period-pair 안에서 log-likelihood와 log odds z를 기준으로 정렬한 최종 순위다.

## Evidence index

- `date`, `title`: 기사 날짜와 제목
- `article_id`: 상하이도서관 데이터에서 추출한 기사 식별자
- `context_uids`: 연구용 문맥 식별자, 복수일 경우 세미콜론으로 구분
- `paper_section`: 논문에서 해당 기사를 사용한 분석 부분
- `evidence_terms`: 인용문과 관련된 핵심어 또는 주변어

원문과 번역문은 논문 본문을 참고한다.
