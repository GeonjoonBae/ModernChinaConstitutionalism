# 분석 방법과 조건

## 네트워크 구성

토큰 `i`와 `j`가 설정 window 안에서 함께 등장한 횟수를 edge weight `w_ij`로 사용한다. 본 논문의 네트워크는 방향을 고려하지 않는 undirected network다.

## Jaccard 유사도

핵심어 A, B의 주변어 집합을 각각 `N_A`, `N_B`라고 할 때 다음과 같이 계산한다.

```text
J(A,B) = |N_A ∩ N_B| / |N_A ∪ N_B|
```

공유 주변어의 종류가 많을수록 값이 커진다. 논문에서는 각 핵심어와 가장 강하게 연결된 상위 100개 주변어를 기본 비교 범위로 사용한다.

## Weighted Jaccard 유사도

핵심어별 전체 연결량으로 각 주변어의 edge weight를 정규화한 값을 `p_Ai`, `p_Bi`라고 할 때 다음과 같이 계산한다.

```text
WJ(A,B) = Σ min(p_Ai, p_Bi) / Σ max(p_Ai, p_Bi)
```

공유 주변어의 유무뿐 아니라 각 주변어가 두 핵심어의 연결 분포에서 차지하는 상대적 비중까지 비교한다.

## Pair-conditioned keyness

공유 주변어 `t`의 target pair 연결량은 다음과 같다.

```text
count_pair(t) = min(w_At, w_Bt)
```

비교군은 같은 시기의 나머지 다섯 핵심어 쌍이다. `log_odds_z`, `log_likelihood`, `log_ratio`, `chi_square`, `tfidf`를 모두 계산한다. 각 지표의 양의 방향 상위 30개 안에 포함된 횟수를 robust score로 정의하며, 논문은 score 4 이상 후보를 대상으로 한다.

최종 Top20은 `log_likelihood` 내림차순으로 선정하고, 동률이면 `|log_odds_z|`가 큰 후보를 우선한다.

## 논문 조건

| 분석 | profile | window | neighbor scope | 추가 조건 |
| --- | --- | ---: | --- | --- |
| 전체 시기 주변어 목록·가중치 유사도 | strict | 1, 5, 10, 20 | top100 | core exact |
| 시기별 주변어 가중치 유사도 | strict | 10 | top100 | core exact |
| 공유 주변어 keyness | full | 10 | top100 | min, same_period_other_pairs |

## 해석 범위

유사도와 keyness는 신문 지면에서 나타난 어휘적 관계를 나타낸다. 두 개념의 철학적 동일성, 직접적인 인과관계 또는 정치세력의 의도를 자동으로 입증하지 않는다.
