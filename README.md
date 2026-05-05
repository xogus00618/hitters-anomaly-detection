# Hitters Anomaly Detection — Isolation Forest + PCA 연구

Hitters 야구 선수 데이터에서 구조적 이상치(변수 간 상관관계 위반)를 탐지하기 위해
Isolation Forest, PCA, Extended iForest를 비교 분석한 학부 연구 프로젝트입니다.
(지도교수: 손원, 단국대학교)

## 연구 배경

iForest는 변수를 하나씩 분할하기 때문에 변수 간 상관관계를 위반하는
구조적 이상치를 탐지하는 데 한계가 있습니다. 본 연구는 PCA를 활용하여
이 한계를 극복하는 방법을 탐색합니다.

## 데이터

- ISLR 패키지의 Hitters 데이터셋 (MLB 야구 선수 성적)
- 수치형 변수 16개 사용 (Salary 제외)
- 합성 이상치: CAtBat=8000, CHits=1000, 나머지 전체 평균 (263번째 관측값)

## 실험 흐름

| 단계 | 방법 | 주요 결과 |
|---|---|---|
| 1 | iForest (Baseline) | 합성 이상치 112위 |
| 2 | 일반 PCA + iForest | 134위 (성능 하락) |
| 3 | Sparse PCA 3개 + iForest | 88위 |
| 4 | Sparse PCA 1개 + iForest | 62위 (최고 성능) |
| 5 | 샘플 단위 PCA + iForest | ss=50 기준 2~3위 |
| 6 | Extended iForest (ndim=16) | 중간 수준 |
| 7 | 관측값 배수 실험 (1~5배) | 전체PCA 압도적 안정성 |
| 8 | ntrees 수렴 실험 (100~1000) | 200트리 이후 안정적 수렴 |

## 주요 발견

- Sparse PCA 1개(커리어 누적 변수 상관관계 포착)가 가장 효과적
- 전체PCA + iForest: 관측값 증가에도 안정적 탐지 성능 유지
- 샘플 단위 PCA: seed 고정 + ntrees 충분히 확보 시 안정적 수렴
- 고차원(변수 100개) 희소 이상치에서는 모든 방법 탐지 한계 확인

## 구현 특이사항

샘플 단위 PCA + iForest는 isotree 패키지의 C++ 소스코드를 직접 분석하여
내부 동작 원리를 파악한 후 R로 구현한 차선책입니다.
매 트리마다 서로 다른 샘플로 PCA를 수행하여 다양한 축 방향을 학습합니다.

## SHAP 분석

PCA 및 Sparse PCA 적용 후 상위 10개 이상치 후보와 합성 이상치에 대해
independence, ctree 두 가지 approach로 SHAP 값을 계산하였습니다.
CAtBat이 일관되게 1위 기여 변수로 나타났으나,
CHits의 기여는 낮아 변수 간 상관관계가 완전히 반영되지 못한 한계가 존재합니다.

## 사용 라이브러리

`ISLR` `isotree` `shapr` `elasticnet`
