# 연구 가이드 — Pallet 6D Pose Geometry-aware Self-Training

> **논문 제목:** 파렛트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 도메인 적응
> **핵심 키워드:** 6D pose estimation, geometry-aware self-training, synthetic data, geometric filter, unsupervised domain adaptation
> **작성일:** 2026-03-25 (v5)
> **작성자:** 민재
> **중요** 이거는 논문과 github에 코드를 올려서 다른사람들도 테스트하거나 실험할수 있도록 재현성이 있어야됨 그래서 파일 구조와 정리가 중요

---

## 문서 구조

### 전처리 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, 3D cuboid convention (Y=UP), 팔레트 규격
data_pipeline.md                합성 데이터 생성/검증/병합 워크플로우
```

### 모델 아키텍처 (`method/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
overview.md                     연구 개요, 문제 정의, 제안 해법, 전체 파이프라인
step1_synthetic_data.md         Step 1: Isaac Sim 합성 데이터 생성 + DOPE 학습
step2_geometric_filter.md       Step 2: RANSAC subset consensus Filter + Pseudo-label 생성
step3_finetuning.md             Step 3: Finetuning + 반복적 Self-Training 루프
generalization.md               다양한 팔레트 일반화 전략 + 데이터셋 구성
formulation.md                  수식 정의 + 평가 메트릭 (ADD, Reproj, 5cm5°)
implementation.md               구현 세부사항, Contribution, 참고문헌
```

### 모델 카탈로그 (`models/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
README.md                       모델 요약, 평가 비교 테이블, 상세 카드 링크
{model_name}.md                 개별 모델 카드 (학습 설정, 데이터, 평가 결과, 비고)
```

### 필터 연구 (`filter/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
README.md                               필터 폴더 인덱스, 현행 설계 (RANSAC c≥6) 요약
2026-04-11_selection.md                 23 후보 × 3 모델 GT 기반 P/R 실측 (RANSAC 선정)
2026-04-11_design_rationale.md          "필터 하나로 충분한 이유" + 논문 프레이밍
```

### 실험 (`experiments/`)

실험 단위로 파일 분할 후 5 개 분야 서브폴더로 재구성 (2026-04-12). 각 파일
은 하나의 Table 또는 Figure 에 대응. 전체 인덱스와 진행 상태는
`experiments/README.md` 참조.

```
폴더 / 파일                                내용                                 상태
──────────────────────────────────────────────────────────────────────────────────────
README.md                                  인덱스 + 평가 프로토콜                 —
model_catalog.md                           모델 카탈로그 (cross-cutting)          갱신
related_work.md                            T10 Related Work 비교                 예정
filter/
├── ablation.md                            T1 Filter Ablation main                예정
├── selection.md                           T3 Filter Selection P/R                ★ 완료
└── consensus_sweep.md                     T7 RANSAC consensus sweep              ★ 완료
loss/
├── ablation.md                            T2 Loss Ablation — coord               ★ 완료
└── coord_strategy.md                      T4 Coord Loss 학습 전략                예정
self_training/
├── rounds.md                              F1 Self-Training Round Figure          예정
├── alpha.md                               T6 α 민감도                            예정
└── forgetting.md                          T8 Catastrophic Forgetting             예정
eval/
├── seen_unseen.md                         T5 Real Seen vs Unseen                 촬영 대기
├── inference_speed.md                     Inference Speed breakdown              예정
└── qualitative.md                         Qualitative Failure Analysis           예정
synthetic/
├── multisource.md                         T9 Multi-source (legacy)               부분
└── sigma_sensitivity.md                   Sigma Sensitivity                      optional
```

### 서베이 (`survey/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
survey-6d-pose-estimation.md            6D Pose Estimation 분야 서베이 (방법론/학습 전략/메트릭 비교)
```

### 데이터 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, 3D cuboid convention (Y=UP), 팔레트 규격
data_pipeline.md                합성 데이터 생성/검증/병합 워크플로우
```

### Real Test Data

```
파일                                            내용
──────────────────────────────────────────────────────────────────────────────
data/pallet/real_data/README.md                 Real data split 정의, 촬영 프로토콜, AprilTag GT, 평가 메트릭
```

### 작업 기록 (`history/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
changelog.md                    과거 작업 이력 (렌더링 개선, 학습, 트러블슈팅)
```

---

## 변경 이력

```
날짜          버전    변경 내용
──────────────────────────────────────────────────────────────────────────────
2026-03-10    v1      초안 작성
2026-03-10    v2      팔레트 일반화 전략, NVIDIA 워크플로우 기반 Stage 1 보강
2026-03-10    v3      실전 렌더링 가이드, 품질 체크리스트 추가
2026-03-13    v3.2    Stage 1 코드 기준 동기화, DR 상세 파라미터
2026-03-19    v4      전면 구조 변경: FixMatch 제거, 3-Step Geometry-aware Self-Training으로 전환. 3단계 Geo Filter 신규 설계. 수식 정의 추가.
2026-03-25    v5      문서 구조 재편: preprocessing/method/experiments/survey/history 하위 폴더 분리. 키포인트 정의 복원. 합성 데이터 파이프라인 문서 추가. 작업 이력 정리.
2026-03-30    v6      멀티소스 학습: Blender 데이터 학습, 실험 관리 체계(compare_experiments.py), 3D 부피 비교 메트릭, 멀티소스 비교 실험 결과 추가
2026-04-11    v7      Filter 재선정: 23 후보 GT 기반 P/R 비교 후 canonical A∧B∧C → RANSAC subset consensus (c≥6) 교체. `filter_type` dispatcher + _docs/filter/ 전용 폴더 신설. overview/formulation/implementation/step2 전면 동기화.
```
