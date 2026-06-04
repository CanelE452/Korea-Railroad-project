# 1. 연구 개요

## 1.1 문제 정의

- 산업 현장의 리프터(포크리프트)에 장착된 카메라로 팔레트의 6D 포즈(위치 + 방향)를 추정해야 함
- 플라스틱 팔레트는 표면 반사, 화물 적재에 의한 가림(occlusion) 등으로 인식이 어려움
- 기존 공개 데이터셋은 목재 팔레트에 한정되며, 플라스틱 팔레트 6D pose 데이터셋은 부재
- 실제 환경의 labeled 데이터 확보가 어려움 (6D pose annotation 비용 매우 높음)

## 1.2 제안 해법 요약

1. Isaac Sim + Omniverse Replicator로 다양한 팔레트의 합성 데이터를 생성하여 DOPE 모델을 사전 학습
2. 학습된 모델로 레이블 없는 실제 이미지에 대해 예측 수행
3. **RANSAC subset consensus + LOO cross-validation** 로 pseudo-label 의 타당성을 검증 (23 후보 P/R 비교로 선정, 2026-04-11)
4. 검증을 통과한 pseudo-label과 합성 데이터를 합쳐 모델 finetuning
5. Step 2~3을 반복하여 점진적으로 성능 향상

## 1.3 핵심 설계 결정사항

```
항목                결정                                                        근거
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Pose 추정 모델      DOPE                                                        NVIDIA synthetic data pipeline과 호환, keypoint 기반
Pose 복원 방법      EPnP + RANSAC (OpenCV)                                      DOPE 원논문과 일관, gradient 불필요
DA 방식             Geometry-aware Self-Training                                 pseudo-label을 기하학적으로 검증, depth 불필요
Pseudo-label 필터   RANSAC subset consensus (c≥6) + LOO (τ=0.05)                 23 후보 P/R 비교 (2026-04-11) 로 F1 1위 선정 — _docs/filter/ 참조
Depth 카메라        미사용 (RGB only)                                            DOPE+PnP는 RGB만으로 가능, depth 의존성 제거
Synthetic Data 생성 NVIDIA 공식 워크플로우 기반                                  Isaac Sim + Replicator + Domain Randomization
```

## 1.4 방법론의 카테고리

본 연구의 방법은 **Geometry-aware Self-Training for Sim-to-Real Domain Adaptation**에 해당한다.

```
Domain Adaptation 방법 분류에서의 위치:

 [데이터 수준]  Domain Randomization (Isaac Sim)
               → 합성 데이터 생성 시 다양한 조건으로 domain gap 사전 축소

 [예측 수준]    Geometric Self-Training (핵심 Contribution)
               → 모델 예측을 기하학적으로 검증하여 신뢰할 수 있는
                 pseudo-label만 학습에 활용

 참고: Adversarial DA(Domain Classifier + GRL)는 추가 실험으로
       비교 가능하나, 메인 프레임워크에는 포함하지 않음
```

---

# 2. 전체 파이프라인

## 2.1 3단계 요약

```
Step 1                     Step 2                      Step 3
Isaac Sim 합성 데이터  →  Real inference          →  Finetuning
+ DOPE supervised 학습     + Geo Filter               합성 + pseudo-label
                           + Pseudo-label 생성         L = L_syn + α·L_real
                                                            │
                                                      Step 2로 반복
```

## 2.2 파이프라인 흐름도

```
┌──────────────────────────────────────────────────────────────┐
│                    Step 1: DOPE 초기 학습                      │
│                                                               │
│  Isaac Sim + Omniverse Replicator                            │
│  → 합성 데이터 (RGB + 8 keypoint GT)                          │
│  → Domain Randomization (재질, 조명, 가림, 배경)              │
│                         │                                     │
│                         ▼                                     │
│                    DOPE Supervised 학습                        │
│                    Loss = L_pose (MSE)                        │
│                         │                                     │
│                    Pre-trained DOPE 모델                       │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│            Step 2: Pseudo-label 생성 + 필터링                  │
│                                                               │
│  Real Unlabeled 이미지                                        │
│           │                                                   │
│           ▼                                                   │
│     DOPE inference (no grad)                                  │
│           │                                                   │
│           ▼                                                   │
│     8개 keypoint 예측                                         │
│           │                                                   │
│     ┌─────┴─────────────────────────────────┐                │
│     │         Geometric Filter               │                │
│     │                                        │                │
│     │  [1] Pre-filter                        │                │
│     │      detected keypoint ≥ 5             │                │
│     │                                        │                │
│     │  [2] RANSAC subset consensus (main)    │                │
│     │      n_iter=50, subset=5               │                │
│     │      reproj τ=5px, consensus ≥ 6       │                │
│     │      → best pose (R, t)                │                │
│     │                                        │                │
│     │  [3] LOO cross-validation               │                │
│     │      median(e_i / D) < τ_LOO (= 0.10)  │                │
│     │                                        │                │
│     └────────────┬───────────────────────────┘                │
│                  │                                             │
│          통과 → pseudo-label ✓                                │
│          실패 → 버림 ✗                                        │
└──────────────────┼───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    Step 3: Finetuning                          │
│                                                               │
│  Synthetic (labeled)  → DOPE → L_pose_syn (GT 기반)           │
│  Real (pseudo-label)  → DOPE → L_pose_real (pseudo 기반)      │
│                                                               │
│  L_total = L_pose_syn + α · L_pose_real                       │
│                                                               │
│  → 모델 업데이트                                               │
│  → Step 2로 반복 (2~3 라운드)                                  │
└──────────────────────────────────────────────────────────────┘
```
