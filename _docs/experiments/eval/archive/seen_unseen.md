# T5. Real Seen vs Unseen

상태: **촬영 대기** — AprilTag GT 가 있는 Real 데이터셋 필요

## 목적

본 연구의 일반화 주장을 **학습에 쓰이지 않은 팔레트 종류** 에 대해 정량적
으로 입증. Seen / Unseen 성능 차이가 작으면 "self-training 이 실제 일반화
한다" 는 근거가 된다.

## 실험 설계

```
Split            정의                                      목표 수량
────────────────────────────────────────────────────────────────────────
Seen             학습 3D 모델과 같은 종류의 플라스틱 팔레트   50 ~ 100 장
Unseen           학습 3D 모델에 없는 플라스틱 팔레트          50 ~ 100 장
Unseen - Wood    목재 팔레트 (domain 외삽, future work)       50 ~ 100 장
```

촬영 프로토콜: `data/pallet/real_data/README.md` (AprilTag 여러 개 배치 후
solvePnP 로 GT pose 추정).

## Table 5

```
Method                 Seen Reproj ↓   Unseen Reproj ↓   Seen ADD ↑   Unseen ADD ↑   Seen 5 cm 5° ↑   Unseen 5 cm 5° ↑
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only          ?                ?                 ?            ?              ?                 ?
+ coord ft (v8_A)       ?                ?                 ?            ?              ?                 ?
+ ST (RANSAC, Ours) ★   ?                ?                 ?            ?              ?                 ?
```

이 데이터셋에서는 3D ADD / 5 cm 5° 를 사용할 수 있음 (AprilTag GT 가
proper object frame 으로 제공됨). capture0403middle 과 달리 3D 수치가
primary.

## Δ Unseen 지표 (논문 핵심)

```
Δ Unseen = Unseen Metric (Ours) − Unseen Metric (Synthetic-only)
```

Δ 가 positive 이고 Seen Δ 와 비슷한 크기 → "ST 가 모든 팔레트 종류에 일반화"
Δ 가 크게 작음 → "ST 가 Seen 에만 편향" (nuance 필요)

## 선행 조건

1. 촬영: Seen / Unseen 각각 50 ~ 100 장 (카메라 ~3 가지 각도 × 거리)
2. AprilTag GT 생성: `scripts/data_prep/apriltag/apriltag_gt.py`
3. 저장 위치: `data/pallet/real_data/real_test_seen/`,
   `real_test_unseen/`
4. 평가 스크립트: `scripts/data_prep/eval/evaluate_real.py` (ADD + 5 cm 5° + Reproj)

## 관련

- 촬영 프로토콜: `data/pallet/real_data/README.md`
- AprilTag 스크립트: `scripts/data_prep/apriltag/apriltag_gt.py`,
  `scripts/data_prep/apriltag/apriltag_gt_multitag.py`
- 평가 스크립트: `scripts/data_prep/eval/evaluate_real.py`
- 일반화 전략: `_docs/method/generalization.md`
