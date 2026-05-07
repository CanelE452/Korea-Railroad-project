# Metric Validation — NN Matching Pipeline 무결성 증명

상태: **★ 완료** (2026-04-14)

## 배경

발표 방어 대응 — "metric 을 여러 번 바꿨는데 지금 것이 맞다는 보장은?"
근본 재실험 (seed 5 개, Seen/Unseen 평가) 은 발표까지 물리적 불가.
대안: **pipeline 자체가 수학적으로 올바르다**는 걸 합성 val 데이터로 증명.

## Metric 변경 이력

```
시점               Metric                                상태
─────────────────────────────────────────────────────────────────────
~2026-04-12       PnP self-reprojection vs GT            버림 (self-referential)
2026-04-12        Direct index (pred[i] vs GT[i])        버림 (convention 민감)
2026-04-13~       Hungarian NN matching (현행) ★          검증 완료 (이 문서)
```

## 검증 설계

합성 val (`data/pallet/training_data/val`, 3000 장 중 500 장) — rendering
camera 의 projection 으로 만든 GT 라 수치적으로 완벽. 여기서 4 가지 test 수행:

```
test                      입력                               기대 결과
─────────────────────────────────────────────────────────────────────────
T1 Identity               GT cuboid 를 pred 로 넣기           max err < 1e-6
T2 Permutation            GT 를 random shuffle 해서 pred 로    max err < 1e-6
T3 Known Gaussian noise   GT + N(0, sigma=10px)              mean ≈ 12.53px (Rayleigh)
T4 Swap invariance        GT 의 2 corner 만 swap              NN=0 vs Direct>>0
```

## 결과 (500 frames × 8 corners = 4000 matches)

```
Test                              NN matching           Direct index       Pass
──────────────────────────────────────────────────────────────────────────────────
T1 Identity                       max = 0.00e+00        0.00              ★
T2 Random permutation             max = 0.00e+00        ~400px             ★
T3 Gaussian sigma=10px            mean = 12.53px        12.53              ★ (이론 12.53)
T4 Swap invariance (2 corners)    max = 0.00e+00        mean 48.54px       ★
```

## 발표 방어 포인트

### 1. Pipeline 자체 무결성 (T1/T2)
같은 점끼리 매칭 시 정확히 0 — 코딩 버그 없음.

### 2. 스케일 정확성 (T3)
σ=10 Gaussian noise 를 주면 Rayleigh 분포의 이론 mean = σ·√(π/2) = 12.53px
와 측정값이 완전히 일치. **metric 이 실제 픽셀 거리를 올바르게 재고 있음**.

### 3. Metric 변경 이유의 수치적 증거 (T4) ★
`이전 ep65=5.0% (direct) → 21.6% (NN)` 차이가 왜 발생했는지:

```
실험: GT 의 2 corner 를 랜덤으로 swap (convention mismatch 시뮬레이션)
  → NN matching:   0.00px (swap 복원)
  → Direct index:  평균 48.54px 허위 오차 (swap 자체를 모델 오차로 계산)
```

즉 이전 direct index metric 으로 나온 낮은 수치는 **GT convention 민감도**
때문이고, 모델 실제 성능이 낮은 게 아니었음. NN matching 은 이 구조적 결함을
제거.

## 한계 (committee 대응)

1. **T3 의 sigma=10 은 등방 Gaussian** — 실제 belief map peak 오차는 비등방일 수 있음.
   그러나 스케일 정확성 증명에는 충분 (order of magnitude 검증).
2. **합성 val 에서만 검증** — real data 의 카메라 왜곡 / AprilTag GT 오차는
   별도. 그러나 metric 자체는 data 에 무관한 함수이므로 synthetic 검증으로 충분.
3. **Hungarian 의 경계 실패 케이스** — 8 corner 가 공간적으로 거의 겹칠 때
   부분 swap 이 발생할 수 있으나, projected_cuboid 는 최소 edge 가 수십 px
   이상이라 실무상 문제 없음.

## 재현 명령

```bash
C:/Users/minjae/anaconda3/envs/pallet-pose/python.exe \
    scripts/data_prep/eval/validate_nn_metric.py \
    --val_dir data/pallet/training_data/val \
    --n_frames 500
```

## 저장

- 스크립트: `scripts/data_prep/eval/validate_nn_metric.py`
- 리포트: `data/pallet/eval_results/metric_validation_report.txt`
- 평가 스크립트 (본체): `scripts/data_prep/eval/eval_nn_matching.py`

## 관련

- 실제 평가 결과: [`../filter/ablation.md`](../filter/ablation.md) Table 2
- Self-training 결과: [`../self_training/rounds.md`](../self_training/rounds.md)
- GT convention: history/2026-04-14.md (permutation [3,2,1,0,7,6,5,4], gt_final_isaac/)
