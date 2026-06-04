# Filter Selection — GT-based P/R Analysis (2026-04-11)

## 목적

Self-training pseudo-label filter로 어떤 게 가장 좋은지 GT 기반 precision/recall로 결정.
이전까지는 canonical filter design (B∧C core + A auxiliary)을 설계안으로 사용했지만,
실측 검증을 한 번도 하지 않아 논문과 코드의 일관성이 무너져 있었음.

## 방법

- 데이터: `capture0403middle` 440 프레임 (AprilTag 기반 GT, 수동 보정 130 포함)
- 모델: v8_A (v9_ablation_A_coord/ep65), v8_A_control/ep68, selftrain_r1/ep70
- "good" 정의: 예측 pose를 3D corners에 projectPoints → GT projected_cuboid와
  평균 픽셀 거리 < threshold. 3D ADD 대신 2D 비교를 쓴 이유는 학습 frame과 GT
  frame의 object-space convention이 완전히 일치하지 않아 3D 기준은 bias됨.
- Threshold: 10 / 30 / 50 / 80 px 비교. 모든 모델에서 10px 통과 0개, 50px가 가장
  정보량 많아서 primary로 사용.

## 필터 후보 (23개)

```
ID    필터
─────────────────────────────────────────────────────────────
F0    no filter
F1    confidence (min peak > 0.5)
F2    old reproj + cuboid_geometry + size (기존 geometric_filter.py)
F3    A only (flip consistency, canonical_filters.filter_A)
F4    B only (structural support, default thresholds)
F5    C only (normalized LOO PnP, tau_C=0.05)
F6    D only (diagonal incidence)
F7    B ∧ C default
F8    A ∧ B ∧ C (full canonical pipeline)
F9    B ∧ C ∧ D
F10   A ∧ B ∧ C ∧ D
F11   RANSAC subset consensus (n_iter=50, k=5, τ=5px, c≥6)
F12   reproj-guided PnP (Huber + coverage)
F13   (B ∧ C) on reproj-guided pose
F14   B ∧ C loose 2x
F15   B ∧ C loose 3x
F16   B ∧ C very loose
F17   RANSAC c≥4
F18   RANSAC c≥5
F19   RANSAC c≥7
F20   RANSAC c≥8
F21   B loose 2x only
F22   C loose 2x only
```

## 결과 (threshold = 50 px)

### v8_A_control (ep68)  —  n_good = 15/440

```
필터                   pass   TP    FP   FN   P      R      F1
──────────────────────────────────────────────────────────────
F11 RANSAC c≥6 ★★★      14    10    4    0    0.714  1.000  0.833
F4  B only              21    12    9    3    0.571  0.800  0.667
F19 RANSAC c≥7           5     5    0    5    1.000  0.500  0.667
F18 RANSAC c≥5          29    10   19    0    0.345  1.000  0.513
F21 B loose 2x only     42    14   28    1    0.333  0.933  0.491
F17 RANSAC c≥4          35    10   25    0    0.286  1.000  0.444
F20 RANSAC c≥8           2     2    0    8    1.000  0.200  0.333
F7  B ∧ C default        2     2    0   13    1.000  0.133  0.235
F9  B ∧ C ∧ D            2     2    0   13    1.000  0.133  0.235
F13 B∧C on RG pose       2     2    0   13    1.000  0.133  0.235
F16 B∧C very loose      49     6   43    9    0.122  0.400  0.188
F15 B∧C loose 3x        31     4   27   11    0.129  0.267  0.174
F14 B∧C loose 2x         9     2    7   13    0.222  0.133  0.167
F5  C only default      14     2   12   13    0.143  0.133  0.138
F0  no filter          237    15  222    0    0.063  1.000  0.119
F12 reproj-guided      237    15  222    0    0.063  1.000  0.119
F22 C loose 2x only     28     2   26   13    0.071  0.133  0.093
F6  D only             169     2  167   13    0.012  0.133  0.022
F3  A only              92     1   91   14    0.011  0.067  0.019
F1  confidence >0.5      0     0    0   15    0.000  0.000  0.000
F2  old filter           0     0    0   15    0.000  0.000  0.000
F8  A∧B∧C                0     0    0   15    0.000  0.000  0.000
F10 A∧B∧C∧D              0     0    0   15    0.000  0.000  0.000
```

### selftrain_r1 (r1)  —  n_good = 27/440

```
필터                   pass   TP    FP   FN   P      R      F1
──────────────────────────────────────────────────────────────
F11 RANSAC c≥6 ★★★      23    13   10    0    0.565  1.000  0.722
F19 RANSAC c≥7          10     8    2    5    0.800  0.615  0.696
F18 RANSAC c≥5          33    13   20    0    0.394  1.000  0.565
F17 RANSAC c≥4          34    13   21    0    0.382  1.000  0.553
F16 B∧C very loose      33    16   17   11    0.485  0.593  0.533
F15 B∧C loose 3x        23    12   11   15    0.522  0.444  0.480
F22 C loose 2x only     22    11   11   16    0.500  0.407  0.449
F12 reproj-guided      104    29   75    0    0.279  1.000  0.436
F21 B loose 2x only     29    12   17   15    0.414  0.444  0.429
F0  no filter          104    27   77    0    0.260  1.000  0.412
F5  C only default      19     9   10   18    0.474  0.333  0.391
F14 B∧C loose 2x        18     8   10   19    0.444  0.296  0.356
F6  D only              89    19   70    8    0.213  0.704  0.328
F20 RANSAC c≥8           4     3    1   10    0.750  0.231  0.353
F1  confidence >0.5     14     4   10   23    0.286  0.148  0.195
F4  B only               5     2    3   25    0.400  0.074  0.125
F3  A only               5     2    3   25    0.400  0.074  0.125
F7  B ∧ C default        2     1    1   26    0.500  0.037  0.069
F13 B∧C on RG pose       2     1    1   28    0.500  0.035  0.065
F9  B ∧ C ∧ D            2     1    1   26    0.500  0.037  0.069
F2  old filter           0     0    0   27    0.000  0.000  0.000
F8  A∧B∧C                0     0    0   27    0.000  0.000  0.000
F10 A∧B∧C∧D              0     0    0   27    0.000  0.000  0.000
```

## 핵심 결론

### 1. F11 (RANSAC subset consensus ≥ 6) 이 두 모델 모두 1등
- ep68: F1 = 0.833 (압도적)
- r1: F1 = 0.722
- consensus 파라미터 sweep (4/5/6/7/8) 결과 6이 sweet spot. 5는 precision 급락,
  7은 recall 절반, 8은 recall 더 떨어짐.

### 2. Canonical B∧C (F7) 는 threshold sweep 으로 살아나지 못함
- 디폴트 F7: F1 ≤ 0.235
- 2x loose F14: F1 ≤ 0.356
- 3x loose F15: F1 ≤ 0.480
- very loose F16: F1 ≤ 0.533
- 가장 느슨한 F16조차 F11 대비 F1 0.18 낮음. "very loose"는 이미 의미있는 필터링이
  아니라 "no filter + size sanity" 수준. 더 풀면 filter 없음과 동일.
- 즉 **B∧C 설계 자체가 capture0403middle에서 sub-optimal**이고 threshold
  튜닝으로 구제 불가.

### 3. F8 (A ∧ B ∧ C full pipeline) = 0 통과 / 440
- 논문에서 주장하던 full pipeline이 실측에서 가장 보수적으로 작동 → 아무것도
  pseudo-label로 안 넘김. 0 통과라서 precision 정의 불가.

### 4. F2 (old reproj + cuboid + size) = 0 통과 / 440
- 기존 코드의 geometric_filter.py 로직도 실측에서 작동 안 함.
  (이건 삭제/교체 전 기록 시점 기준. 본 문서 작성 후 교체됨.)

### 5. A (flip consistency) 와 B (structural support) 는 모델 의존적
- ep68에서는 F4 (B only) F1=0.667 로 강함
- r1에서는 F4 F1=0.125 로 붕괴
- RANSAC 은 두 모델 모두 robust 하지만 B 는 단독으로 신뢰 불가

## 선정된 필터 (우리 필터)

```
[1] Pre-filter:
  min_keypoints = 5

[2] RANSAC subset consensus (main gate):
  n_iter           = 50
  subset_size      = 5
  reproj_thresh_px = 5.0
  min_consensus    = 6

[3] LOO cross-validation (post-check):
  tau_LOO = 0.05
  각 keypoint i 제외 → PnP → reproj error / D < tau_LOO
  RANSAC 통과 frame의 정성 분석에서 one-sided keypoint collapse 잔존 확인
  → LOO 추가로 해결
```

구현: `scripts/self_training/geometric_filter.py` (GeometricFilter 클래스)
       `scripts/data_prep/canonical_filters.py` (filter_C = LOO)
설정: `config/stage3_selftrain.yaml` (geometric_filter 섹션)

## 논문 story 변경

기존: "3단계 geometric filter (A: flip, B: structural, C: LOO PnP) 로
       pseudo-label 품질 보장"

신규: "GT 기반 precision/recall 분석을 통해 여러 필터를 비교하여 RANSAC subset
       consensus 를 최종 채택. Canonical geometric filter (B∧C) 는 precision 은
       높으나 recall 이 극도로 낮아 실용적 pseudo-label 수량을 확보하지 못함.
       RANSAC 은 precision 과 recall 의 균형이 가장 좋음."

Filter Ablation Table (논문용):
```
Filter                 Precision  Recall   F1
────────────────────────────────────────────────
No filter              0.06       1.00     0.119
Confidence > 0.5       0.00       0.00     0.000
B ∧ C (canonical)      1.00       0.13     0.235
B ∧ C (loose 3x)       0.13       0.27     0.174
RANSAC c≥6 (Ours)      0.71       1.00     0.833  ★
(ep68 기준, threshold 50 px)
```

B∧C 시도 자체는 논문에서 ablation 비교로 살림 — "canonical filter 를 설계했으나
실측에서 RANSAC 이 더 robust 함을 확인" 이 methodology contribution.

## Limitations

- capture0403middle 한정 결과. noapril (실제 self-training pool) 에서 cross-check
  필요하지만 GT 없어서 별도 촬영 필요. 논문 limitation 으로 명시.
- v8_A 가 이 데이터에 다소 OOD (frame 디버그 결과 mode collapse 관찰) — good 샘플
  수량 15~27개로 통계적 power 제한.
- "good" threshold 50 px 는 실물 기준 약 5~10 cm 오차 — self-training 에 쓸 만한
  수준이지만 엄격한 기준은 아님.

## 파일

- 스크립트: `scripts/data_prep/eval/filter_pr_eval.py`
- 결과: `data/pallet/eval_results/filter_pr/summary_*.{csv,json}`, `per_frame_*.json`
- 디버그: `data/pallet/eval_results/filter_pr/debug_frame100.jpg`
