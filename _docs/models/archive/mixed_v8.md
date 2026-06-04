# mixed_v8

## 학습 설정

```
Weight:     weights/mixed_v8/final_net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:     60
Batch size: 4
LR:         1e-4
Sigma:      4.0
Image size: 448
Augmentation: 중간 강도 (brightness ±35%, HueSaturation, RandomGamma)
```

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v8_train/
이미지 수:  9,000
구성:
  Isaac Sim 2000 (mixed_v1에서 균등 샘플링)
  Blender 2000 (mixed_v1에서 균등 샘플링)
  test_blender_4000 (4000, 신규 Blender 데이터)
  test_blender_v1000 (1000, 신규 다양한 시점)
```

## 평가

```
평가 조건                  PCK@3px   PCK@10px   PnP Rate   Reproj mean   Vol<20%
────────────────────────────────────────────────────────────────────────────────────
mixed_v1_val 200장         0.460     0.635      50.0%      205.7 px      35.4%
capture0403noapril 188장   -         -          49.5%      -             -
  Avg KP: 3.2/9
```

## Real PnP 역대 최고

```
모델               Real PnP (noapril)
───────────────────────────────────────
mixed_v1           30.9%
mixed_v7_sym       37.2%
mixed_v8           49.5% ★
```

Val은 떨어졌지만 Real PnP가 49.5%로 역대 최고.
test_blender 데이터가 real 도메인에 더 가까운 것으로 판단.

## Self-Training (mixed_v8_st_8only)

noapril에서 canonical filter(A2+B2+C2) 통과한 8장으로 finetune:

```
Weight:     weights/mixed_v8_st_8only/final_net_epoch_0091.pth
초기 weight: mixed_v8 ep60
Epochs:     61 → 91 (31 epoch)
데이터:     pseudo-label 8장만 (noapril A2+B2+C2 passed)
소요시간:   22초
```

### Self-Training 효과

```
메트릭                mixed_v8      mixed_v8_st_8only
──────────────────────────────────────────────────────
noapril Avg KP        3.2/9         5.5/9 ★
noapril PnP Rate      49.5%         78.7% ★
noapril A2+B2+C2      8장           97장 ★
real_data PnP (20장)  ~77%          75%  (소폭 하락)
real_data A2+B2+C2    100장         361장 ★
```

8장 pseudo-label만으로 PnP 49.5% → 78.7%, 필터 통과 8 → 97장.

## Canonical Filter v2 개발

mixed_v8 평가 과정에서 canonical geometric filter 재설계:

```
기존 (v1)          신규 (v2 canonical)
────────────────────────────────────────────────────────────
A: Flip Consistency      A: auxiliary score (core에서 제외)
B: 2D Diagonal           B: Structural Coverage + Non-Collinearity
C: LOO PnP               C: Normalized LOO PnP Stability
절대 px threshold        무차원 비율 (데이터셋/해상도 불변)
```

코드: `scripts/data_prep/canonical_filters.py`, `scripts/data_prep/infer_and_filter_v2.py`
