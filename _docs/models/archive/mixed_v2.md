# mixed_v2

## 학습 설정

```
Weight:     weights/mixed_v2/final_net_epoch_0091.pth
초기 weight: mixed_v1 ep60
Epochs:     61 → 91 (31 epoch 추가)
Batch size: 4
LR:         5e-5
Sigma:      4.0
Image size: 448
Seed:       66
소요시간:   5시간 50분
```

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v2_train/
이미지 수:  10,000
구성:       mixed_v1 8K (idx 0-7999) + blender_manydir 2K (idx 8000-9999)
```

### blender_manydir 색상 분포

```
Family:   plastic 1198장, wood 802장
Variant:  green 430, blue 349, natural_wood 251, weathered_brown 209,
          gray 170, weathered_gray 169, black 110, painted_green 99,
          orange 76, painted_blue 74, red 63
```

## 평가 (Synthetic Val)

```
평가 조건                  PCK@3px   PCK@5px   PCK@10px   PnP Rate   Reproj mean   Reproj med   Vol Ratio   Vol<20%   Vol<50%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1_val 200장         0.466     0.551     0.693      77.5%      112.4 px      75.9 px      1.048       54.6%     94.8%
```

## mixed_v1 대비 변화

```
메트릭         mixed_v1   mixed_v2   변화
──────────────────────────────────────────
PCK@3px        0.469      0.466      -0.3%
PCK@10px       0.731      0.693      -3.8%
PnP Rate       72.5%      77.5%      +5.0%
Reproj mean    88.1 px    112.4 px   악화
Vol Ratio      1.159      1.048      개선 (1.0에 가까움)
Vol<20%        55.3%      54.6%      -0.7%
```

## Real Data 추론

```
데이터                     Avg KP   PnP Rate   비고
───────────────────────────────────────────────────────
capture0403noapril (188장) 2.9/9    27.1%      mixed_v1 대비 오히려 하락
```

## 비고

- PnP 성공률은 올랐지만, keypoint 정밀도(PCK)와 reproj error는 악화
- blender_manydir의 black pallet이 110장(전체 1.1%)으로 부족
- 31 epoch 추가 학습이 과적합을 심화한 것으로 판단
- **결론: mixed_v1 대비 개선 없음 — 어두운 팔레트 문제는 합성 데이터 색상 다양화가 필요**
