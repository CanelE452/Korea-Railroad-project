# mixed_v3

## 학습 설정

```
Weight:     weights/mixed_v3/final_net_epoch_0091.pth
초기 weight: mixed_v1 ep60
Epochs:     61 → 91 (31 epoch 추가)
Batch size: 4
LR:         5e-5
Sigma:      4.0
Image size: 448
Seed:       -
소요시간:   6시간 18분
특수 설정:  --geo_loss --geo_lambda 0.1 (Geometric Loss 포함)
```

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v2_train/
이미지 수:  10,000
구성:       mixed_v1 8K + blender_manydir 2K
```

## 평가

```
평가 조건                  PCK@3px   PCK@10px   PnP Rate   Reproj mean   Vol Ratio   Vol<20%
────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1_val 200장         0.470     0.719      70.5%      88.6 px       0.764       33.0%
capture0403noapril 188장   -         -          27.1%      -             -           -
  Avg KP: 2.3/9
```

## 비고

- **Geometric Loss 최초 적용 모델**
- cuboid 3D 형태 대폭 개선 — mixed_v1의 납작한 cuboid가 실제 직육면체로 변화
- 하지만 감지율 하락 (avg KP 3.2→2.3) — soft-argmax가 약한 peak 억제
- Volume ratio 0.764로 과소추정 — soft-argmax centroid bias
- 어두운 팔레트는 여전히 미감지
