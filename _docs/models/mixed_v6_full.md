# mixed_v6_full

## 학습 설정

```
Weight:     weights/mixed_v6_full/final_net_epoch_0060.pth
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
데이터:     data/pallet/training_data/mixed_v6_full_train/
이미지 수:  9,100
구성:       mixed_v1_train 8K + blender_dark 100 + blender_view 1K

소스별 내용:
  mixed_v1_train   Isaac Sim 4K + Blender 4K (1:1)
  blender_dark     어두운 팔레트 (black, charcoal, dark_gray 등)
  blender_view     다양한 카메라 각도/거리 (top_down, mid, far, close)
```

## 평가

```
평가 조건                  PCK@3px   PCK@10px   PnP Rate   Reproj mean   Vol<20%   Vol<50%
────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1_val 200장         0.495     0.632      66.0%      143.8 px      52.1%     91.5%
capture0403noapril 188장   -         -          26.6%      -             -         -
  Avg KP: 2.9/9
```

## 비고

- **Val PCK@3px 0.495 — 전 모델 중 최고** (mixed_v1 대비 +2.6%)
- 어두운 팔레트에서 3/9 kps 감지 시작 (mixed_v1의 0/9에서 개선)
- augmentation 중간 강도 + dark 데이터 추가의 효과
- mixed_v7_sym의 base model로 사용
