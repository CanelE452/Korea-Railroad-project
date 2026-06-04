# mixed_v1

## 학습 설정

```
Weight:     weights/mixed_v1/final_net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:     60
Batch size: 4
LR:         1e-4
Sigma:      4.0
Image size: 448
Seed:       3742
```

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v1_train/
이미지 수:  8,000
소스:       Isaac Sim 4K (idx 0-3999) + Blender 4K (idx 4000-7999), 1:1 균형
```

## 평가 (Synthetic Val)

```
평가 조건                  PCK@3px   PCK@5px   PCK@10px   PnP Rate   Reproj mean   Reproj med   Vol Ratio   Vol<20%   Vol<50%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v1_val 200장         0.469     0.569     0.731      72.5%      88.1 px       72.1 px      1.159       55.3%     93.6%
```

## Real Data 추론

```
데이터                     Avg KP   PnP Rate   비고
───────────────────────────────────────────────────────
real_data (1924장)         -        80.6%      geo filter passed: 751장
capture0403noapril (188장) 3.2/9    30.9%      어두운 팔레트 전멸
```

## 비고

- Fair eval에서 종합 1위 — 안정적인 Volume Ratio (std=1.606)
- 1:1 소스 균형이 핵심 성공 요인
- Self-training (selftrain_r1) 및 mixed_v2의 base model로 사용됨
- 어두운 색상 팔레트에서는 약함 (학습 데이터에 dark pallet 부족)
