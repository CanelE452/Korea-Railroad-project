# mixed_v9

## 핵심 변경

```
변경                        내용                                              목적
───────────────────────────────────────────────────────────────────────────────────────────
1. Structural Loss          flip equivariance + sparse edge + coord Huber     구조 학습
2. 데이터셋 축소            Isaac Sim 500 + Blender 2000 = 2,500장            경량 실험
3. Warmup 10 epoch          structural loss는 epoch 10부터 ramp-up            초기 안정
```

## 학습 설정

```
Weight:     weights/mixed_v9/net_epoch_0060.pth
초기 weight: scratch (VGG-19 ImageNet pretrained)
Epochs:     60
Batch size: 4
LR:         1e-4
Sigma:      4.0
Image size: 448
Struct loss: warmup=10, flip=0.02, edge=0.05, coord=0.10
```

## Structural Loss 구성

```
Component        Weight   Warmup   Ramp       설명
─────────────────────────────────────────────────────────────────────────────
flip equivar.    0.02     10ep     10ep 선형   좌우반전 좌표 일관성 (Huber)
sparse edge      0.05     10ep     10ep 선형   cuboid 12 edge + 4 diag log-ratio
coord Huber      0.10     10ep     10ep 선형   soft-argmax 좌표 normalized Huber
─────────────────────────────────────────────────────────────────────────────
모든 거리는 object diagonal로 normalize, Huber delta=0.03
flip branch는 stop-gradient (한쪽만 밈)
```

학습 흐름:
- epoch 0-9: belief MSE + affinity MSE (기존 DOPE만)
- epoch 10-19: 위 + structural loss × ramp (0→1 선형 증가)
- epoch 20-59: 위 + structural loss × 1.0 (full)

## 실패 기록

### v9a (geo_loss, lambda=0.1, warmup=5)
- BPnP 기반 geometric loss 사용
- epoch 5에서 PnP loss ON 순간 loss 42.7로 폭발 (400배)
- 이후 0.3~1.1 사이에서 불안정, 수렴 실패

### v9b (geo_loss, lambda=0.01, warmup=20)
- lambda 10배 감소, warmup 4배 증가
- epoch 20까지 안정 (0.07~0.14), epoch 38에서 0.73 스파이크
- BPnP gradient 자체가 불안정 → structural loss로 전환

### v9 (현재: struct_loss, warmup=10)
- BPnP 제거, 순수 2D structural loss로 전환
- flip equivariance: shortcut 학습 직접 방지
- sparse edge: cuboid 구조 일관성
- coord Huber: 좌표 정확도 보조

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v9_train/
이미지 수:  8,500
구성:
  Isaac Sim 500 (train/에서 랜덤 샘플링)
  Blender 2000 (blender_train_manydir/ 전체)
  test_indoor_v1 1000 (실내 촬영)
  test_blender_4000 4000 (신규 Blender)
  test_blender_v1000 1000 (다양한 시점)
```

## 평가 데이터

```
blender_dark          100장     어두운 팔레트 (val 평가)
```

## 평가

```
평가 조건                  PCK@3px   PCK@5px   PCK@10px   PnP Rate   Reproj mean
──────────────────────────────────────────────────────────────────────────────────
blender_dark 100장         0.435     0.595     0.744      46.0%      219.67 px
```

학습 시간: 11시간 34분 (22:43 → 10:17)

## 설계 의도

BPnP가 불안정하므로, PnP 없이 2D 수준에서 구조 제약을 거는 방향.

- flip equivariance: 모델이 shortcut 대신 구조를 배우도록 직접 regularize
- sparse edge: 점들이 개별로 맞는 게 아니라 전체 cuboid shape가 맞도록
- coord Huber: heatmap MSE보다 좌표 정확도를 직접 잡음
- 모든 loss는 object diagonal normalize + Huber (outlier robust)
- ramp-up: epoch 10-20 동안 선형 증가 (급격한 loss 변화 방지)
- pseudo-label에서는 struct loss OFF 또는 약화 (noisy label 증폭 방지)

## noapril 결과 (실패)

```
noapril 188장:  0/9 kps 144장 (76.6%), avg ~0.2/9 — 거의 전멸
real_data 20장: 9/9 kps 7장 (35%) — real_data에서는 일부 작동
```

v8 대비 noapril에서 16배 성능 하락. 원인 분석:
- v10 (같은 데이터, struct loss 없음)도 동일하게 전멸 → **데이터 구성 변경이 주범**
- Isaac Sim 2000→500 감소가 noapril 도메인 커버리지를 threshold 이하로 떨어뜨림
- struct loss는 부차적 원인 (belief peak 약간 억제)

## v8 기반 Ablation 결과 (핵심)

v8(ep60)에서 5ep finetune, mixed_v8_train(9000장) 사용:

```
실험              loss 설정                 noapril 9/9   PnP   B∧C
──────────────────────────────────────────────────────────────────
A (coord)         coord λ=0.003            38           117    6  ★★★
E (rel)           rel λ=0.005              43           118    2
B (edge)          edge λ=0.003             36            -     4
C (coord+edge)    coord+edge               23            -     0
D (flip)          flip λ=0.02              11            -     1
```

**v8_A (coord-only)를 메인 모델로 확정.**

## A+E mixed 실험

v8_A(ep65)에서 3ep 더 학습:

```
실험                   PnP    B∧C   결론
──────────────────────────────────────────
v8_A (ep65)            117     6    ★ 최적점
v8_A_control (ep68)     83     5    3ep 더 → PnP 하락
v8_AE_mixed (ep68)      91     3    E 추가 → C 악화
```

3ep 더 학습하면 keypoint 검출은 좋아지지만 pose quality는 하락.
v8_A ep65가 sweet spot.
