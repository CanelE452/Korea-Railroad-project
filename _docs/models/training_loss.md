# DOPE Training Loss

## 기본 Loss (항상 활성)

```
total_loss = loss_belief + loss_affinities [+ geo_lambda × loss_geo]
```

```
Loss               수식                  대상                                          채널 수
──────────────────────────────────────────────────────────────────────────────────────────────────
loss_belief        mean((pred - gt)²)    9개 keypoint 히트맵 (8 corner + 1 centroid)   9
loss_affinities    mean((pred - gt)²)    16개 affinity field (8 edge × 2 방향)         16
```

## Symmetric Loss (`--symmetric_loss` 플래그로 활성화)

팔레트처럼 앞/뒤가 시각적으로 동일한 대칭 물체를 위한 loss.
GT belief map의 180° swap 버전도 정답으로 인정.

```
swap 매핑: 0↔5, 1↔4, 2↔7, 3↔6, centroid(8) 유지  (180° Y축 회전, 좌우 뒤집힘 반영)

loss_belief = min(MSE(pred, gt_orig), MSE(pred, gt_swapped))
```

- 앞/뒤 방향 구분 포기 (180° yaw 모호성)
- 앞 vs 옆은 991mm vs 1192mm로 시각적 구분 가능 → 90° swap 불필요
- `--symmetric_loss` 플래그 없으면 기존과 동일

### 멀티스케일 Supervision

```python
for stage in range(len(output_aff)):
    loss_affinities += mean((output_aff[stage] - target_affinities)²)
    loss_belief += mean((output_belief[stage] - target_belief)²)
```

VGG-19의 각 stage 출력마다 loss를 누적 — 중간 레이어에서도 학습 신호 전달.

### GT Belief Map 생성

- 각 keypoint 위치에 `sigma=4.0`의 2D Gaussian을 찍어서 GT 히트맵 생성
- 출력 해상도: 50×50 (input 448 → output 50)
- `sigma < 1`이면 gradient vanishing 발생

## Structural Loss (`--struct_loss` 플래그로 활성화)

Belief map MSE와 병행하여 keypoint 좌표/구조를 직접 최적화하는 loss.
DOPE 모델 구조 변경 없음 — loss 계산용으로만 사용.

### Soft-Argmax vs Argmax

```
argmax (기존 DOPE inference):
  heatmap에서 가장 큰 값의 픽셀 인덱스 반환
  (x, y) = argmax(belief_map) -> 정수 좌표
  미분 불가능 -> gradient가 흐르지 않아서 loss로 사용 불가

soft-argmax (structural loss에서 사용):
  heatmap을 softmax로 확률 분포로 변환 후, 좌표의 가중 평균 계산
  weights = softmax(belief_map / temperature)  # 합=1
  x = sum(weights * x_coords)  # 기대값
  y = sum(weights * y_coords)
  실수 좌표 반환 (예: 23.7, 41.2)
  미분 가능 -> "좌표가 틀리면 heatmap을 고쳐라" gradient 전달 가능
```

기존 DOPE MSE는 heatmap 전체 픽셀 값을 맞추는 loss라 peak 위치가 약간 밀려도 loss가 크게 변하지 않는다.
Soft-argmax 기반 coord loss는 추출된 (x,y) 좌표를 직접 비교하므로 위치 정밀도를 강제한다.

### 구성 요소 (3가지, 각각 독립 on/off 가능)

```
Loss               수식                                              효과
───────────────────────────────────────────────────────────────────────────────────────────
struct/coord       Huber(soft_argmax(pred) - soft_argmax(gt)) / D    좌표 정밀도 직접 강제
struct/edge        Huber(pred edge lengths - gt edge lengths) / D    cuboid 변 길이 보존
struct/flip        FlipEquivariance(pred, flip(input))               좌우 반전 일관성
```

D = object diagonal (크기 불변 정규화)

### 파라미터

```
파라미터             CLI flag           기본값     설명
───────────────────────────────────────────────────────────────────────────────────
활성화               --struct_loss      off       플래그 추가 시 활성화
coord 가중치        --struct_coord     0.10      좌표 Huber loss 스케일
edge 가중치         --struct_edge      0.05      edge length loss 스케일
flip 가중치         --struct_flip      0.02      flip equivariance loss 스케일
Huber delta         --struct_delta     0.03      Huber loss transition point
warmup              --struct_warmup    10        활성화 시작 epoch (이후 10 epoch ramp-up)
```

### Ablation 실험 (v9, base=mixed_v8)

```
Ablation    coord   edge    flip   noapril PnP   B pass   C pass   B^C
─────────────────────────────────────────────────────────────────────────
v8 (base)   -       -       -      49.5%          1        0        0
A (coord)   0.003   0       0      62.2%          16       6        6
B (edge)    0       0.003   0      54.3%          20       4        4
C (co+ed)   0.003   0.002   0      55.3%          13       2        0
D (flip)    0       0       0.02   29.3%          7        2        1
E (rel)     rel_lambda=0.005       62.8%          25       4        2
```

coord loss만으로 PnP 49.5% -> 62.2%, B^C 0 -> 6장. keypoint 양쪽 분포 + PnP 안정성 대폭 개선.

### 사용법

```bash
# coord-only (ablation A)
bash scripts/train_dope.sh --finetune --exp_name v9_A \
    --struct_loss --struct_coord 0.003 --struct_edge 0 --struct_flip 0

# 전체 structural loss
bash scripts/train_dope.sh --finetune --exp_name v9_full \
    --struct_loss --struct_coord 0.003 --struct_edge 0.05 --struct_flip 0.02
```

코드: `Deep_Object_Pose/train/geo_loss.py` (StructuralLoss 클래스)

## Geometric Loss (`--geo_loss` 플래그로 활성화)

Soft-argmax + BPnP(Backpropagatable PnP)로 3D 기하학적 제약 추가.
DOPE 모델 구조는 변경 없음 — loss 계산용으로만 사용, inference 시 제거.

```
학습 시:  이미지 → DOPE → belief map → soft-argmax → BPnP → 3D loss
                     ↑                                          ↓
                     └──────────── gradient 전달 ────────────────┘

추론 시:  이미지 → DOPE → belief map → argmax → PnP (기존 그대로)
```

### Geometric Loss 구성

```
Loss               단계       수식                                    필요 기술
───────────────────────────────────────────────────────────────────────────────────────────
geo/kp_l2          soft-argmax    ||pred_kp - gt_kp||²                    soft-argmax
geo/diagonal       soft-argmax    cuboid 대각선 중점 불일치                soft-argmax
geo/reproj         BPnP           reproject(R,t) vs gt_kp 거리           soft-argmax + BPnP
geo/volume         BPnP           ||V_pred/V_gt - 1||²                   soft-argmax + BPnP
geo/add            BPnP           avg 3D point distance (pred vs gt)     soft-argmax + BPnP
```

### 파라미터

```
파라미터             CLI flag           기본값     설명
───────────────────────────────────────────────────────────────────────────────────
활성화               --geo_loss         off       플래그 추가 시 활성화
전체 가중치          --geo_lambda       0.1       geometric loss 전체 스케일
BPnP warmup         --geo_warmup       5         PnP 기반 loss 활성화 시작 epoch
soft-argmax 온도    --geo_temperature  1.0       낮을수록 sharp, 높을수록 smooth
카메라 내부 파라미터  --geo_fx/fy/cx/cy  D435i     원본 이미지 기준 intrinsics
원본 해상도          --geo_img_w/h      640/480   합성 데이터 원본 크기
```

### BPnP (Backpropagatable PnP) 작동 원리

```
Forward:  cv2.solvePnP(EPnP) — 기존 OpenCV PnP 그대로 사용
Backward: implicit function theorem으로 gradient 계산
          d(pose)/d(kp2d) = (J^T J + λI)^{-1} J^T
          (J = projection Jacobian, λ = damping)
```

- 학습 파라미터 없음 (순수 수학 연산)
- PnP 실패 시 해당 샘플의 geometric loss 자동 skip (validity mask)
- 코드: `Deep_Object_Pose/train/geo_loss.py`

### 사용법

```bash
# 기본 학습 (geometric loss 없음 — 기존과 동일)
bash scripts/train_dope.sh --exp_name mixed_v1

# Geometric loss 포함 학습
bash scripts/train_dope.sh --exp_name mixed_v3 --geo_loss --geo_lambda 0.1

# Geometric loss + finetune
bash scripts/train_dope.sh --finetune --exp_name mixed_v3_geo --geo_loss --geo_lambda 0.1 --geo_warmup 0
```
