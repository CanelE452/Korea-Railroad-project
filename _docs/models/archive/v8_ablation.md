# v8 Ablation — Structural / Reliability Loss

mixed_v8 위에 structural loss 구성 요소를 개별 ablation.
모든 모델: mixed_v8 (ep60) base, mixed_v8_train (9000장), 5 epoch finetune, LR=5e-5, batch=4.

## Ablation 설정

```
Ablation   설명                    coord    edge     flip     rel       Weight 경로
───────────────────────────────────────────────────────────────────────────────────────────────
A (coord)  좌표 Huber loss만       0.003    0        0        -         weights/v8_ablation_A_coord/
B (edge)   edge length loss만      0        0.003    0        -         weights/v8_ablation_B_edge/
C (co+ed)  coord + edge            0.003    0.002    0        -         weights/v8_ablation_C_coord_edge/
D (flip)   flip equivariance만     0        0        0.02     -         weights/v8_ablation_D_flip/
E (rel)    uncertainty-weighted     -        -        -        0.005     weights/v8_ablation_E_rel/
```

## Loss 상세

### A: Coordinate Huber Loss
```
L = Huber(||soft_argmax(pred) - soft_argmax(gt)|| / D, delta=0.03)
```
- soft-argmax로 heatmap에서 미분 가능한 좌표 추출
- GT 좌표와 직접 비교 (object diagonal D로 정규화)
- 모든 keypoint 동일 가중치

### B: Sparse Edge Loss
```
L = Huber(||pred_edge_lengths - gt_edge_lengths|| / D, delta=0.03)
```
- cuboid의 변 길이(edge length)를 pred vs GT로 비교
- 3D 형태 보존 목적

### C: Coord + Edge (A+B 결합)

### D: Flip Equivariance Loss
```
L = ||soft_argmax(pred) - flip_remap(soft_argmax(pred_flipped))||
```
- 이미지 좌우 반전 후 추론한 결과가 원본과 일치하는지 확인
- symmetric keypoint 매핑 적용

### E: Reliability-Aware Coordinate Loss
```
mu_i    = sum(softmax(heatmap) * coords)           <- 좌표 (가중 평균)
sigma_i = sum(softmax(heatmap) * (coords - mu_i)^2) <- uncertainty (가중 분산)

L = Huber(||mu_i - gt_i|| / D) / (sigma_i + eps) + 0.5 * log(sigma_i + eps)
```
- A와 같이 좌표를 직접 비교하되, 각 keypoint의 sigma(신뢰도)로 가중
- sigma 작은 점(confident) -> 강하게 맞춤
- sigma 큰 점(uncertain) -> 약하게 맞춤
- log(sigma) 항 -> sigma 무한대 방지 (heteroscedastic uncertainty weighting)

## noapril 추론 결과 (capture0403noapril, 188장)

```
             PnP Rate    A pass    B pass    C pass    B∧C
v8 (base)    49.5%       43장      1장       0장       0장
A (coord)    62.2%       19장      16장      6장       6장  << B∧C 최고
B (edge)     54.3%       43장      20장      4장       4장
C (co+ed)    55.3%       31장      13장      2장       0장
D (flip)     29.3%       19장      7장       2장       1장
E (rel)      62.8%       24장      25장      4장       2장  << PnP/B 최고
```

## 핵심 발견

1. **coord loss (A)가 B∧C 기준 최고** — 좌표 직접 비교가 양쪽 endpoint + PnP 안정성에 가장 효과적
2. **edge loss (B)는 B pass 최다(20)** 이지만 C pass에서 밀림 — 변 길이는 맞추지만 위치 정밀도 부족
3. **coord+edge (C)는 합쳤는데 오히려 악화** — edge loss가 coord loss와 간섭
4. **flip (D)는 최악** — PnP 49.5->29.3%로 성능 해침
5. **rel (E)는 PnP/B에서 최고** — uncertainty weighting이 분포를 넓히지만 C(LOO 안정성)에서 약함
6. **coord/rel loss가 flip equivariance를 깨뜨림** — A를 core gate에서 뺀 필터 설계 재확인

## v8ablationa real_data 추론 결과 (1924장)

```
Total:    1924장
PnP OK:   1522 (79.1%)
A pass:    152 (10.0%)
B pass:    340 (22.3%)
C pass:    554 (36.4%)
B∧C:       222 (11.5%)
```

## 비교 이미지

```
경로                                                  내용
──────────────────────────────────────────────────────────────────────────────────
data/pallet/eval_results/v8_vs_v8ablationa_compare/   v8 vs A, 필터별 2열 비교
data/pallet/eval_results/ablation_5model_compare/     v8/A/B/C/D, 필터별 5열 비교
data/pallet/eval_results/v8_A_E_compare/              v8/A/E, 필터별 3열 비교
```
