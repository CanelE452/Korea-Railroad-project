# 9. 수식 정의

> 키포인트 ID 매핑 및 3D 좌표 convention → [keypoint_definition.md](../preprocessing/keypoint_definition.md) 참조.

## 9.1 기호 정의 (Notation)

```
기호                       정의
──────────────────────────────────────────────────────────────────────────────────
x                          입력 이미지
f_θ                        DOPE 모델 (파라미터 θ) — 6 stage belief + affinity head
B̂_s = f_θ(x)[bel]_s       stage s 의 예측 belief map, shape (9, H', W'), s = 1..6
Â_s = f_θ(x)[aff]_s       stage s 의 예측 affinity field, shape (16, H', W')
B, A                       GT belief map / affinity field (9 / 16 channel)
p̂ᵢ = softargmax(B̂_s,last)_i   soft-argmax 로 얻은 i 번째 예측 keypoint (i = 1..8)
pᵢ                          GT i 번째 keypoint (sub-pixel)
D_s = {(xⱼˢ, yⱼˢ)}        synthetic labeled dataset
D_r = {xⱼʳ}                real unlabeled dataset
D̃_r = {(xⱼʳ, ỹⱼʳ)}       geo filter 통과한 pseudo-labeled real dataset
K                          카메라 내부 파라미터 행렬
K_3D = {K_3D_i ∈ R³}       canonical Y=UP pallet cuboid corners (i = 0..7)
```

## 9.2 Loss 정의

DOPE 는 6 stage belief + affinity head 를 가진다. Base loss 는 두 head 에
대한 stage-wise MSE 합, 여기에 본 연구는 **soft-argmax coordinate Huber loss**
를 추가 항으로 도입한다 (Sun et al., *Integral Human Pose Regression*, ECCV
2018 — heatmap 의 sub-pixel 위치를 직접 supervise 하여 discretization bias
를 제거). coord loss 는 v8_A fine-tune ablation 에서 noapril PnP rate
49.5% → 62.2% 로 유의미한 개선을 보였으며, edge/flip/VP 등 다른 structural
loss 는 동일 설정에서 모두 실패했다 (§11.3 Loss Ablation).

### 9.2.1 DOPE base loss (belief + affinity)

```
              6
   L_belief = Σ  ‖ B̂_s - B ‖²_F / N_b
              s=1

              6
   L_affinity = Σ  ‖ Â_s - A ‖²_F / N_a
                s=1

   L_base = L_belief + L_affinity
```

(`N_b`, `N_a` 는 각각 belief / affinity tensor 의 요소 수 — `.mean()` 정규화.)

### 9.2.2 Coord Huber loss (contribution)

마지막 stage 의 belief map 에 soft-argmax 를 적용하여 sub-pixel keypoint
좌표 `p̂ᵢ` 를 얻고, GT 좌표 `pᵢ` 와의 Huber loss 를 계산한다.

```
                 8
   L_coord = (1/8) Σ  Huber_δ ( p̂ᵢ - pᵢ )
                 i=1

   Huber_δ(r) = { ½‖r‖²            if ‖r‖ ≤ δ
                { δ(‖r‖ - ½δ)       otherwise        (δ = 0.03 정규화 좌표)
```

soft-argmax 는 belief map softmax 의 기댓값으로 정의:
```
   p̂ᵢ = Σ_{u,v}  softmax(B̂_{6,i})(u,v) · (u, v)
```

### 9.2.3 Synthetic / Pseudo-real loss

**Synthetic (GT 있음):**
```
   L_syn = L_base(B̂, B, Â, A) + λ_coord · L_coord(p̂, p)     (λ_coord = 0.003)
```

**Pseudo-real (belief 만 — affinity/coord GT 없음):**

Pseudo-label 은 2D keypoint 로부터 `CreateBeliefMap(σ=2.0)` 으로 belief 만
재구성된다. affinity GT 는 없고, keypoint 위치는 belief peak 와 동치이므로
coord loss 도 의미가 없다. 따라서:
```
                           6
   L_real = L_belief_real = Σ  ‖ B̂_s^r - B̃^r ‖²_F / N_b
                            s=1
```

여기서 `B̃^r` 은 RANSAC subset consensus 를 통과한 pseudo-label keypoint 로
부터 생성된 belief map.

### 9.2.4 Total loss

```
              1                              1
   L(θ) = ─── Σ  L_syn(xⱼˢ, yⱼˢ; θ)  +  α · ──── Σ  L_real(xⱼʳ, ỹⱼʳ; θ)
          |D_s| j                          |D̃_r| j

   (α: pseudo-label 가중치, default 1.0)
```

논문 Loss Ablation 에서는 `λ_coord ∈ {0, 0.003}` 와 edge/flip 변종을 비교한다
(§11.3 참조).

## 9.3 Geometric Filter 조건 (RANSAC Subset Consensus)

**D̃_r 에 포함되려면 아래 조건을 모두 만족 (2026-04-11 개정):**

```
[1] Pre-filter:
    |{ i : f_θ(xⱼʳ)_i is detected }| ≥ min_keypoints (= 5)

[2] RANSAC subset consensus (main gate):
    n_iter = 50 회 반복:
        S ⊂ {detected indices}, |S| = 5 을 랜덤 선택
        (R_S, t_S) = EPnP(K_3D[S], K_2D[S])
        c_S = |{ i : ‖ π(K_3D_i; R_S, t_S) - K_2D_i ‖ < 5 px }|

    c* = max_S c_S,  (R*, t*) = argmax_S c_S

    c* ≥ min_consensus (= 6)

[3] LOO cross-validation:
    D = max_{i,j} ‖ π(K_3D_i; R*, t*) - π(K_3D_j; R*, t*) ‖   (projected diagonal)

    for each detected keypoint i:
        S_-i = detected \ {i}
        (R_-i, t_-i) = EPnP(K_3D[S_-i], K_2D[S_-i])
        e_i = ‖ π(K_3D_i; R_-i, t_-i) - K_2D_i ‖ / D

    median(e_i) < τ_LOO (= 0.05)
```

여기서 `π(·; R, t)` 는 카메라 행렬 K 로의 투영, `K_3D_i` 는 canonical Y=UP
pallet corner (i = 0..7). [3] LOO 는 각 keypoint 를 하나씩 제외하고 PnP 를
재추정하여, 제외된 점의 reproj error 가 일정 수준 이하인지 확인한다.
위 세 조건을 모두 통과한 프레임 j 의 (R*, t*) 로부터
pseudo-label ỹⱼʳ 를 구성한다.

Ablation 비교용으로 `filter_type ∈ {ransac, bc, conf, none}` dispatcher 가
같은 데이터에 대해 서로 다른 필터 정의를 적용할 수 있다 — 실제 ablation
수식과 그 P/R 결과는 `_docs/filter/2026-04-11_selection.md` 참조.

---

# 10. 평가 메트릭

본 연구는 데이터셋 별로 사용 가능한 metric 이 다르다. 혼란을 피하기 위해
먼저 **적용 가능성 매트릭스** 를 정의하고, 이어서 각 metric 을 서술한다.

## 10.0 Metric × Dataset 적용 가능성

```
Metric                 capture0403middle   Seen / Unseen (AprilTag GT)   Synthetic Val
────────────────────────────────────────────────────────────────────────────────────────
ADD (3D)                 ✗ (frame conv.)    ✅ primary                    ✅
5 cm 5° (3D)             ✗ (frame conv.)    ✅ primary                    ✅
Reproj error (2D)        ✅ primary          ✅ secondary                  ✅
PnP success rate         ✅                  ✅                            ✅
Filter P / R / F1        ✅ (GT 기반)        ✅                            (n/a)
PCK @ k px               (n/a, 2D GT)       ✅                            ✅ primary
Volume ratio (Appendix)  ✅ (참고)           (n/a)                         ✅
```

**핵심 규칙**:
- `capture0403middle` 은 object-frame convention mismatch 로 ADD / 5 cm 5°
  를 신뢰할 수 없다. 논문 본문 표 어디에도 이 데이터셋의 3D 수치를 넣지
  않는다. Primary 지표는 **2D reproj ≤ 50 px** + **PnP success rate**.
- Real Seen / Unseen (촬영 완료 후) 에서만 ADD, 5 cm 5° 를 primary 로 쓴다.
- Synthetic val 은 1 차 screening 용 (PCK @ 3 px) 이며 최종 순위 결정 권한
  없음.

## 10.1 ADD — Average Distance of Model Points (3D)

```python
def compute_ADD(R_gt, t_gt, R_pred, t_pred, model_points):
    transformed_gt = (R_gt @ model_points.T).T + t_gt
    transformed_pred = (R_pred @ model_points.T).T + t_pred
    add = np.mean(np.linalg.norm(transformed_gt - transformed_pred, axis=1))
    diameter = compute_diameter(model_points)
    return add, add < 0.1 * diameter  # ADD < 10% diameter → 성공
```

**적용**: AprilTag GT 가 있는 Real Seen / Unseen + Synthetic Val 에만 사용.

## 10.2 Reprojection Error (2D)

```python
def compute_reproj_error(kp_gt_2d, kp_pred_2d):
    """mean L2 distance between 2D projected cuboid GT and predicted keypoints."""
    return np.mean(np.linalg.norm(kp_gt_2d - kp_pred_2d, axis=1))
```

**임계값**: capture0403middle 에서 "good" 판정 기준은 mean reproj ≤ **50 px**
(frame 전체 평균). 이 값은 3D 기준 약 5 ~ 10 cm pose error 에 대응하며,
filter selection (§11.4) 과 모델 평가 (§11.2) 의 primary metric.

**적용**: 전 데이터셋. capture0403middle 의 primary, Seen/Unseen 의 secondary,
synthetic val 의 보조 지표.

## 10.3 5 cm 5° Metric (3D)

```python
def compute_5cm_5deg(R_gt, t_gt, R_pred, t_pred):
    trans_error = np.linalg.norm(t_gt - t_pred) * 100  # cm
    R_diff = R_gt @ R_pred.T
    angle_error = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    return (trans_error < 5.0) and (angle_error < 5.0)
```

**적용**: AprilTag GT 가 있는 Real Seen / Unseen + Synthetic Val. 산업 현장
로봇 파지 정밀도에 대응하는 실용 임계값.

## 10.4 PnP Success Rate

```python
def compute_pnp_rate(per_frame_results):
    """Fraction of frames where RANSAC subset consensus succeeded."""
    n_total = len(per_frame_results)
    n_success = sum(
        1 for r in per_frame_results
        if r["consensus"] >= 6 and r["size_pass"]
    )
    return n_success / n_total
```

Frame level 에서 PnP (RANSAC subset consensus + LOO cross-validation) 가 성공한 비율.
"모델이 몇 % 의 real frame 에서 pose 를 뽑아낼 수 있는가" 의 실용 지표이자,
self-training 의 yield 직접 측정치. capture0403middle / noapril 에서 본 연구
의 **가장 많이 사용되는 실용 metric** 이며 coord loss / ST 효과를 가장 먼저
드러낸다 (baseline 49.5 → v8_A 62.2 → ST R1 78.7).

**적용**: 전 real 데이터셋. GT 가 없는 noapril pool 에서도 측정 가능.

## 10.5 Filter Precision / Recall / F1

GT 기반 pseudo-label 품질 평가 지표. capture0403middle 440 프레임의 GT 로
"good" / "bad" 를 구분하고 (§10.2 기준), 필터가 accept 한 frame 을 예측으로
보아 P/R/F1 을 계산한다.

```python
def compute_filter_pr(per_frame, good_threshold_px=50.0):
    TP = sum(1 for r in per_frame if r["filter_pass"] and r["reproj"] < good_threshold_px)
    FP = sum(1 for r in per_frame if r["filter_pass"] and r["reproj"] >= good_threshold_px)
    FN = sum(1 for r in per_frame if not r["filter_pass"] and r["reproj"] < good_threshold_px)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
```

**적용**: 필터 선정 (§11.4), 필터 ablation (§11.2), consensus threshold
민감도 (§11.9) 의 핵심 metric.

## 10.6 PCK @ k px (Synthetic Val screening)

```python
def compute_pck(kp_pred, kp_gt, k_px=3.0):
    """Fraction of keypoints within k pixels of GT."""
    dists = np.linalg.norm(kp_pred - kp_gt, axis=-1)
    return (dists < k_px).mean()
```

**적용**: Synthetic val 에서 1 차 screening 용. `PCK @ 3 px` 이 너무 낮은
모델은 real test 단계로 진행 없이 탈락. 최종 순위 결정에는 사용하지 않는다.

## 10.7 Appendix: 3D Volume Ratio (참고용, 본문 미사용)

예측된 2D keypoint 를 PnP depth 로 back-project 하여 3D cuboid 부피를 추정
하고 GT 부피와 비교하는 비표준 metric. v8 이전 비교에서 사용했으나, 다른
논문에서 쓰지 않는 지표라 reviewer 혼동을 피하기 위해 **본문 표에 넣지
않는다**. Synthetic val screening 의 참고 지표로만 사용.

```
V_gt      = width × depth × height   (pallet 고유 규격)
V_pred    = ‖P₁ − P₀‖ · ‖P₃ − P₀‖ · ‖P₄ − P₀‖   (back-projected edges)
ratio     = V_pred / V_gt            (1.0 = perfect)
```

- `|ratio − 1| < 0.2`: 부피 오차 20% 이내 비율
- `|ratio − 1| < 0.5`: 부피 오차 50% 이내 비율
