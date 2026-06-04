# Pseudo-Label Filtering 기법 서베이 (Self-Training 6D Pose)

작성일: 2026-06-02
목적: 팔레트 6D pose self-training의 pseudo-label(PL) filter 개선. 현재 geometric RANSAC subset-consensus 필터(n_iter=50, k=5, τ=5px, c≥6) + size sanity가 unlabeled pool 확대에 따라 noisy PL을 너무 많이 통과시킴. **precision(통과 PL 순도) 향상**이 핵심.

---

## 개요

SSL/self-training PL filtering 문헌은 크게 4계열로 정리된다.

1. **Confidence thresholding** (FixMatch → FlexMatch → Dash → FreeMatch): 고정 임계 → class/시점 adaptive 임계로 진화. 핵심 교훈은 "단일 고정 threshold는 학습 진행에 따라 항상 최적이 아니다".
2. **Uncertainty-aware selection** (UPS, Seq-UPS, deep ensemble, MC-dropout): confidence만으로는 mis-calibration 때문에 noise가 새어 들어가므로, uncertainty를 **AND 게이트**로 추가해 false-positive PL을 제거.
3. **6D pose self-training** (Self6D, ECCV'22 bin-picking, Pseudo-Flow-Consistency): classification confidence가 없는 pose 문제에서는 **render-and-compare(2D appearance) + 3D geometry consistency**를 PL 품질 proxy로 사용. 우리 RANSAC reproj consensus와 직접 대응.
4. **Curriculum / adaptive threshold** (Dash, CPL): 초기엔 strict(high-precision) → 후기엔 relax. 또는 데이터 분포 기반 `μ+σ` 동적 임계.
5. **TTA consistency**: weak/strong aug 또는 multi-view 예측 일치도를 PL 신뢰 척도로.

우리에게 가장 직접적인 건 **(3) 6D pose self-training의 adaptive consensus** + **(2) uncertainty AND 게이트**의 결합이다.

---

## 논문별 요약

| # | 제목 / 저자 / 연도 | 핵심 아이디어 | 우리 필터에 적용할 take-away |
|---|---|---|---|
| 1 | **FixMatch** (Sohn et al., NeurIPS 2020) | weak-aug 예측이 고정 confidence threshold(0.95) 넘으면 그 hard label을 strong-aug에 supervision으로 사용 | 단일 strict cutoff가 baseline. 우리 `c≥6 / τ=5px`가 이에 대응하나, 고정값이라 pool 분포 변화에 둔감 |
| 2 | **FlexMatch / Curriculum Pseudo-Labeling** (Zhang et al., NeurIPS 2021) | class별로 학습 난이도를 추정해 class-specific threshold를 동적 조정. 쉬운 시점/클래스는 임계 ↑, 어려운 건 ↓ | "한 종류의 PL에 한 임계"는 비효율. 우리는 **per-domain(scene/팔레트 종류/시점)** 으로 threshold를 쪼갤 근거 |
| 3 | **Dash** (Xu et al., ICML 2021) | 학습 진행에 따라 global threshold를 **점진적으로 강화(grow)**. 초기엔 loose, 후기엔 strict하게 손실 기준 cutoff 상승 | curriculum 방향의 정량 근거. self-training round가 진행될수록 strict하게 죄는 schedule이 confirmation bias를 줄임 |
| 4 | **FreeMatch** (Wang et al., ICLR 2023) | global + class-local threshold를 unlabeled confidence의 **EMA로 self-adaptive** 조정. class-fairness regularization 추가. FlexMatch 대비 CIFAR-10(1-label) 5.78%↓ | **EMA 기반 분포 추종 threshold**가 hand-tuned 고정값보다 강함. 우리도 reproj-error/consensus 분포의 EMA로 cutoff를 자동 산출 가능 |
| 5 | **UPS: In Defense of Pseudo-Labeling** (Rizve et al., ICLR 2021) | confidence와 **uncertainty를 분리한 이중 게이트**: `p_c > τ AND u(p_c) < κ`. MC-dropout으로 uncertainty 추정. mis-calibrated high-confidence noise 제거. negative learning 도입 | **핵심 차용 포인트**: confidence(=우리 belief peak/PnP inlier)와 uncertainty(=consistency 분산)를 **AND로 결합**. 둘 다 통과해야 PL 채택 → precision 직접 상승 |
| 6 | **Seq-UPS** (Patel et al., 2022) | UPS를 시퀀스(text recognition) 도메인으로 확장. MC-dropout 다중 forward의 분산을 uncertainty로, teacher-forcing으로 sample 간 prediction consistency 강제 | keypoint sequence(8 corner + centroid)에도 forward 분산 기반 uncertainty 적용 가능. dropout 다중 추론으로 keypoint별 분산 측정 |
| 7 | **Self6D** (Wang et al., ECCV 2020) | neural rendering 기반 visual + geometric alignment로 6D pose self-supervision (PL 없는 self-sup) | render-and-compare가 pose 품질의 강한 proxy. 우리 reproj consensus의 상위호환 신호로 silhouette/mask IoU 추가 고려 |
| 8 | **Sim-to-Real 6D Pose via Iterative Self-Training** (Chen et al., ECCV 2022, bin-picking) | **우리와 가장 유사.** teacher가 real unlabeled에 pose 예측 → (a) 2D appearance: mask overlap × perceptual distance, (b) 3D geometry: Chamfer distance, 두 신호를 **AND 게이트**로 PL 선별. 임계는 **분포 기반 `τ = μ + σ`**(고정값 아님). student→new teacher 반복으로 PL 품질·정확도 동반 상승. ADD(-S) +11.49%/+22.62%, bin-picking 성공률 +19.54% | **가장 직접적 청사진**: ① 2D(appearance) + 3D(geometry) **상보적 다중 metric AND 합의**, ② threshold를 unlabeled metric 분포의 `μ+σ`로 **adaptive 산출**, ③ iterative re-labeling. 우리의 reproj-consensus는 geometry축, 여기에 **appearance축(렌더 silhouette IoU / crop perceptual dist)을 추가하면 precision 상승** |
| 9 | **Pseudo Flow Consistency for Self-Sup 6D Pose** (Hai et al., ICCV 2023) | pure RGB. pixel-level **flow consistency**를 학습 이미지 쌍 사이 geometry 제약으로. 동적 PL 생성 | 보조 정보 없이 RGB만으로 geometry consistency 측정. multi-view/연속 프레임이 있다면 flow consistency가 추가 필터 신호 |
| 10 | **Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017) / **MC-Dropout** (Gal & Ghahramani, ICML 2016) | 다중 모델/다중 dropout forward의 예측 분산 = predictive uncertainty. ensemble이 MC-dropout보다 calibration 우수하나 비용 큼 | uncertainty 신호 구현 옵션. 비용이 부담이면 MC-dropout(저비용), 정확도 우선이면 small ensemble. keypoint heatmap의 forward 간 분산으로 PL 신뢰도 정량화 |

---

## 우리 4방향에 대한 시사점

### A. top-K / 백분위 quality 수량 제어
- **근거**: Dash·FreeMatch·CPL 모두 "절대 임계 고정"보다 **분포 기준 상대 선택**이 강함을 보임. 백분위(top-K%) 선택은 pool 크기/난이도가 바뀌어도 통과 PL의 *순도 분포*를 일정하게 유지.
- **시사점**: 현재 `c≥6 / τ=5px` 고정 cutoff는 pool이 커지면 절대 통과 수만 늘 뿐 순도가 떨어진다. **reproj-consensus 점수 상위 K%만 채택**(예: per-round top 30%)하면 noisy tail을 구조적으로 잘라낸다. 단, top-K는 "전부 나쁜 round"에서도 K%를 강제로 뽑는 위험 → **절대 floor(c≥6)와 AND**로 안전판 결합 권장.
- **주의**: FlexMatch 교훈상 K를 **per-domain/시점별로** 잡아야 한 시점이 PL을 독식하지 않음.

### B. consensus / strict threshold 강화
- **근거**: UPS·bin-picking 모두 **단일 metric → 다중 metric AND**로 precision을 끌어올림. FixMatch 0.95처럼 strict cutoff는 recall은 줄지만 self-training에선 precision이 더 중요(confirmation bias).
- **시사점**: 현재 RANSAC consensus(c≥6, τ=5px)를 **τ를 더 죄거나(예 3px)** consensus 비율 기준(8개 중 7개 이상)으로 강화. 다만 단순 강화는 recall 급감 → 차원을 늘리는 **C(결합 게이트)** 가 더 효율적.

### C. confidence × geometry 결합 게이트 ★ (가장 유망)
- **근거**: UPS의 `p_c>τ AND u<κ`, bin-picking의 `d_a<τ_a AND d_g<τ_g` — **서로 다른 실패 모드를 잡는 상보적 신호의 AND 게이트**가 문헌의 일관된 best practice. confidence는 in-plane/texture 오류, geometry는 out-of-plane/scale 오류를 잡음.
- **시사점**: 우리는 geometry축(RANSAC reproj consensus)만 있고 **confidence축이 약하다**. 추가 신호 후보:
  - **belief-map peak sharpness/height** (DOPE heatmap 신뢰도) → confidence축
  - **PnP RANSAC inlier ratio** → geometry confidence
  - **렌더 silhouette IoU / crop perceptual distance** (Self6D·bin-picking식 appearance축)
  - **MC-dropout keypoint 분산** (UPS식 uncertainty축)
  - 게이트: `(geometry consensus 통과) AND (confidence ≥ τ_c) AND (uncertainty ≤ κ)`. 셋 다 통과만 PL 채택.

### D. per-domain adaptive threshold
- **근거**: FlexMatch(class-local), FreeMatch(global+local EMA), bin-picking(`μ+σ` 분포 기반) 모두 단일 global 임계의 한계를 보임.
- **시사점**: 팔레트 종류(scene 1~4)/카메라 시점/배경 난이도별로 reproj-error 분포가 다르므로, **각 domain의 통과 metric 분포에서 `μ+σ` 또는 백분위로 threshold를 자동 산출**. 고정 5px를 모든 도메인에 적용하면 쉬운 도메인은 noise 통과, 어려운 도메인은 PL 고갈. EMA로 round마다 갱신(FreeMatch식).

**결론 우선순위**: C(결합 게이트) ≈ D(adaptive) > A(top-K) > B(단순 강화). C+D를 함께 가는 것이 문헌상 가장 검증된 조합이며 bin-picking 논문이 정확히 그 형태(다중 metric AND + `μ+σ` adaptive + iterative).

---

## 추가 제안 필터 후보 (구체적 구현)

### 후보 1: Dual-Gate Adaptive Consensus Filter (DGAC) — 최우선 추천
bin-picking(ECCV'22) + UPS를 우리 keypoint 파이프라인에 이식. **C + D 동시 충족**.

```
입력: teacher가 unlabeled 프레임 i에 예측한 keypoints K_i, PnP pose T_i, belief maps B_i
도메인 d = domain_of(i)   # 팔레트 종류 / 시점 bin

# --- 축 1: Geometry consensus (기존 RANSAC, 강화) ---
g_i = ransac_reproj_inlier_ratio(K_i, T_i)        # 8 corner 중 inlier 비율 (0~1)

# --- 축 2: Confidence (DOPE belief) ---
c_i = mean_peak_height(B_i)                         # keypoint별 belief 피크 평균

# --- 축 3: Uncertainty (MC-dropout, M회 forward) ---
u_i = mean_keypoint_std(MC_dropout_forward(frame_i, M=5))   # px 단위 분산

# --- 도메인별 adaptive threshold (EMA로 round마다 갱신) ---
τ_g[d] = ema(percentile(g_dist[d], 50))            # 또는 μ_g[d] (상위 절반)
τ_c[d] = ema(μ_c[d] - σ_c[d])
κ_u[d] = ema(μ_u[d] + σ_u[d])

# --- AND 게이트 ---
accept_i = (g_i ≥ max(τ_g[d], 6/8))   # 절대 floor와 AND
           AND (c_i ≥ τ_c[d])
           AND (u_i ≤ κ_u[d])
```
- 핵심: 세 축(geometry/confidence/uncertainty)을 **AND**로, threshold는 **per-domain EMA 분포 기반**.
- 비용: MC-dropout M=5회 추가 forward(unlabeled 1회성). 부담되면 축 3 생략하고 geometry×confidence 2축만으로도 UPS 대비 핵심은 유지.
- 기대: 절대 통과 수 대신 *순도*를 도메인별로 제어 → precision 직접 상승. iterative round마다 τ 강화(Dash식)로 confirmation bias 억제.

### 후보 2: Render-Consistency Re-Ranking Gate (RCR) — 보조/검증용
Self6D·bin-picking의 appearance축을 our geometry축에 **상보적으로 추가**. geometry consensus를 통과한 PL만 대상으로 2차 검증.

```
geometry 통과한 후보 PL에 대해:
  render_i = silhouette_render(pallet_USD, T_i, K_cam)   # 예측 pose로 USD 렌더
  iou_i    = mask_IoU(render_i, segmask_i)               # 관측 마스크와 silhouette IoU
  accept_2 = iou_i ≥ τ_iou[d]      # 도메인별 μ+σ 또는 top-K%
최종 PL = geometry_pass AND accept_2
```
- 이유: reproj consensus는 keypoint 위치만 보지만 **silhouette IoU는 scale/out-of-plane 오류(reproj가 놓치는 모드)** 를 잡음 → bin-picking의 "2D appearance + 3D geometry 상보성" 그대로.
- 비용: 후보당 1회 렌더(이미 USD 모델 보유). geometry 통과분에만 적용하므로 저렴.
- top-K(A방향)를 여기에 결합: IoU 상위 K%만 채택하면 수량 제어까지 동시.

---

## 참고문헌

1. Sohn et al. **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.** NeurIPS 2020. https://arxiv.org/abs/2001.07685
2. Zhang et al. **FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.** NeurIPS 2021. https://arxiv.org/abs/2110.08263
3. Xu et al. **Dash: Semi-Supervised Learning with Dynamic Thresholding.** ICML 2021. https://arxiv.org/abs/2109.00650
4. Wang et al. **FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning.** ICLR 2023. https://arxiv.org/abs/2205.07246
5. Rizve et al. **In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework (UPS).** ICLR 2021. https://arxiv.org/abs/2101.06329 · code: https://github.com/nayeemrizve/ups
6. Patel et al. **Seq-UPS: Sequential Uncertainty-aware Pseudo-label Selection for Semi-Supervised Text Recognition.** 2022. https://arxiv.org/abs/2209.00641
7. Wang et al. **Self6D: Self-Supervised Monocular 6D Object Pose Estimation.** ECCV 2020. https://arxiv.org/abs/2004.06468
8. Chen et al. **Sim-to-Real 6D Object Pose Estimation via Iterative Self-training for Robotic Bin Picking.** ECCV 2022. https://arxiv.org/abs/2204.07049
9. Hai et al. **Pseudo Flow Consistency for Self-Supervised 6D Object Pose Estimation.** ICCV 2023. https://arxiv.org/abs/2308.10016
10. Lakshminarayanan et al. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.** NeurIPS 2017. https://arxiv.org/abs/1612.01474
11. Gal & Ghahramani. **Dropout as a Bayesian Approximation (MC-Dropout).** ICML 2016. https://arxiv.org/abs/1506.02142
12. (survey) **A Review of Pseudo Labeling for Semi-Supervised Learning.** 2024. https://arxiv.org/abs/2408.07221
