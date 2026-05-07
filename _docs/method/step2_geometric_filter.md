# Step 2: Real 이미지 예측 → Filter → Pseudo-label

## 4.0 설계 변경 공지 (2026-04-11)

초기 설계는 3단계 canonical geometric filter (A: flip consistency, B: visible
structural support, C: normalized LOO PnP, 선택적 D: diagonal incidence)였다.
2026-04-11 에 `capture0403middle/gt_final` 440 프레임 GT 기반 precision/recall
분석을 수행한 결과, canonical B∧C core gate 는 실측에서 recall 이 극도로 낮아
(F1 ≤ 0.235) 실용적 pseudo-label 수량을 확보하지 못했다. Threshold 를 3배까지
느슨하게 풀어도 RANSAC subset consensus 를 이기지 못함.

→ **Primary filter 를 RANSAC subset consensus 로 변경**. 상세 P/R 표와 선정
근거는 `_docs/filter/2026-04-11_selection.md` 참조 (filter 전용 폴더: `_docs/filter/README.md`).

Canonical filter (A/B/C/D) 는 논문 ablation 비교용으로 `scripts/data_prep/canonical_filters.py` 에
보존. 이 문서의 canonical 설명은 ablation 섹션에 해당.

## 4.1 개요 (현행)

Step 1에서 학습된 DOPE 모델로 real unlabeled 이미지에 대해 inference 하고,
**RANSAC subset consensus + LOO cross-validation** 를 통과한 pose 만 pseudo-label 로 채택한다.

핵심 아이디어: DOPE 가 예측한 9 개 2D keypoint 에 대해, 랜덤 5 개 subset 으로
EPnP 를 50 회 반복하여 후보 pose 를 생성하고, 각 후보에 대해 전체 detected
점의 reprojection inlier 수 (<5px) 로 voting 한다. 최대 consensus ≥ 6 이면
해당 pose 를 pseudo-label 로 accept. 이 방식은 2 개 이상 outlier 가 있어도
robust 하며, solver 와 filter 를 단일 단계로 통합한다.

구현: `scripts/self_training/geometric_filter.py` — `GeometricFilter.solve_and_validate()`
설정: `config/stage3_selftrain.yaml` — `geometric_filter` 섹션

## 4.2 Inference

```python
model.eval()
with torch.no_grad():
    belief_maps = model(real_image)                  # (9, 50, 50)
    keypoints_2d = extract_peaks(belief_maps)        # list[9] of (u, v, conf) or None
```

`extract_peaks()` 는 belief map 별 argmax 주변 5×5 가중 평균으로 sub-pixel
좌표를 구하고, 해당 belief peak 값을 `conf` 로 함께 반환한다. 이 `conf` 는
downstream 에서 `filter_type=conf` 브랜치의 판정과 PnP weighted refinement 에
사용된다.

구현: `scripts/self_training/self_train.py` — `extract_peaks()`.

## 4.3 RANSAC Subset Consensus (Primary)

### 4.3.1 알고리즘

```
입력: detected 2D keypoints (최소 5 개, 각각 (u, v, conf))
      대응 3D canonical keypoints (Y=UP cuboid corners)

반복 n_iter=50 회:
    랜덤 5 개 subset 선택 → EPnP → 후보 pose (rvec, tvec)
    전체 detected 점 재투영 → inlier 수 집계 (|err| < 5.0 px)
    최대 inlier 수 갱신 시 best pose 저장

최종 best_consensus >= 6 이고 LOO cross-validation 통과면 accept.
```

핵심 속성:
1. **2 개 이상 outlier 에 robust** — 일반적 RANSAC 특성
2. **Solver 와 filter 가 통합** — 외부에서 별도 PnP → 필터 할 필요 없음
3. **Consensus 가 직접 pose quality score** — heuristic threshold 아님

구현: `scripts/self_training/geometric_filter.py` — `GeometricFilter.solve_and_validate()`.

### 4.3.2 파라미터

```yaml
# config/stage3_selftrain.yaml — geometric_filter 섹션
filter_type: ransac              # ransac | bc | conf | none

ransac_n_iter: 50                # RANSAC 반복 횟수
ransac_subset: 5                 # subset 당 keypoint 수 (EPnP 최소값)
ransac_reproj_px: 5.0            # inlier 판정 거리 (px)
ransac_min_consensus: 6          # accept 임계 inlier 수 (sweet spot)

tau_size_min: 0.5                # 복원된 pallet width 하한 (m)
tau_size_max: 2.5                # 복원된 pallet width 상한 (m)

min_keypoints: 5                 # pre-filter: 최소 감지 keypoint 수
seed: 0                          # subset sampling RNG seed
```

모든 값은 GT 기반 P/R 분석 (2026-04-11) 으로 선정. `ransac_min_consensus` sweep
결과 4/5 는 precision 하락, 7/8 은 recall 급감 → **6 이 sweet spot**.

### 4.3.3 LOO cross-validation (post-check)

RANSAC 통과 후 각 detected keypoint i 를 하나씩 제외하고 나머지로 PnP 를
재추정한 뒤, 제외된 점의 reproj error 를 projected diagonal D 로 정규화.
`median(e_i / D) < τ_LOO (= 0.05)` 이면 accept.

RANSAC 은 inlier 수 기반이라 6 개 이상이 대략 맞으면 통과시키지만,
one-sided keypoint collapse (한쪽 면의 점들만 정확하고 반대쪽이 크게 밀린
경우) 를 잡지 못한다. LOO 는 각 점의 개별 안정성을 검증하여 이를 방지.

구현: `scripts/data_prep/canonical_filters.py` — `filter_C()` (τ = 0.05).

## 4.4 Filter Ablation Dispatch

논문 ablation 에서 **동일 코드 경로 / 동일 데이터 / 동일 학습 설정**으로
필터 타입만 바꿔 비교하기 위해 `filter_type` dispatcher 를 제공한다.

| filter_type | 역할 | 구현 경로 |
|---|---|---|
| `ransac` (default) | Primary — RANSAC subset consensus + LOO | `GeometricFilter.solve_and_validate()` + `filter_C()` |
| `bc` | Ablation — canonical B ∧ C + size | `canonical_filters.filter_B/C` + 외부 PnP |
| `conf` | Baseline — belief peak ≥ `conf_min` | 내부 dispatcher |
| `none` | Upper bound — min_keypoints 만 | 내부 dispatcher |

CLI 에서 `--filter_type {ransac,bc,conf,none}` 으로 override 가능. 동일 seed /
동일 unlabeled pool 에서 4 회 `self_train.py` 를 돌리면 Phase 1 Filter Ablation
결과 (Table 1) 가 된다.

Dispatcher 구현: `scripts/self_training/self_train.py` — `_apply_filter()`.

### 4.4.1 Canonical filters (A/B/C/D) — Ablation 전용 설계

Historical 3-단계 설계 (A: flip consistency, B: visible structural support,
C: normalized LOO PnP stability, 선택 D: diagonal incidence) 는
`scripts/data_prep/canonical_filters.py` 에 구현이 그대로 보존된다. Runtime
루프에서는 사용되지 않지만 논문 ablation table 에서 RANSAC 과 직접 비교하기
위한 baseline 역할.

각 필터의 상세 수식 / 임계값 / 실측 P/R 은 `_docs/filter/2026-04-11_selection.md`
§3–§4 참조.

## 4.5 필터 선정 근거 (요약)

전체 P/R 실측은 `_docs/filter/2026-04-11_selection.md` (23 후보 × 3 모델),
설계 철학은 `_docs/filter/2026-04-11_design_rationale.md` 참조. 핵심만 요약:

```
Filter                   Precision   Recall   F1      (ep68 @ reproj 50 px)
─────────────────────────────────────────────────────────────────────────────
F0  No filter              0.06        1.00     0.119
F1  Confidence > 0.5        0.00        0.00     0.000
F7  B ∧ C canonical         1.00        0.13     0.235
F14 B ∧ C loose (2x)        0.09        0.20     0.167
F15 B ∧ C loose (3x)        0.10        0.33     0.174
F16 B ∧ C very loose        0.11        0.60     0.188
F11 RANSAC c≥6 (Ours) ★     0.71        1.00     0.833
F19 RANSAC c≥7              0.67        0.67     0.667
```

관찰:
1. **F11 이 양쪽 모델 (ep68, r1) 에서 동시 top-1** — robust 선정
2. **B ∧ C 는 threshold sweep 으로도 F11 을 따라잡지 못함** — 구조적 한계
3. **No filter / confidence-only 는 너무 약함** — RANSAC 의 decision power 확인

## 4.6 현재 상태 vs Future Work

**현재**: RANSAC subset consensus + LOO cross-validation 로 단일 gate 구성. 23 후보 P/R
비교로 선정 완료 (2026-04-11). canonical A/B/C 는 ablation 용으로만 보존.

**Future Work**:
1. **Learning-based quality estimator** — 2D keypoint 분포 + belief 로 MLP
   가 pseudo-label quality score 를 회귀. 데이터 필요.
2. **Temporal consistency** — 연속 프레임 간 pose 안정성 (영상 촬영 필요)
3. **noapril 에 AprilTag GT 생성** → cross-validation. 현재는 capture0403middle
   단일 데이터셋으로만 선정되어 있음.

> Contribution 프레이밍은 "A/B/C 3 단계 필터" 가 아니라 **"23 후보를 GT P/R
> 로 엄밀히 비교하여 RANSAC subset consensus 가 이 task 에 최적임을 증명"**
> 하는 것. Canonical filter 의 negative result 도 contribution 의 일부.
