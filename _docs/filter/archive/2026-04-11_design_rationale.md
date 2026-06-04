# Filter Design Rationale — 왜 RANSAC 하나로 충분한가

작성일: 2026-04-11
관련: `2026-04-11_selection.md` (P/R 실측 결과)

## 요지

현재 필터는 **RANSAC subset consensus** 가 main gate 이고, **LOO cross-validation**
이 post-check 로 one-sided keypoint collapse 를 방지한다. Pre-filter 는
최소 keypoint 수만 확인하는 가벼운 사전 필터. 이 3단계 구조가 의도된
결과인 이유와 더 복잡한 필터를 쌓지 않는 근거를 정리한다.

## 1. self-training 은 기하학적 이해를 주지 못한다

### self-training 이 하는 것
- real 이미지에서 pseudo-label 생성 → 그걸로 DOPE fine-tune
- **"real 환경에서 keypoint 를 어디에 찍어야 하는가"** 를 더 잘 배움
- domain adaptation, 위치 정밀도 향상 (49.5% → 78.7%)

### self-training 이 못 하는 것
- CNN 구조 자체를 flip-equivariant 로 바꾸는 것
- 모델에 "이건 직육면체다" 라는 기하학적 prior 를 주는 것
- VGG-19 의 padding / stride 비대칭을 고치는 것

즉 self-training 은 **데이터 분포에 적응**하는 것이지 **기하학적 구조를 이해**
시키는 것이 아니다. flip equivariance 같은 구조적 property 는 그대로 남는다.

### 그런데 이게 문제가 되는가?

**안 된다.** 오늘 (2026-04-10 ~ 11) 실험들이 이미 확인함:

```
시도                          결과                          이유
────────────────────────────────────────────────────────────────────────
VP loss                       dead (gradient ≈ 0)           belief MSE 가 이미 만족
Edge loss                     오히려 악화                    작은 오차를 amplify
Flip loss                     최악 (F1 ≈ 0)                 coord 와 충돌
Coord loss                    best                          남은 오차는 구조 붕괴
                                                            가 아니라 위치 밀림
```

→ 모델이 기하학적 구조를 "이해" 할 필요 자체가 없음
→ belief map MSE 가 **암묵적으로** cuboid 구조를 복사
→ 남은 오차는 "구조 붕괴" 가 아니라 "위치 밀림"
→ 위치는 coord loss + self-training 으로 잡으면 됨

자세한 메커니즘: `memory/project_one_sided_collapse_mechanism.md`,
`project_structural_loss_line_closed.md`, `project_vp_loss_dead_on_arrival.md`.

## 2. 역할 분담이 이미 되어 있다

```
역할                    누가 담당              상태
──────────────────────────────────────────────────────────────
기하학적 구조 유지       belief map MSE (암묵)   자동 (gradient ≈ 0)
위치 정밀도 향상         coord loss              v8_A 에서 확정
real domain 적응        self-training           49.5% → 78.7% 입증
outlier 제거            RANSAC filter           F1 = 0.833 (ep68)
flip equivariance       ???                     해결 안 됨, 근데 필요 없음
```

flip equivariance 가 정말 필요한 상황:
- flip consistency 를 필터로 쓰고 싶을 때 (→ RANSAC 이 더 좋음)
- 모든 viewpoint 에서 동일 성능 보장 (→ self-training + DR 로 간접 해결)

즉 현재 구조에서는 flip equivariance 를 따로 해결할 motivation 이 없다.

## 3. 필터 구조 시각화

3 단계 필터 파이프라인.

```
입력: 9 개 2D keypoint predictions
              │
              ▼
┌────────────────────────────────────────────┐
│ [1] Pre-filter:                            │
│   detected ≥ 5                             │  (min_keypoints, 가벼운 사전 필터)
└────────────────────────────────────────────┘
              │  pass
              ▼
┌────────────────────────────────────────────┐
│ [2] Main gate:                             │
│   RANSAC subset consensus                  │
│   - n_iter = 50                            │  ★ decision power 핵심
│   - subset_size = 5 (random)               │
│   - reproj_thresh = 5.0 px                 │
│   - min_consensus ≥ 6                      │
└────────────────────────────────────────────┘
              │  pass
              ▼
┌────────────────────────────────────────────┐
│ [3] Post-check:                            │
│   LOO cross-validation (τ = 0.05)          │
│   각 keypoint i 제외 → PnP → reproj error │
│   median(e_i / D) < τ_LOO                  │  (one-sided collapse 방지)
└────────────────────────────────────────────┘
              │  pass
              ▼
           Accept ✓
           (pseudo-label 채택)
```

LOO 는 RANSAC 통과 후에도 잔존하는 one-sided keypoint collapse (한쪽 면의
점들만 정확하고 반대쪽이 크게 밀린 경우) 를 걸러낸다. RANSAC 은 inlier
수 기반이라 6 개 이상이 대략 맞으면 통과시키지만, LOO 는 각 점의 개별
안정성을 검증한다.

### 다른 필터를 추가할 수 있는가?

| 후보 | 효과 | 결론 |
|------|------|------|
| Confidence gate (peak > 0.5) | RANSAC 전에 약한 점 제거 | 효과 미미 — RANSAC 이 이미 약한 점을 subset 에서 배제 |
| B∧C (canonical) 추가 | RANSAC pass 한 것 중 B∧C 로 한 번 더 | **오히려 악화** — B∧C 가 너무 보수적이라 좋은 것도 버림 |
| Temporal consistency | 프레임 간 pose 안정성 | 영상 데이터 필요, 현재 단일 프레임 기준이라 적용 불가 |
| Learning-based quality | MLP 가 score 예측 | 학습 데이터 필요, scope 밖 |

→ RANSAC 위에 더 쌓아서 좋아지는 것은 없다. 실측으로 확인됨 (F7, F13).

## 4. "필터 1 개" 는 약점이 아니라 장점

### 논문 프레이밍

**약한 프레이밍 (피해야 함)**:
"본 연구는 RANSAC subset consensus 를 필터로 사용한다."
→ "겨우 하나?" 인상

**강한 프레이밍 (추천)**:
"본 연구는 pseudo-label 검증 필터를 선정하기 위해 23 개 후보를 GT 기반
 precision / recall 로 비교했다. Canonical 기하학적 필터 (A: flip consistency,
 B: structural support, C: LOO PnP stability) 와 그 조합들, RANSAC 변형, old
 reprojection-based filter 를 포함한 매트릭스에서 **RANSAC subset consensus
 (n=50, k=5, τ=5 px, c ≥ 6) 가 단일 최고**였다 (F1 = 0.833). 특히 canonical
 B ∧ C 는 threshold sweep (2 x, 3 x, very loose) 로도 RANSAC F1 에 도달하지
 못해, precision 은 높으나 recall 이 극도로 낮은 구조적 한계를 드러냈다.
 본 연구는 이 엄밀한 비교의 결과로 간결한 RANSAC 기반 필터를 채택한다."

### Contribution 구성

```
Contribution          내용
────────────────────────────────────────────────────────────────
① 합성 데이터 파이프라인   Isaac Sim + Blender multi-source
② Filter selection        23 후보 비교 → RANSAC 선정 (★ 이게 이 섹션)
                          - 비교 methodology 자체가 contribution
                          - canonical negative result 포함
③ Self-training 적용      49.5% → 78.7% 입증
```

즉 contribution 은 **"우리가 RANSAC 을 발명했다"** 가 아니라 **"우리가 여러
후보를 엄밀히 비교해서 RANSAC 이 이 문제에 적합함을 증명했다"** 이다.
Negative result (canonical B ∧ C 의 실패) 도 contribution 의 일부.

### Paper Table draft

```
Filter                    Precision   Recall   F1      Pseudo-label #
──────────────────────────────────────────────────────────────────────
No filter                  0.06        1.00     0.119   237
Confidence > 0.5           0.00        0.00     0.000   0
B ∧ C (canonical)          1.00        0.13     0.235   2
B ∧ C (loose 3 x)          0.13        0.27     0.174   31
A ∧ B ∧ C (full)           0.00        0.00     0.000   0
RANSAC c ≥ 6 (Ours) ★      0.71        1.00     0.833   14
                                               (ep68, threshold 50 px)
```

## 5. 더 기하학적인 모델을 원한다면 (전부 scope 밖)

| 방법 | 난이도 | 효과 | 평가 |
|------|--------|------|------|
| Loss 로 강제 (VP, edge, flip) | 쉬움 | ✗ 실패 확인됨 | 이미 시도, dead |
| Self-training | 쉬움 | △ domain 만 | 구조는 못 바꿈 |
| Equivariant Network (E2CNN 등) | 매우 높음 | ★★★ 근본 해결 | DOPE 구조 전면 교체 |
| Transformer backbone | 높음 | ★★☆ context 학습 | DOPE 대체 수준 |
| Geometric pre-training | 중간 | ★★☆ 3D prior | 별도 연구 주제 |

전부 석사 논문 scope 를 넘음. 현재 성능이 충분하므로 이 방향은 **future work**
섹션에서 언급만 한다.

## 한 줄 요약

> 모델이 기하학을 몰라도 coord loss 로 위치를 잡고 RANSAC 으로 outlier 를
> 걸러내면 self-training 이 나머지를 해결한다. 이 역할 분담이 성립하므로
> 필터는 RANSAC 하나로 충분하고, "하나뿐인 필터" 는 23 후보 비교 끝에
> 남은 엄밀한 결정이다.

## 관련 문서

- `2026-04-11_selection.md` — 23 후보 P/R 실측 표
- `README.md` — filter 폴더 인덱스
- `_docs/method/step2_geometric_filter.md` §4.0 — 설계 변경 공지
- `_docs/history/2026-04-11.md` — 당일 작업 기록
- `memory/project_one_sided_collapse_mechanism.md` — belief MSE 우세 이유
- `memory/project_structural_loss_line_closed.md` — loss 시도 실패 종합
- `memory/project_filter_selection_ransac.md` — 필터 선정 메모
