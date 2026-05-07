# scripts/self_training/

FixMatch 기반 준지도 Self-Training 파이프라인. Step 2-3에 해당.
설정: `config/stage3_selftrain.yaml`

## 스크립트

| 파일 | 역할 | 타입 |
|------|------|------|
| `self_train.py` | 메인 Self-Training 루프 (pseudo-label 생성 → 필터링 → 학습 반복) | 실행 |
| `geometric_filter.py` | Pseudo-label 검증 (RANSAC subset consensus + size sanity) | 모듈 |
| `pnp_solver.py` | EPnP + RANSAC로 2D keypoint → 6D pose 복원 | 모듈 |
| `augmentations.py` | FixMatch weak/strong augmentation (photometric only, 좌표 불변) | 모듈 |
| `metrics.py` | 6D 포즈 평가 메트릭 (ADD, 5cm-5°, reproj error) | 모듈 |

## 파이프라인 흐름

```
Real image → Weak aug → DOPE inference → RANSAC subset consensus (solve + gate)
                                                            ↓
                                                    Pseudo-label (통과)
                                                            ↓
                            Synthetic (GT) + Real (pseudo) → Mixed training
                                                            ↓
                                                    다음 라운드 반복
```

## 사용법

```bash
python scripts/self_training/self_train.py --config config/stage3_selftrain.yaml
```

## Geometric Filter (2026-04-11 개정)

**RANSAC subset consensus** — GT 기반 precision/recall 분석 (`_docs/filter/2026-04-11_selection.md`)
을 통해 23개 후보 중 최종 선정됨. canonical B∧C (이전 설계) 보다 F1 점수가 두 모델
모두에서 압도적으로 높음 (ep68: 0.833 vs 0.235).

1. **RANSAC subset consensus (primary gate)**
   - 검출된 keypoint에서 랜덤 5개 subset으로 EPnP → 총 `ransac_n_iter=50` 회 반복
   - 각 후보 pose에 대해 전체 detected 점의 reprojection inlier 수(`<5px`)로 voting
   - 최대 consensus ≥ `ransac_min_consensus=6` 이면 accept
   - 이 과정이 solver + filter 역할을 한 번에 수행 (`solve_and_validate()`)
2. **Physical size sanity** — 복원된 pallet width가 0.5~2.5m 범위인지 (`tau_size_*`)

임계값은 `config/stage3_selftrain.yaml`의 `geometric_filter` 섹션에서 관리.

Canonical filters (A: flip, B: structural support, C: LOO PnP, D: diagonal incidence)
는 논문 ablation 비교용으로 `scripts/data_prep/canonical_filters.py` 에 그대로 유지됨.
