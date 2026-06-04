# T3. Filter Selection — GT-based P/R

상태: **★ 완료** (2026-04-11)

## 요약

`capture0403middle` 440 장 GT 에서 23 필터 후보 × 2 모델 (v8_A ep68,
selftrain_r1) frame-level Precision / Recall / F1 측정. "good" = 2D mean
projected_cuboid reproj error < 50 px.

**선정**: F11 RANSAC subset consensus (n_iter=50, subset=5, reproj 5 px,
c≥6) — **양 모델 동시 top-1**.

## Table 3 (요약)

```
Filter ID   Type                     Precision   Recall   F1 (ep68)   F1 (r1)
────────────────────────────────────────────────────────────────────────────────
F0          no filter                  0.063       1.000    0.119       0.147
F1          confidence > 0.5           0.000       0.000    0.000       0.000
F2          old reproj+cuboid+size     0.000       0.000    0.000       0.000
F4          B only                     —           —        0.667       0.125
F7          B ∧ C canonical            1.000       0.133    0.235       0.069
F8          A ∧ B ∧ C                  0.000       0.000    0.000       0.000
F14         B ∧ C loose (2x)           0.087       0.200    0.167       0.356
F15         B ∧ C loose (3x)           0.100       0.333    0.174       0.480
F16         B ∧ C very loose           0.110       0.600    0.188       0.533
F11 ★       RANSAC subset c≥6          0.714       1.000    0.833       0.722
F19         RANSAC c≥7                 1.000       0.500    0.667       0.696
```

전체 23 후보 표는 `_docs/filter/2026-04-11_selection.md`. RANSAC consensus
threshold sweep (c ∈ {4, 5, 6, 7, 8}) 은 [`consensus_sweep.md`](./consensus_sweep.md).

## 관찰 3 가지

1. **F11 이 두 모델 동시 top-1** — 모델 교체에도 robust
2. **B ∧ C 는 threshold sweep 으로도 회복 불가** — F14/F15/F16 어떤
   loose 변형도 F11 F1 을 따라잡지 못함 (구조적 한계)
3. **confidence 단일 필터는 0 점** — v8_A 가 OOD 라 belief peak 가
   낮아 `> 0.5` 자체가 거의 통과 안 됨

## 관련

- 상세: `_docs/filter/2026-04-11_selection.md` (23 후보 전체 + 해석)
- 설계 rationale: `_docs/filter/2026-04-11_design_rationale.md`
- 필터 인덱스: `_docs/filter/README.md`
- 원 결과: `data/pallet/eval_results/filter_pr/summary_ep68_t50.json`,
  `summary_r1_t50.json`
- 스크립트: `scripts/data_prep/eval/filter_pr_eval.py`
