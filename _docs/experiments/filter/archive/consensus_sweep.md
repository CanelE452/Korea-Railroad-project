# T7. RANSAC Consensus Threshold Sensitivity

상태: **★ 완료** (2026-04-11)

## 목적

RANSAC subset consensus 의 `min_consensus` 파라미터 민감도 분석. 채택된
값 (c ≥ 6) 이 sweet spot 임을 sweep 으로 증명.

## 데이터 출처

`data/pallet/eval_results/filter_pr/summary_ep68_t50_sweep.json` 의 F11,
F17 ~ F20 (F11 = c≥6, F17 = c≥4, F18 = c≥5, F19 = c≥7, F20 = c≥8). 데이
터셋 = capture0403middle 440 장, "good" threshold = 50 px mean reproj.
평가 모델 = v8_A_control ep68.

## Table 7

```
consensus    n_pass   TP    FP    Precision   Recall   F1
──────────────────────────────────────────────────────────────
c ≥ 4        35       10    25    0.286       1.000    0.444
c ≥ 5        29       10    19    0.345       1.000    0.513
c ≥ 6 ★      14       10     4    0.714       1.000    0.833
c ≥ 7         5        5     0    1.000       0.500    0.667
c ≥ 8         2        2     0    1.000       0.200    0.333
```

## 해석

1. **c ≥ 6 이 sweet spot** — P 0.714 / R 1.000 / F1 0.833 로 단일 최고
2. **c ≤ 5**: recall 은 만점이지만 precision 이 낮아 FP 가 TP 보다 많음
   (noise 유입)
3. **c ≥ 7**: precision 이 만점이지만 recall 이 반토막 → 유효 PL 수 급감
   (5 장 / 2 장). 실제 학습용으로는 부족
4. R = 1.000 은 "good frame 은 전부 통과" — c ≥ 6 기준 440 장 중 good 이
   10 장, 전부 pass. FP 4 장 = good 아닌 걸 통과시킨 frame

## 2 차 모델 (selftrain_r1) 재현

`summary_r1_t50_sweep.json`:

```
consensus    n_pass   TP    FP    Precision   Recall   F1
──────────────────────────────────────────────────────────────
c ≥ 4        34       13    21    0.382       1.000    0.553
c ≥ 5        33       13    20    0.394       1.000    0.565
c ≥ 6 ★      23       13    10    0.565       1.000    0.722
c ≥ 7        10        8     2    0.800       0.615    0.696
c ≥ 8         4        3     1    0.750       0.231    0.353
```

두 모델 모두 c ≥ 6 단일 1 위 → robust 선정 근거.

## 논문 draft 한 줄

> "Consensus threshold c ≥ 6 achieves Precision = 0.714, Recall = 1.000,
> F1 = 0.833 on ep68 and remains the top-1 threshold on a second (selftrain-r1)
> model, confirming c = 6 as a robust sweet spot across model updates."

## 관련

- 원 스크립트: `scripts/data_prep/eval/filter_pr_eval.py`
- Raw: `data/pallet/eval_results/filter_pr/summary_*_t50_sweep.json`
- Filter 선정 전체 맥락: `_docs/filter/2026-04-11_selection.md`
- 필터 구현: `scripts/self_training/geometric_filter.py`
