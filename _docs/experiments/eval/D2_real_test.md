# D2. Real Test — ADD / 5cm5° / Reproj  (논문 Table)

> 상태: **미시작** | 의존: paper_base 또는 paper_r1, real GT(치수 known)
> 구분: **다시** (camera-facing + SQPnP로 재)

## 목적 (한 줄)
실제 파렛트에서 6D pose 정확도(metric). 치수 known 데이터에서만 (PnP 용도분리 B).

## 판단 지표
**ADD · 5cm5° · reproj(9kp)** — SQPnP, dims known.
(주의: monocular라 5cm5°는 약할 수 있음 — reproj median이 keypoint 품질의 깨끗한 신호)

## 설정
- 모델: paper_base / paper_r1 (+비교: challenge 과제 모델)
- GT: real manual GT (outside_combined·night_combined·forklift, dims per-frame), `_exclude.txt` 반영
- PnP: **SQPnP** (`cv2.SOLVEPNP_SQPNP`+RefineLM), order-free 비교
- 메모리: evaluate-on-val convention 버그 주의 (order-free PnP로)

## 방법
1. 추론 → 9kp → SQPnP → 6D
2. ADD/5cm5°/reproj 집계 (도메인별 + 전체)
3. R0 vs R1 개선

## 결과 (TBD)
```
model        ADD_med   5cm5°   reproj_med
──────────────────────────────────────────
paper_base   TBD       TBD     TBD
paper_r1     TBD       TBD     TBD
```

## 결론 (TBD)
