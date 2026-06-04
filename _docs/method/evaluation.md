# Evaluation — 메트릭 + PnP 용도 분리 (camera-facing)

> 폐기 v8 평가(formulation/implementation)는 `archive/`.

## PnP 용도 분리 (핵심)

비율 unknown 인 처음 본 파렛트 때문에 PnP 를 무조건 쓸 수 없다. 용도별로 분리:

| 용도 | 치수 필요? | 방법 |
|------|-----------|------|
| A. self-training PL 필터 | ❌ | 2D 기하 (PnP 불필요) → 처음 본 파렛트 가능 |
| B. 정확도 평가 (ADD) | ✅ | 치수 known GT(내 파렛트)에서 **SQPnP** |
| C. 거리(z) 추정 | ✅ | 과제용 challenge(내 파렛트)에서 **SQPnP** |

## PnP solver = SQPnP

- `cv2.SOLVEPNP_SQPNP` + RefineLM. EPnP+RANSAC 대비 reproj median 5.27→3.12px,
  ADD 96.6→90.7mm 개선 (2026-06-02 YOLO 경로 검증).
- 팔레트는 얇은 near-planar 직육면체라 globally optimal SQPnP 가 유리.
- `scripts/self_training/pnp_solver.py` (현재 EPnP+RANSAC) → 평가/거리용 SQPnP 교체 필요.

## 메트릭

- **keypoint**: PCK, reproj (order-free 비교 — convention 차이 흡수)
- **6D (치수 known)**: ADD, 5cm5°
- **필터**: Precision / Recall / F1 (GT 대비 PL good 판정)
- **self-training**: 도메인별 per-frame 검출 정확도 (R0→R1→R2 매트릭스)

## GT 평가셋

- `data/_eval_sets/outside_combined` (129), `night_combined` (90), 합성 val.
- ⚠️ convention 정합(camera-facing vs object-frame) 확인 필요 → order-free 비교로 흡수.

## 주의 (기록된 버그)

- `evaluate_on_val.py` reproj 130px 는 convention 삼중 불일치 (memory
  `evaluate-on-val-convention-bug`). same-index 아닌 **order-free PnP** 로 풀어야
  진짜 reproj(한 자리수)가 나옴. 검출률/PnP success 는 신뢰 가능.
