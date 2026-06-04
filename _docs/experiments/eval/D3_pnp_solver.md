# D3. PnP Solver — SQPnP vs EPnP+RANSAC  (논문 Table, 부가)

> 상태: **부분 (challenge에서 검증됨, 논문용 재기록)** | 의존: real GT
> 구분: **새로** (평가/거리용 solver 확정)

## 목적 (한 줄)
near-planar 팔레트에서 SQPnP가 EPnP+RANSAC보다 정확한지 — 평가/거리추정 solver 선택 근거.

## 판단 지표
**reproj median · ADD median · PnP 성공률** (동일 keypoint, solver만 교체).

## 설정
- 동일 예측 9kp, solver만 EPnP+RANSAC vs SQPnP(+RefineLM, median reproj>12px reject)
- GT: real (dims known)
- 기존 검증(2026-06-02 YOLO 경로): reproj 5.27→3.12px, ADD 96.6→90.7mm

## 방법
1. 같은 keypoint에 두 solver 적용
2. reproj/ADD/성공률 비교
3. `scripts/self_training/pnp_solver.py`(현 EPnP) → SQPnP 교체 반영

## 결과 (TBD)
```
solver           reproj_med   ADD_med   성공률
─────────────────────────────────────────────
EPnP+RANSAC      TBD          TBD       TBD
SQPnP            TBD          TBD       TBD
```

## 결론 (TBD)
