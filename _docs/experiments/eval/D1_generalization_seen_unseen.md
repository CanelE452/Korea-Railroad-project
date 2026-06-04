# D1. 일반화 — Seen vs Unseen 파렛트  (논문 Table)

> 상태: **미시작** | 의존: paper_base, real test 데이터
> 구분: **새로/다시** (v1/v2 제외 학습이라 unseen 정의가 바뀜 — 본 연구 핵심 주장)

## 목적 (한 줄)
논문용 모델은 **인터넷 합성만 학습(내 파렛트 v1/v2 제외)** → 내 실제 파렛트가 곧 **unseen**. 처음 본 파렛트 일반화를 정량화.

## 판단 지표
- **seen**(학습에 쓴 합성 파렛트 유형) vs **unseen**(처음 본 real 파렛트)의 keypoint reproj(9kp)·검출률
- PnP-free 2D 필터가 unseen에서도 작동하는지 (비율 unknown 적용성)

## 설정
- 모델: `paper_base` / `paper_r1`(self-train 후)
- seen 셋: 학습 분포 합성 파렛트
- unseen 셋: real(capturepallet/night/forklift) — 학습에 미사용 = unseen
- convention: camera-facing 0123

## 방법
1. seen/unseen 각각 추론 → 9kp reproj·검출
2. self-training 전후(R0 vs R1) unseen 개선폭
3. 일반화 갭(seen−unseen) 보고

## 결과 (TBD)
```
set       model        9kp_med   det%
──────────────────────────────────────
seen      paper_base   TBD       TBD
unseen    paper_base   TBD       TBD
unseen    paper_r1     TBD       TBD
```

## 결론 (TBD)
