# Sigma Sensitivity (optional)

상태: **optional** — 본문 주장 필수 아님, 일정 여유 시 Appendix 용으로 측정

## 목적

belief map Gaussian sigma 가 학습 gradient / keypoint 정밀도에 미치는
영향. DOPE 공식 기본값 (`sigma = 4.0`) 와 self-training pseudo-label 기본
값 (`sigma = 2.0`) 차이에 대한 감수성 확인.

## Table

```
sigma   Belief Peak (px)   Val PCK@3 ↑   Val PnP% ↑   비고
────────────────────────────────────────────────────────────────────────────
0.5     ~1                 ?             ?            gradient vanishing 예상
1.0     ~5                 ?             ?
2.0     ~13 × 13           ?             ?            self-train PL 기본값
4.0     ~25 × 25           ?             ?            DOPE pretrain 기본값
6.0     ~37 × 37           ?             ?            너무 flat 할 수 있음
```

## 측정 방법

같은 synthetic train set, 같은 optimizer, sigma 만 바꿔 60 ep scratch 학습.
각 학습 ~10 시간 → 전체 50 시간. 비싸므로 **Phase 5 에서도 시간 여유
있을 때만**.

축소판: mixed_v8 anchor 유지 + sigma 만 바꿔 5 ep ft 만 수행해도 belief
peak sharpness 차이는 관찰 가능 (단, 수렴 비교는 부정확).

## 주의

- sigma < 1: gradient vanishing → train loss 안 떨어짐
- sigma > 5: belief map 이 너무 flat → peak argmax 정밀도 하락
- self-training pseudo-label 은 sigma=2.0 으로 고정된 것을 유지 권장
  (별도 실험 없이)

## 관련

- 원 가이드: `_docs/method/step1_synthetic_data.md` §3.6
- 메모리: `feedback_brightness_skip.md`
- 학습 스크립트: `scripts/train_dope.sh`
