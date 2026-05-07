# T9. Multi-source Synthetic 비교 (Legacy)

상태: **부분 (2026-03-30 완료)** — v10 완료 후 재측정 필요

## 목적

Isaac Sim 단독 / Blender 단독 / 혼합 학습이 real 성능에 미치는 영향.
v1 시절 확정된 결과는 "1:1 mixed 가 가장 robust". v8 이후는 이 결과를
받아들여 `mixed_v*` 만 학습했고, 이 축의 ablation 은 일단락.

## Table 9 (legacy, 2026-03-30)

```
Experiment     Train Data                      Size     PCK@3px   PCK@10px   PnP%    Reproj    Vol Ratio   Vol<20%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
blender_v1     Blender only                     3,600    0.635     0.891      100.0   44.1      0.881       46.7%
combined_v1    Isaac 6K + Blender 3.6K          9,600    0.498     0.607      79.0    134.5     ?           ?
mixed_v1       Isaac 4K + Blender 4K (1:1)      8,000    ?         ?          ?       ?         ?           ?
mixed_v8       Isaac + Blender (1:1, v8)        9,000    TBD       —          —       —         —           —
mixed_v10      Isaac + Blender (1:1, v10)       10,000   TBD       —          —       —         —           —
```

## 편향 주의 (중요)

> 이 표의 절대 순위로 최종 결론을 내면 안 됨.
> - combined_v1 은 `max_frames=200` 으로 Isaac Sim 프레임만 평가된 편향 있음
> - Blender v1 은 real 근거리 정면에서 약함, combined_v1 은 사람 가림 /
>   원거리에서 더 robust — metric 으로 잡히지 않는 질적 차이 존재
> - **공정한 비교**: 동일 source-balanced val (mixed_v1_val 800 장) 에서
>   `max_frames=800` 으로 재평가 필요

## v10 후 재측정 계획

mixed_v10 학습 완료되면:

1. mixed_v10 을 mixed_v1_val 에서 평가
2. 위 표의 mixed_v8 / mixed_v10 row 채움
3. capture0403middle 에서도 PnP rate / 2D reproj 추가 측정
4. real Seen / Unseen 까지 가서 3 stage 최종 순위

## 관련

- v1 시절 실험 기록: `_docs/history/2026-03-30.md` (존재 시)
- 모델 카탈로그: [`../model_catalog.md`](../model_catalog.md)
- Val 평가 스크립트: `scripts/data_prep/eval/evaluate_on_val.py`
- Real 평가 스크립트: `scripts/data_prep/eval/evaluate_real.py`
