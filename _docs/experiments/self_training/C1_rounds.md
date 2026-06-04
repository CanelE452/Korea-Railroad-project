# C1. Self-training R0→R1→R2 × 도메인  (★논문 핵심 Figure F1 + Table)

> 상태: **미시작** | 의존: paper_base 학습, 필터 선정(diag/diag∧ratio)
> 구분: **다시** (v8 발표 셋업을 camera-facing + paper_base로 재현)

## 목적 (한 줄)
2D 기하 필터로 선별한 신뢰 PL로 self-training 반복 시, **도메인별(indoor/outside/night) 성능이 R0→R1→R2로 향상**되는가.

## 판단 지표
도메인별 **per-frame 검출 정확도(NN<20px) + reproj(9kp)**, R0/R1/R2 곡선.
(발표 교훈: PL 수보다 품질. indoor 소량 PL로 R1↑, outdoor/night 다량인데 R2↓ → 좋은 필터로 재현 검증)

## 설정
- anchor R0 = `paper_base`
- 필터: outside=`diag`, night=`diag∧ratio` (indoor=PL 신뢰 낮음 → 1라운드 후 재필터)
- unlabeled pool: outside 9894 / night 9134 / indoor(noapril) 188 (TBD: camera-facing 재확인)
- GT 평가셋: outside_combined(129)·night_combined(90)·capture0403middle(440) [exclude.txt 반영]
- 학습: train.py finetune, 누적 epoch (memory `dope-finetune-cumulative-epoch`)

## 방법
1. R0(paper_base) → unlabeled 추론 → 필터 → PL 추출
2. PL로 R1 finetune → 도메인별 평가
3. R1 → PL 재추출 → R2 → 평가
4. R0/R1/R2 매트릭스 + 곡선

## 결과 (TBD)
```
도메인     R0      R1      R2      필터
──────────────────────────────────────────
indoor     TBD     TBD     TBD     (재필터)
outside    TBD     TBD     TBD     diag
night      TBD     TBD     TBD     diag∧ratio
```

## 결론 (TBD)

## 산출물 (예정)
- round figure(F1), PL pool 증가표, 도메인 cross 평가
