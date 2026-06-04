# B2. 필터별 Self-training Downstream  (논문 Table)

> 상태: **미시작** | 의존: paper_base, C1(rounds 인프라)
> 구분: **다시** (v8 폐기 → camera-facing 필터로 재). B1(Stage1 P/R)은 `filter/pr_screening.md` 참조.

## 목적 (한 줄)
Stage1 P/R로 선정한 필터(diag 등)가 **실제 self-training downstream 향상에서도 최선인가** — P/R proxy ↔ 실제 향상 상관 검증 (4월 빗나감 교정).

## 판단 지표
필터별 **R1 도메인 향상폭**. + Stage1 9kp 오차 랭킹 ↔ downstream 향상 랭킹 상관(산점도).

## 설정
- 후보 필터: diag / ratio / diag∧ratio / fullkp / ransac_loo (대조군 none)
- anchor = paper_base, 도메인별
- Stage1 결과(pr_screening.md): outside diag 9.9px / night diag∧ratio 7.9px

## 방법
1. 각 필터로 PL 추출 → R1 학습
2. 도메인별 향상폭 측정
3. **Stage1 9kp오차 랭킹 vs downstream 향상 랭킹** 상관 그림 (P/R proxy 신뢰도 검증)

## 결과 (TBD)
```
필터          Stage1 9kp_med   R1 향상(outside)   R1 향상(night)
─────────────────────────────────────────────────────────────
diag          9.9              TBD                TBD
diag∧ratio    (night 7.9)      TBD                TBD
ransac_loo    낮음(물량사망)    TBD                TBD
```

## 결론 (TBD)
- P/R proxy가 downstream을 예측하는가? (4월엔 빗나감 — 이번엔 명시 검증)
