# C2. PL 품질 vs 수량 trade-off  (논문 Table)

> 상태: **미시작** | 의존: C1(rounds)
> 구분: **다시/새로** (발표 교훈을 camera-facing 필터로 정량화)

## 목적 (한 줄)
필터 strict 정도(통과량↓·순도↑)가 self-training 향상에 미치는 영향 — **"PL 수보다 품질"** 가설 검증.

## 판단 지표
필터별(느슨~빡셈: none / diag / diag∧ratio / ransac_loo) **통과 PL 수 vs R1 도메인 향상폭**.
좋은 PL 절대수(통과량×순도)가 향상과 상관되는지.

## 설정
- anchor = paper_base, 도메인별 동일 pool
- 필터 strict 단계: none → diag → diag∧ratio → ransac_loo(고순도 저물량)
- 평가: 각 필터로 R1 학습 후 도메인 검출/reproj

## 방법
1. 필터 strict 단계별 PL 추출(수·순도 기록)
2. 각각 R1 학습 → 향상폭
3. (통과량, 순도, 좋은PL 절대수) vs 향상 산점도

## 결과 (TBD)
```
필터          통과PL수   순도(9kp good%)   R1 향상(pp)
─────────────────────────────────────────────────────
none          많음       낮음              TBD
diag          중간       중간              TBD
diag∧ratio    적음       높음              TBD
ransac_loo    매우적음   매우높음(물량사망) TBD
```

## 결론 (TBD)
