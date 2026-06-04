# Step 3 — Self-training Finetuning (camera-facing)

> 폐기 v8 버전은 `archive/step3_finetuning.md`.

## 루프

```
paper_base → unlabeled 추론 → 2D 기하 필터로 PL 선별 (Step 2)
  → PL 로 finetune → paper_r1
  → paper_r1 로 PL 재추출 → finetune → paper_r2 → ...
```

- finetune = 누적 epoch (memory `dope-finetune-cumulative-epoch`).
- 도메인: indoor / outdoor / night (도메인 갭 robustness 검증).

## 핵심 교훈 (이전 발표에서 확인 — 필터 재실험의 동기)

- **indoor**: PL 수 적었지만 R1 에서 성능 크게 ↑
- **outdoor/night**: PL 수 많았지만 ~1% ↑ 에 그치고, **R2 에서 오히려 ↓**
- → **PL 수보다 품질(신뢰도)이 핵심.** 다량의 noisy PL 은 self-training 을 악화시킴.
  좋은 2D 기하 필터로 신뢰도 높은 PL 을 만드는 것이 성공의 열쇠.

## 평가 (2단계)

- **Stage 1 (P/R 스크리닝)**: 학습 없이 기존 모델 추론 → 필터별 P/R 로 후보 선별.
- **Stage 2 (downstream)**: 상위 필터로 실제 R1/R2 학습 → 도메인별 성능 매트릭스.
  - 4월 교훈: P/R 랭킹 ↔ downstream 향상 상관을 명시 검증 (P/R proxy 가 빗나갈 수 있음).

## 체크리스트

- [ ] (Stage 1) 필터 P/R 스크리닝
- [ ] (Stage 2) 상위 필터 R0→R1 도메인별 학습 + 평가
- [ ] R2 (R1 승자만) — 발표 매트릭스 재현
