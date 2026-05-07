# Filter Research

Self-training pseudo-label filter 관련 작업 전용 폴더. 설계 변경, P/R 분석,
threshold sweep, ablation 결과 등 필터에 관한 모든 실험/결정을 시간순으로
누적한다.

전체 파이프라인 (Step 1/2/3) 요약은 `_docs/method/` 에 있고, 이 폴더는
**필터 하나에 대한 깊은 기록**을 담는다.

## 현행 설계

**Primary filter**: RANSAC subset consensus + LOO cross-validation
- [1] Pre-filter: detected ≥ 5
- [2] RANSAC: n_iter = 50, subset = 5, reproj_thresh = 5.0 px, min_consensus = 6
- [3] LOO cross-validation: τ_LOO = 0.05 (one-sided collapse 방지)
- 선정일: 2026-04-11 (RANSAC), 2026-04-13 (LOO 추가)
- 구현: `scripts/self_training/geometric_filter.py` + `scripts/data_prep/canonical_filters.py` (filter_C)
- 설정: `config/stage3_selftrain.yaml` (`geometric_filter` 섹션)

**Ablation 비교용 (런타임 미사용)**:
- A: Flip consistency
- B: Visible structural support (span + endpoint + non-collinearity)
- C: Normalized LOO PnP stability
- D: Conditional diagonal incidence
- 구현: `scripts/data_prep/canonical_filters.py` — 논문 비교용으로 보존

## 문서 목록 (시간 역순)

| 날짜 | 문서 | 요약 |
|------|------|------|
| 2026-04-11 | [2026-04-11_design_rationale.md](2026-04-11_design_rationale.md) | 왜 RANSAC 하나로 충분한가 — self-training 이 기하학적 이해를 못 주는 이유, 역할 분담, 3 단계 구조, 논문 프레이밍 |
| 2026-04-11 | [2026-04-11_selection.md](2026-04-11_selection.md) | 23 필터 후보 P/R 분석 → RANSAC c≥6 선정. canonical B∧C reject 근거. |

## 관련 파일

### 코드
- `scripts/self_training/geometric_filter.py` — 런타임 필터 (RANSAC)
- `scripts/self_training/self_train.py` — `solve_and_validate()` 호출 사이트
- `scripts/data_prep/canonical_filters.py` — A/B/C/D ablation 필터
- `scripts/data_prep/eval/filter_pr_eval.py` — P/R 분석 스크립트

### 결과
- `data/pallet/eval_results/filter_pr/` — P/R summary JSON/CSV + per-frame 데이터

### 이전 문서 (요약만)
- `_docs/method/step2_geometric_filter.md` — Step 2 전체 요약, §4.0 에 설계 변경 공지
- `_docs/history/2026-04-11.md` — Filter 재선정 당일 작업 기록

## 다음 작업 후보

- [ ] Filter Ablation 실험 (Phase 1) — F0 / F1 / F2(old, deprecated) / F7(B∧C) / F11(RANSAC) 5개로 self-training 실제 성능 비교
- [ ] Consensus parameter 재검증 (n_iter=100, subset=6 등)
- [ ] noapril 에 AprilTag GT 생성 후 cross-validation
- [ ] Threshold (reproj_px=5.0 → 3.0 / 7.0) 민감도
- [ ] Learning-based quality estimator (optional, long-term)
