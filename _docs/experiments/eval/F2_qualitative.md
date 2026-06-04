# F2. Qualitative / Failure Analysis  (논문 Figure)

> 상태: **미시작** (일부 overlay 존재) | 의존: 위 실험들
> 구분: **다시** (camera-facing 케이스로)

## 목적 (한 줄)
정성 결과와 실패 유형을 그림으로 — 강점(통과 PL 정합)과 한계(diag scale-skew, 뒷면 오차, truncation).

## 판단 지표
대표 케이스 overlay (성공/필터가 거른 bad/필터가 놓친 케이스). 정량 아님, 정성 전달.

## 설정
- 모델: paper_base / paper_r1
- 케이스: 도메인별 good / diag가 거른 catastrophic / diag가 놓친 scale-skew / 뒷면 오차 / truncation 복원
- 기존 자산: `filter_domain_analysis/overlays_s2/`, `diag_pass_overlays/`, `pl_gt_diff/exp4_overlays/`

## 알려진 실패 유형 (채울 것)
- diag scale-skew: 중심 맞고 균일 스케일 틀림 → diag 통과 (indoor)
- 뒷면(4-7) 오차: monocular depth ambiguity, top-down에서 큼
- 검출 붕괴: held-out 모델 real에서 빈약

## 결과 (TBD)
- figure 경로 TBD

## 결론 (TBD)
