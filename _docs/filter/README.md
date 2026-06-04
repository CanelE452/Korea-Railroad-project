# Filter Research

Self-training pseudo-label 필터 연구 폴더. camera-facing 0123 기반 2D 기하 필터의
설계·P/R 분석·ablation 을 시간순 누적.

> 폐기: v8(object-frame) 시절의 RANSAC c≥6 선정(`archive/2026-04-11_selection.md`,
> `archive/2026-04-11_design_rationale.md`)은 object-frame 기준이라 무효.
> object-frame 에선 코너 순서가 시점마다 달라 직사각형 기하 필터가 안 먹혔음.

## 현행 방향 (camera-facing 2D 기하 필터)

camera-facing 0123 이라 직사각형 cuboid 의 2D 기하 관계가 image 상에서 일관 →
**PnP 없이 2D 만으로** pseudo-label 신뢰도 판정 (처음 본 파렛트, 비율 unknown 대응).

핵심 후보:
- **공간 대각선 교점 ≈ centroid(8)** — projective invariant ★
- {0,1,4,5} 위 / {2,3,6,7} 아래 순서
- 변 비율 (0-1≈4-5, 0-4≈1-5) — perspective 보정
- 9 keypoint 전부 검출 strict, conf×geometry, per-domain adaptive (서베이 권장)

정확한 인덱스/불변량/임계값 설계는 `3d-expert` 위임 예정.

## 문서 목록

| 날짜 | 문서 | 요약 |
|------|------|------|
| 2026-06-02 | [2026-06-02_survey_pseudolabel_filtering.md](2026-06-02_survey_pseudolabel_filtering.md) | pseudo-label filtering 서베이 — conf×geometry AND, 분포기반 adaptive threshold, per-domain best |
| (예정) | `../experiments/filter/pr_screening.md` | 2D 기하 필터 P/R 스크리닝 (학습 불필요) |
| archive | `archive/` | 폐기 v8 필터 (RANSAC selection/rationale) |

## 평가 방법 (학습 불필요)

```
camera-facing 모델(dope_cropaug_ft_s2 등) → GT 평가셋 추론 → 예측 9 keypoint
  → 2D 기하 필터 → GT 대비 good 판정(order-free) → 필터별 P/R
```

## 동기 (발표 교훈)

indoor 는 소량 PL 로 R1 크게↑, outdoor/night 는 다량 PL 인데 ~1%↑·R2 에서↓.
→ **PL 수보다 품질**. 좋은 기하 필터로 신뢰도 높은 PL 을 만드는 것이 핵심.

## 관련

- method: `../method/step2_geometric_filter.md`
- convention: `../preprocessing/keypoint_definition.md`
- 코드(재설계 예정): `scripts/self_training/geometric_filter.py`
