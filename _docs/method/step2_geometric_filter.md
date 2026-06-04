# Step 2 — 2D 기하 필터 + Pseudo-label (camera-facing)

> camera-facing 0123 기반 2D projective 기하 필터. 폐기 v8(RANSAC c≥6 object-frame)
> 버전은 `archive/step2_geometric_filter.md`. 정확한 설계는 3d-expert 위임 예정.

## 핵심 아이디어

camera-facing 0123 이라 직사각형 cuboid 의 2D 기하 관계가 image 상에서 일관되게
성립한다. 이를 이용해 **PnP 없이 2D 만으로** pseudo-label 의 신뢰도를 판정 →
처음 본 파렛트(비율 unknown)도 필터링 가능. (PnP 용도 분리: 필터엔 PnP 불필요.)

이전 object-frame 에선 코너 순서가 시점마다 달라 이런 기하 제약이 안 먹혔다.
camera-facing 으로 비로소 가능해진 것이 본 연구의 필터 contribution.

## 후보 기하 제약 (사용자 제안 — 3d-expert 가 정확한 인덱스/임계값 확정)

1. **위/아래 순서**: {0,1,4,5} 가 {2,3,6,7} 보다 image y 위쪽
2. **변 비율 일관성**: 앞면 위변(0-1) ≈ 뒷면 위변(4-5), 좌 depth(0-4) ≈ 우 depth(1-5)
   — perspective foreshortening 영향 → 느슨하게 또는 vanishing point 보정
3. **공간 대각선 교점 ≈ centroid(8)**: 0-6, 2-4 등 cuboid 공간 대각선의 교점이
   centroid keypoint 와 가까운지. **직선 교점은 projective invariant** → 비율/거리/
   스케일/시점 무관 ★ 가장 강력
4. (옵션) confidence × geometry 결합, per-domain adaptive threshold — 서베이 권장
   (`../filter/2026-06-02_survey_pseudolabel_filtering.md`)

## 평가 방법 (학습 불필요)

기존 camera-facing 모델 추론만으로 필터 P/R 연구 가능 (Stage 1):
```
camera-facing 모델 → GT 평가셋 추론 → 예측 9 keypoint
  → 2D 기하 필터 적용 → GT 대비 good 판정(order-free 비교) → 필터별 P/R
```
- 상세: `../experiments/filter/pr_screening.md`

## 설계 원칙

- PL 수보다 **품질(precision)** 우선 (발표 교훈: 다량 noisy PL → R2 악화).
- "9 keypoint 전부 검출 시에만" 같은 strict pre-filter 도 후보 (신뢰도↑).

## 체크리스트

- [ ] 2D projective 기하 필터 정확한 인덱스/불변량/임계값 설계 — 3d-expert
- [ ] 기존 모델로 필터 P/R 스크리닝 (학습 불필요)
- [ ] 상위 필터 → Step 3 downstream 검증
