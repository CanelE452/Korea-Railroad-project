# mixed_v7_sym

## 핵심 변경 3가지

```
변경                  내용                                          효과
──────────────────────────────────────────────────────────────────────────────────────
1. Symmetric Loss     180° swap (0↔4,1↔5,2↔6,3↔7) 중 min 채택     앞/뒤 혼란 해소
2. 어두운 팔레트 데이터  blender_dark 100장 (black, charcoal 등)     어두운 팔레트 감지
3. Augmentation 조정   밝기 ±35%, 색조, 감마 (중간 강도)            색상 도메인 확장
```

## 학습 설정

```
Weight:     weights/mixed_v7_sym/final_net_epoch_0091.pth
초기 weight: mixed_v6_full ep60
Epochs:     61 → 91 (31 epoch finetune)
Batch size: 4
LR:         5e-5
Sigma:      4.0
Image size: 448
소요시간:   5시간 16분
특수 설정:  --symmetric_loss
```

## 학습 데이터

```
데이터:     data/pallet/training_data/mixed_v6_full_train/
이미지 수:  9,100
구성:       mixed_v1_train 8K (Isaac 4K + Blender 4K)
            + blender_dark 100 (어두운 팔레트)
            + blender_view 1K (다양한 카메라 각도/거리)
```

## Symmetric Loss 상세

팔레트는 앞/뒤가 시각적으로 거의 동일 → 기존 MSE는 모순된 GT를 줌.

```
기존:     loss = MSE(pred, gt)
변경:     loss = min(MSE(pred, gt_orig), MSE(pred, gt_swapped))

swap 매핑: 0↔5, 1↔4, 2↔7, 3↔6, centroid(8) 유지  (180° Y축 회전, 좌우 뒤집힘 반영)
```

네트워크가 앞/뒤를 헷갈려도 penalty 없음 → belief map 확신도 향상.

## 평가

```
메트릭             mixed_v1 (baseline)   mixed_v7_sym          변화
──────────────────────────────────────────────────────────────────────────
Val PCK@3px        0.469                 0.506                 +3.7%  ★
Val PCK@10px       0.731                 0.715                 -1.6%
Val PnP Rate       72.5%                 86.0%                 +13.5% ★
Val Reproj <10px   6.9%                  11.0%                 +4.1%
Val Vol Ratio      1.159                 1.060                 개선 (→1.0)
Val Vol<20%        55.3%                 63.5%                 +8.2%  ★
Val Vol<50%        93.6%                 94.4%                 +0.8%
Real Avg KP        3.2/9                 3.4/9                 +0.2
Real PnP Rate      30.9%                 37.2%                 +6.3%  ★
```

## 어두운 팔레트 감지 비교 (capture0403noapril)

```
이미지                mixed_v1   mixed_v6_full   mixed_v7_sym
──────────────────────────────────────────────────────────────
밝은 팔레트 (첫번째)   9/9        9/9             7/9
어두운 팔레트 (100번)  0/9        3/9             3/9
어두운 팔레트 (150번)  0/9        3/9             6/9 ★
전체 평균              3.2/9      2.9/9           3.4/9
전체 PnP              30.9%      26.6%           37.2%
```

## mixed_v6_full 대비 개선 요인 분석

```
요인                     기여도   근거
──────────────────────────────────────────────────────────────────────────
Symmetric Loss           높음     Val PnP 66%→86% (+20%) — 앞뒤 혼란 해소로 belief 확신도 향상
어두운 데이터 (v6에서)    중간     어두운 팔레트 0/9→3/9 (v6에서 이미 반영됨)
Augmentation 조정 (v6에서) 낮음   과도한 aug 제거로 학습 안정화
```

## 비고

- **거의 모든 메트릭에서 역대 최고** — 단일 변경(symmetric loss)으로 가장 큰 효과
- 앞/뒤 방향 구분은 포기 (180° yaw 모호성) — 팔레트 활용에 실용적 문제 없음
- 다음 단계: geo loss finetune로 cuboid 3D 형태 추가 개선 가능
