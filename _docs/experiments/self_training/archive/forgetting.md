# T8. Catastrophic Forgetting

상태: **★ 완료** (2026-04-13, 9 모델 × 3 데이터셋)

## 목적

Coord ft / self-training 으로 real 성능이 올라갈 때 synthetic 도메인
성능이 유지되는지 검증. 감소폭 < 5% 면 "no catastrophic forgetting" 주장
가능.

## 실험 설계

학습 없이 **inference 만** 수행. 기존 모델 9 개를 동일 synthetic val set
(200 frames, mixed_v8_val) 에서 PCK@3/10 + PnP rate 측정.
추가로 noapril (188 real, GT 없음) + middle (440 real, GT 있음) 도
동시 측정하여 cross-domain 테이블 구성.

## Table 8 — 결과

```
model              syn PCK@3  syn PCK@10  syn PnP%  noapril PnP%  mid PnP%  mid 0/8
──────────────────────────────────────────────────────────────────────────────────────
v8_pretrain          1.1%       7.4%      54.5%       48.4%        23.9%    56.6%
v8_coord_ft ★        1.8%      11.4%      61.5%       72.3%        44.1%    28.9%
v8_coord_ft_repro    1.7%      12.4%      70.0%       75.5%        46.6%    32.0%
v8_coord_scratch     0.5%       8.9%      59.5%       28.2%        17.7%    70.7%
v8_B_edge            1.2%      12.3%      69.0%       70.2%        41.1%    35.7%
v8_C_coord_edge      1.9%      12.4%      70.0%       64.9%        35.2%    31.1%
v8_D_flip            1.2%      11.3%      57.0%       31.4%        12.3%    52.3%
v8_E_rel             0.7%      11.4%      62.5%       75.5%        18.2%    58.2%
selftrain_r1         0.8%      10.6%      66.5%       49.5%        23.2%    33.0%
```

## 결론

1. **Forgetting 전혀 없음** — coord ft 후 synthetic val 이 오히려 개선:
   - PCK@10: 7.4% → 11.4% (+4pp)
   - syn PnP: 54.5% → 61.5% (+7pp)
   - 재현(repro): PCK@10 12.4%, syn PnP 70.0% — 더 좋음

2. **Self-training R1 도 synthetic 유지**: PCK@10 10.6%, syn PnP 66.5%
   (pretrain 대비 +3.2pp / +12pp)

3. **coord loss 가 synthetic 에서도 regularizer 역할**: belief peak 위치를
   보정하면서 synthetic val 의 keypoint 정확도도 함께 올림.
   → "coord ft 는 domain-agnostic 개선" 이라는 논문 주장 가능.
   이 발견은 T4 Coord Strategy 의 "sequential ft 가 joint scratch 보다
   압도적" 결론과 양면: coord loss 는 belief 수렴 후 추가하면 synthetic/real
   모두를 개선하지만, 학습 초기에 추가하면 belief 형성을 방해한다.
   상세: [`../loss/coord_strategy.md`](../loss/coord_strategy.md).

4. **v8_D_flip 은 real 에서 재앙 (31.4%) 이지만 syn 은 57% 유지**:
   flip loss 가 real generalization 만 해치고 synthetic 은 별 영향 없음.

판정: **|Δ Syn PCK| = +4pp (개선)** → forgetting 없음 ✅

## 관련

- Coord strategy: [`../loss/coord_strategy.md`](../loss/coord_strategy.md)
- Loss ablation: [`../loss/ablation.md`](../loss/ablation.md)
- 평가 스크립트: `scripts/data_prep/eval/evaluate_on_val.py`
- Val data: `data/pallet/training_data/val/` (1500장, 200 사용)
