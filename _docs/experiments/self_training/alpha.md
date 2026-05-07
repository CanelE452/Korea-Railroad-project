# T6. α Sensitivity — Pseudo-label Weight

상태: **예정 (Phase 4)**

## 목적

self-training total loss `L = L_syn + α · L_real` 에서 α 값이 성능에 미치
는 영향. 너무 작으면 ST 효과 없음, 너무 크면 catastrophic forgetting 위험.

## 실험 설계

```
고정                   값
──────────────────────────────────────────────────────────
filter_type            ransac (c≥6)
anchor                 v8_A (v8 coord ft)
pool                   noapril 188 장
ft epochs              5
lr                     5e-5
PL 수                  14 (RANSAC 채택분, α 와 무관하게 동일)
```

변경 축: `α ∈ {0.1, 0.5, 1.0, 2.0}`.

## Table 6

```
α        PL #   PnP% ↑   Reproj↓   Syn val PCK@3 (forgetting)   verdict
─────────────────────────────────────────────────────────────────────────────
0.1      14     ?        ?         ?                             weak ST
0.5      14     ?        ?         ?                             conservative
1.0 ★    14     ?        ?         ?                             default
2.0      14     ?        ?         ?                             aggressive
```

`Syn val PCK@3` 컬럼은 [`forgetting.md`](./forgetting.md) 와 공유 — forgetting 이 α 에 민감
하게 반응하는지 동시 관찰.

## 기대 해석

- α = 0.1 ~ 0.5: ST 효과 미미 or 없음 (pseudo-label loss signal 너무 약함)
- α = 1.0 (default): 최적 추정
- α = 2.0: Real PnP 상승 여부 확인 + Syn PCK 감소 여부 확인
  - Syn PCK 큰 감소 → forgetting, α = 1 유지 권장
  - Syn PCK 유지 + Real 개선 → α = 2 로 갱신 여지

## 실행 명령

```bash
for ALPHA in 0.1 0.5 1.0 2.0; do
    python scripts/self_training/self_train.py \
        --config config/stage3_selftrain.yaml \
        --filter_type ransac \
        --num_rounds 1 \
        --epochs_per_round 5 \
        --lr 5e-5 \
        --lambda_real $ALPHA \
        --pretrained weights/v8_A_control/final_net_epoch_0068.pth \
        --output_dir output/alpha_${ALPHA}
done
```

> CLI 에 `--lambda_real` override 가 없으면 추가 필요.

## 관련

- Loss 정의: `_docs/method/formulation.md` §9.2.4
- Forgetting: [`forgetting.md`](./forgetting.md)
