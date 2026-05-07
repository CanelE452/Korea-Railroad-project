# 모델 카탈로그

상태: **갱신** (2026-04-15)

본 프로젝트에서 학습된 모든 DOPE 모델의 요약. 상세 카드는 `_docs/models/`
에 있을 수 있음. 여기는 실험 문서에서 reference 하기 쉬운 요약본.

## 현행 모델

```
모델                    학습 데이터                      이미지 수   초기 weight                           비고
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v8                Isaac + Blender (1:1)             9,000      scratch                               Phase 0 baseline 60 ep
mixed_v8 / ep60         위 scratch 60 ep final            —          —                                     **pretrain baseline**
v8_ablation_A_coord     mixed_v8 + coord ft 5 ep          9,000      mixed_v8/ep60                         **primary anchor** ★ (λ=0.003, seed 8452)
v8_coord_ft_reproduce   위 재현 (다른 seed)                9,000      mixed_v8/ep60                         seed 9587, 결과 일치 확인
v8_exp3_coord_scratch   belief+coord joint 60ep           9,000      scratch (ImageNet)                    warmup=10, **실패** (pretrain 보다 나쁨)
v8_ablation_B_edge      + edge only                       9,000      mixed_v8/ep60                         Loss ablation
v8_ablation_C_coord_edge + coord + edge                   9,000      mixed_v8/ep60                         Loss ablation (실패)
v8_ablation_D_flip      + flip only                       9,000      mixed_v8/ep60                         Loss ablation (실패)
v8_ablation_E_rel       + reliability loss                9,000      mixed_v8/ep60                         noapril good, middle bad
v8_vis                  + visibility-aware reweighting    9,000      mixed_v8/ep60                         ablation (결과 보류)
v8_A_control            v9_A_coord + 3ep ft on v8         9,000      v9_ablation_A_coord/ep65              legacy, 이전 anchor
selftrain_r1            v8_A → ST R1 (legacy filter)      +188 PL    v8_A_control/ep68                     pre-RANSAC 시절
ST_8only                ep60 + 8 PL (B∧C) real-only       8 PL       mixed_v8/ep60                         91 ep, NN <20px 37.3% (middle 440)
T1_none                 ep60 + 64 PL (no filter) mixed    64 PL      mixed_v8/ep60                         5 ep, mixed training 실패
T1_ransac               ep60 + 6 PL (RANSAC) mixed        6 PL       mixed_v8/ep60                         5 ep, oversampling 문제 (5625회/장)
F3                      ep65 + RANSAC+LOO (2,324 pool)    ?          v8_A_coord/ep65                       real-only, 평가 미완
F4                      ep65 + 9 PL (RANSAC only)         9 PL       v8_A_coord/ep65                       real-only 90ep, NN <20px 21.8%
F5 ★                    ep65 + 2 PL (RANSAC+LOO)          2 PL       v8_A_coord/ep65                       real-only 96ep, NN <20px 60.5% (seed=4165)
f5_reproduce            F5 재학습 (다른 seed)              2 PL       v8_A_coord/ep65                       NN <20px 53.9% — seed 민감도 ~6-7pp
mixed_v9                Isaac + Blender + indoor           ~8,500     scratch                               mid-term (test_indoor_v1 포함)
mixed_v10               v8 + test_indoor_v1               10,000     scratch                               **폐기** — annotation broken
v10_exp1_coord_ft       mixed_v10 + coord ft              10,000     mixed_v10/ep60                        **폐기** — pretrain 오염
```

## Legacy 모델 (v1 ~ v7)

```
모델               학습 데이터                      이미지 수   비고
──────────────────────────────────────────────────────────────────────
pallet_category    Isaac Sim train/                 ~2,000     very first baseline
pallet_v11         Isaac Sim train/                 4,000      fine-tune 91 ep
pallet_v11_far     Isaac Sim train/ + far           6,000      far distance 포함
blender_v1         Blender only                     3,600      multi-source ablation (T9)
combined_v1        Isaac 6K + Blender 3.6K          9,600      multi-source ablation (T9)
mixed_v1           Isaac 4K + Blender 4K (1:1)      8,000      first mixed, Fair eval 1 위
mixed_v2 ~ v7      (iteration 단계)                  —          렌더 / DR 실험 반복
```

## 명명 규칙

```
mixed_v{N}                  Isaac + Blender 1:1 혼합, N 차 renderer / DR iteration
mixed_v{N}_train            학습용 train split
mixed_v{N}_val              val split (동일 seed 별도 생성)
v{N}_ablation_{X}           mixed_v{N} base 에 대한 구조적 loss ablation
v{N}_A_control              해당 iteration 의 coord only winner (ablation anchor)
v{N}_a_coord                간략 표기 (v10_a_coord = v10 + coord)
mixed_v{N}_st_{source}      v{N} + self-training on {source}
selftrain_r{k}              legacy ST round 결과 재명명
```

## Primary anchor 방침

- 현재 (2026-04-15): **v8_ablation_A_coord (mixed_v8/ep60 + coord ft 5 ep, λ=0.003)** — Phase 1 ablation 기준 ★
- 재현 실험(seed 9587) 에서 결과 일치 확인 (noapril PnP 72.3% vs 75.5%)
- v10 계열은 test_indoor_v1 annotation broken 으로 전부 폐기
- joint scratch (v8_exp3_coord_scratch) 는 pretrain 보다 나쁨 → sequential ft 가 유일 전략

## 최종 best model (2026-04-14)

- **F5 (`weights/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth`)**
- 설계: ep65 anchor + RANSAC+LOO 필터로 선별한 2 PL + **real-only** fine-tune 96 ep
- NN matching <20px = **60.5%** (seed=4165), 재현 53.9% → 발표 범위 **54~60%**
- ep65 기준 baseline(21.6%) 대비 +38.7pp
- Mixed training (synthetic+PL) 은 전부 실패 — real-only가 유일한 성공 전략

## 관련

- Loss ablation: [`loss/ablation.md`](./loss/ablation.md)
- Multi-source: [`synthetic/multisource.md`](./synthetic/multisource.md)
- Coord strategy: [`loss/coord_strategy.md`](./loss/coord_strategy.md)
- 메모리: `project_ablation_baseline_setup.md`
