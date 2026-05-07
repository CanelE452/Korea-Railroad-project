# T1. Filter Ablation — Main Table

상태: **★ 완료** (2026-04-13 T1 실학습, 2026-04-14 후속 F4/F5 추가)

## 목적

논문의 메인 contribution 검증 — "RANSAC subset consensus 가 다른 필터보다
실제 self-training 효과를 낸다" 를 동일 조건 비교로 입증.

## 실험 설계

```
고정                    값
──────────────────────────────────────────────────────────
unlabeled pool          capture0403noapril (188 장)
fine-tune anchor        weights/mixed_v8/final_net_epoch_0060.pth
ft epochs               5
lr                      5e-5
batch                   8 (self_train.py config)
image size              448
sigma                   4.0
coord Huber λ           0.003 (δ = 0.03)
평가                     capture0403middle 440 장, 2D reproj 50 px
```

변경 축: `filter_type ∈ {none, conf, bc, ransac}` — `scripts/self_training/
self_train.py --filter_type {name}`.

## Table 1 — T1 Mixed Training (2026-04-13)

**학습 모드**: synthetic + PL mixed (α=1.0), 5 epoch ft.
**평가 metric 주의**: T1 평가 시점엔 PnP self-reproj 기준이었음. NN matching
재평가는 일부만 존재 — 신규 실험 F3~F5 는 NN matching 기준.

```
Method                              Filter     PL #     NN <20px    비고
────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (ep60 baseline)      —          —        18.9%       middle 440
+ coord ft 5 ep (v8_A ep65)         —          —        21.6%       baseline for F4/F5
T1 none (mixed)                     none       64       17.3%       baseline 보다 낮음
T1 ransac (mixed)                   ransac     6        16.4%       oversampling (5625회/장)
T1 bc (mixed)                       bc         0        —           B∧C 0 장 통과 → synth-only
T1 conf (mixed)                     conf       —        skip        발표 후 예정
```

**결론**: Mixed training (synthetic+PL) 은 전부 baseline 보다 낮음. PL 수량 /
필터 품질과 무관하게 mixed 모드 자체가 실패. → real-only ft 로 전환.

## Table 2 — Single-Round Real-Only Fine-tune (2026-04-14)

**학습 모드**: PL only (synthetic 제거), 90~96 epoch ft.

```
Method                              Filter          PL #    Base   <20px   median
──────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (ep60 baseline)      —               —       —      18.9%   30.4px
+ coord ft 5 ep (ep65)              —               —       —      21.6%   66.2px
ST_8only (real-only 91 ep)          canonical B∧C    8      ep60   37.3%   33.5px
F4 RANSAC 9PL (real-only 90 ep)     RANSAC c≥6       9      ep65   21.8%   77.9px
F5 RANSAC+LOO 2PL (real-only ep96) ★ RANSAC+LOO      2      ep65   60.5%   12.2px
F5 재현 (다른 seed, ep96)            RANSAC+LOO       2      ep65   53.9%   —
```

**핵심 관찰**:
1. F5 (2 PL, LOO 필터) > ST_8only (8 PL, B∧C) > F4 (9 PL, RANSAC only) — PL 수량보다 **필터 품질**이 지배적
2. F4 (21.8%) ≈ ep65 (21.6%) — RANSAC-only PL 로는 self-training 효과 없음
3. LOO 가 선별한 2 장은 mean 6.7px 로 극도로 깨끗 (cf. RANSAC mean 32.1px) → 적은 양으로도 대폭 개선
4. Seed 민감도 ~6-7pp (2 PL 특성) — 발표는 **54~60% 범위**로 보고

## Table 3 — Filter Quality (ep65, middle 440, NN matching)

각 필터 통과 frame 만 모아 NN matching 정확도 측정 — **필터가 실제로 좋은 frame 을 고르는가** 검증.

```
filter                pass   mean     median   <20px   <50px   <100px
──────────────────────────────────────────────────────────────────────
0_all (no filter)      440   62.4px   66.2px   21.6%   30.0%    58.2%
1_prefilter (kps≥5)     99   97.5px  109.8px    2.0%   16.2%    45.5%   ← 오히려 나쁨
2_ransac                 8   32.1px   35.5px   25.0%   87.5%   100.0%
3_size                 191   83.6px   80.2px    3.7%   17.3%    72.8%
4_flip                  87   74.9px   79.6px    1.1%   12.6%    96.6%
5_structural             7   29.3px   25.4px   28.6%  100.0%   100.0%
6_loo                    2    6.7px    6.7px  100.0%  100.0%   100.0%   ★
ransac+loo               2    6.7px    6.7px  100.0%  100.0%   100.0%
ransac+struct            7   29.3px   25.4px   28.6%  100.0%   100.0%
```

**결론**:
- LOO 가 가장 품질 높은 frame 을 고름 (mean 6.7px, <20px 100%) — 그러나 통과 수 극소 (2/440)
- RANSAC 은 중간 품질 (mean 32px) — 통과 수 8~9
- Prefilter (kps≥5) 는 오히려 no-filter 보다 나쁨 — 모델이 자신있게 틀리는 frame 포함

이 Table 이 F5 (LOO 2 PL) 가 왜 ST_8only (B∧C 8 PL) 보다도 높은지 설명.

## 실행 명령 (4 회 반복)

```bash
for FT in ransac bc conf none; do
    python scripts/self_training/self_train.py \
        --config config/stage3_selftrain.yaml \
        --filter_type $FT \
        --num_rounds 1 \
        --epochs_per_round 5 \
        --lr 5e-5 \
        --pretrained weights/mixed_v8/final_net_epoch_0060.pth \
        --real_dir data/pallet/raw_data/capture0403noapril/rgb \
        --output_dir output/filter_ablation_$FT
done
```

각 run ~50 분 (ft 40 분 + pseudo-label 생성 10 분), 4 개 = 약 4 시간.

## 평가 명령

```bash
for FT in ransac bc conf none; do
    python scripts/data_prep/eval/evaluate_real.py \
        --weights output/filter_ablation_$FT/round_01.pth \
        --test_dir data/pallet/raw_data/capture0403middle \
        --out data/pallet/eval_results/filter_ablation_$FT.json
done
```

## 기대 해석

- `ransac` 이 `none` / `conf` / `bc` 를 PnP rate / Reproj 모두에서 초과
  → Filter 가 실제로 품질 신호를 전달했다는 증거
- `bc` 가 `none` 보다 나쁘면 (PL 수량 급감) → canonical filter 의 recall
  한계가 학습에도 영향을 준다는 증거 (negative result contribution)
- `conf` 가 `none` 과 비슷하면 → confidence 필터는 무의미 baseline

## 관련

- 구현: `scripts/self_training/self_train.py::_apply_filter()`
- 필터 선정 근거: `_docs/filter/2026-04-11_selection.md`
- Smoke test: `scripts/self_training/_smoke_test_filter_dispatch.py`
