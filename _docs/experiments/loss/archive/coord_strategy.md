# T4. Coord Loss 학습 전략 비교

상태: **★ 완료** (2026-04-12, v8 데이터 3-way 비교)

## 목적

[`ablation.md`](./ablation.md) 에서 coord Huber 가 최적 loss 임은 확정됨.
이 실험은 **coord loss 를 학습 파이프라인 어느 시점에 도입하는 것이 최적
인가** 를 묻는다.

Sun et al. (2018) 의 Integral Pose Regression 은 heatmap + coordinate 를
epoch 0 부터 동시 학습하지만, DOPE 의 coarse belief (50x50) 에서도 동일
전략이 유효한지 검증.

## 실험 설계

```
고정                    값
──────────────────────────────────────────────────────────
data                    mixed_v8_train (9,000장)
batch                   4
image size              448
sigma                   4.0
coord λ                 0.003
coord δ (Huber)         0.03
struct_edge/flip/vp     0.0 (coord-only)
```

변경 축:

1. **v8_pretrain** — 60 ep belief+aff only (coord 없음, baseline)
2. **v8_coord_ft** — 60 ep belief → 5 ep coord ft (sequential, lr=5e-5)
3. **v8_coord_scratch** — 60 ep belief+coord from scratch (warmup=10, lr=1e-4)

## Table 4 — 결과

### noapril (188 frames, GT 없음, proxy metric)

```
method                  0/9 ↓      kps>=4 ↑   kps>=6 ↑   PnP OK ↑   9/9 ↑
──────────────────────────────────────────────────────────────────────────────
v8_pretrain             23.9%      48.4%      11.7%      48.4%      10.1%
v8_coord_ft ★           3.2%      81.9%      21.8%      72.3%       6.9%
v8_coord_scratch        33.0%      27.7%       8.0%      28.2%       1.6%
```

### capture0403middle (440 frames, GT-based)

```
method                  0/8 ↓      PnP OK ↑   reproj-S med ↓   eval frames ↑
──────────────────────────────────────────────────────────────────────────────
v8_pretrain             56.6%      23.9%         6.8 px              58
v8_coord_ft ★          28.9%      44.1%         6.3 px             117
v8_coord_scratch        70.7%      17.7%        10.9 px              80
```

### 학습 비용

```
method              epochs   lr     warmup   wall time   GPU hours
──────────────────────────────────────────────────────────────────
v8_pretrain         60       1e-4   —        ~10 h       10
v8_coord_ft         60+5     5e-5   0        ~10.8 h     10.8
v8_coord_scratch    60       1e-4   10       ~10 h       10
```

## 결론

1. **Sequential ft (v8_coord_ft) 가 압도적 승자**
   - pretrain 대비: PnP +24pp (noapril), +20pp (middle)
   - scratch 대비: PnP +44pp (noapril), +26pp (middle)
   - 추가 비용: 0.8h only (5ep ft)

2. **Joint scratch (v8_coord_scratch) 는 pretrain 보다 나쁨**
   - noapril: PnP 48.4% → 28.2% (−20pp)
   - middle: PnP 23.9% → 17.7% (−6pp)
   - 0/8 완전실패: 56.6% → 70.7% (+14pp 악화)

3. **원인 분석**
   - Belief map 해상도 50x50 은 Sun et al. 의 64~128x128 보다 거침
   - warmup=10 ep 시점에서 belief 가 아직 불안정 → coord gradient 가 간섭
   - Sequential ft 는 belief 60ep 완전 수렴 + lr 절반(5e-5) → coord 가 belief 을 망가뜨리지 못함
   - 결국 "coord loss 는 belief 수렴 이후에만 안전하다"

4. **Domain-agnostic regularizer 효과** (T8 Forgetting 과 연결)
   - coord ft 후 synthetic val 이 오히려 개선: PCK@10 +4pp, syn PnP +7pp
   - self-training R1 도 syn PnP +12pp — forgetting 전혀 없음
   - coord loss 는 belief peak 위치를 domain 무관하게 보정
   - 상세: [`../self_training/forgetting.md`](../self_training/forgetting.md)

5. **Sun et al. (2018) 과의 차이**
   - 해상도 차이: 50x50 (DOPE) vs 64~128x128 (Sun)
   - Sun 은 heatmap + L1 coordinate 를 epoch 0 부터 동시 학습 → 잘 됨
   - DOPE 에서는 같은 전략이 belief 형성을 방해 → 실패
   - "coarse belief resolution 에서는 sequential injection 이 필수"

## v10 참고 (폐기)

mixed_v10 (v8 + test_indoor_v1 1000장) 으로도 동일 실험을 시도했으나,
test_indoor_v1 의 3D cuboid annotation 이 degenerate (0.8mm scale,
정상 1100mm) 하여 pretrain 자체가 오염됨 (0/9 검출 66%).
v10 계열 전부 폐기. 상세: `_docs/history/2026-04-12.md`.

## 관련

- Loss ablation (coord vs edge/flip 비교): [`ablation.md`](./ablation.md)
- Loss 정의: `_docs/method/formulation.md` §9.2
- 코드: `scripts/train_dope.sh`, `Deep_Object_Pose/train/train.py`
- Weights: `weights/mixed_v8/ep60`, `weights/v8_ablation_A_coord/ep65`, `weights/v8_exp3_coord_scratch/ep60`
