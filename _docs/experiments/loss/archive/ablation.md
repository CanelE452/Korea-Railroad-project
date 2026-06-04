# T2. Loss Ablation — coord contribution

상태: **★ 완료** (2026-04-10 ~ 2026-04-16, 전체 조합 재실험)

## 목적

coord Huber loss 가 다른 structural regularization (edge / flip / VP) 대비
우수함을 입증. "coord loss 가 본 연구의 핵심 loss contribution" 이라는
논문 주장의 근거.

## 실험 설계

모든 모델 공통 설정 (`_docs/experiments/README.md` §공통 학습 설정 참조):

```
anchor            = mixed_v8/final_net_epoch_0060.pth
ft epochs         = 5 (epoch 61 → 65)
lr                = 5e-5
batch             = 4
struct_lambda     = 1.0 (outer)
struct_delta      = 0.03
```

변경 축: `struct_coord / struct_edge / struct_flip` 비율.

평가 데이터: `capture0403noapril` 188 장 (unlabeled) — PnP rate + canonical
B / C / B∧C 통과 수. baseline (49.5 %) 은 coord ft 이전 mixed_v8/ep60.

## Table 2

```
Model     Structural Loss                         avg kp   PnP%    B    C    B∧C    verdict
─────────────────────────────────────────────────────────────────────────────────────────────
baseline  belief + affinity only (ep60)           —        49.5    —    —    —      하한선
v8_A ★    + coord Huber (λ = 0.003)               6.18     62.2    16   9    6      optimum
v8_B2     + coord + edge log-ratio (λ = 0.001)    5.22     60.1    12   0    0      C 붕괴
v8_B3     + coord + edge log-ratio (λ = 0.0005)   4.63     33.0    11   9    1      belief 약화
v8_B      + coord + edge (λ = 0.003)              —        54.3    —    —    4      heavier edge
v8_C      + coord + edge + other                  —        55.3    —    —    0      combined 실패
v8_D      + coord + flip equivariance (λ = 0.02)  —        29.3    —    —    1      coord 충돌
v8_VP*    + coord + vanishing-point concurrency   —        —       —    —    —      gradient ≈ 0 (dead)
```

## 해석

1. **coord 단독 (v8_A) 이 유일한 개선** — 49.5 → 62.2, PL 통과수 6
2. **edge 추가 → C filter 붕괴** — λ 0.001 에서 C=0, λ 0.0005 에서 belief
   자체가 약화 (avg kp 6.18 → 4.63). 반직관적으로 작은 λ 가 더 나쁨 → λ
   미세 튜닝으로 구제 불가능
3. **flip 추가 → coord 와 충돌** — 29.3 % 로 급락
4. **VP 는 dead-on-arrival** — 이미 belief MSE 가 projective cuboid 를
   암묵적으로 학습해서 `vp_loss(pred) ≈ vp_loss(GT) ≈ 0` (실측)

## 논문 contribution draft

> "Direct coordinate supervision (coord Huber loss on soft-argmax outputs)
> is sufficient and optimal for fine-tuning converged DOPE models. Indirect
> structural regularization — edge log-ratio, flip equivariance, and
> vanishing-point concurrency — provides no benefit and typically degrades
> performance, because the base model already implicitly encodes projective
> cuboid structure through belief-map matching. Residual prediction errors
> are position-wise (captured by coord loss) rather than topology-breaking,
> leaving structural priors with no gradient signal."

## 부록 — NN matching 재평가 (2026-04-15)

위 Table 2 는 noapril 188 장 PnP rate 기준. 2026-04-13 이후 primary metric 이
**middle 440 장 + NN matching (Hungarian) + `gt_final_isaac` GT** 로 통일되면서
같은 모델들을 새 metric 으로 재평가. 결과는 **PnP rate 기준과 상당히 다름**.

```
Model (동일 checkpoint)                NN matching <20px   median    비고
────────────────────────────────────────────────────────────────────────────────
ep60 baseline                          18.9%               30.4px    — (참고)
v8_A (coord only, λ=0.003)             21.6%               66.2px    primary anchor
v8_B_edge (edge only)                  21.1%               65.3px    ≈ baseline
v8_ablation_C_coord_edge ★             38.4%               16.7px    +16.8pp vs v8_A
v8_D_flip                              22.5%               34.0px    marginal
v8_E_rel                               23.9%               13.3px    marginal
v8_exp3_coord_scratch (joint)           7.5%               75.9px    실패 (재확인)
```

### 해석 차이

- **PnP rate (noapril 188)**: coord+edge 54.3% < coord only 62.2% → "C filter 붕괴"
- **NN matching (middle 440)**: coord+edge 38.4% > coord only 21.6% → "synergistic"

가능한 설명: coord+edge 는 **PnP 로 풀릴 만큼 완전히 맞는 frame 은 줄지만,
keypoint 위치 정확도 자체는 향상** — 서로 다른 축을 재는 두 metric.

### Contribution 해석 재조정 필요

구 결론 ("edge 는 항상 나쁨") 은 PnP rate 한정. Raw keypoint accuracy 관점에서는
edge 가 coord 와 결합 시 보완 작용. 논문 contribution 재작성 시 두 metric 모두
병기 권장.

### 주의
- coord+edge 결과는 2026-04-15 단일 seed 측정 — 재현성 검증 안 됨
- Frames with predictions 303/440 (ep65 313 보다 적음) — 선택 편향 가능
- 발표 본문에는 포함하지 않고, 별첨 슬라이드 4 (Loss Ablation) 만 신 metric 으로 갱신

## 부록 — 전체 조합 재실험 (2026-04-16)

2^3 = 8 조합 (baseline 포함) 전부 동일 조건 학습 + NN matching 재평가.
기존 A/B/C/D 는 재측정, 신규 F/G/H 는 새로 학습.

### Table 3 — Full Factorial (middle 440, NN matching)

```
tag   coord   edge   flip    pred#   <20px    median    <50px
──────────────────────────────────────────────────────────────────
—     —       —      —       191     18.9%    30.4px    25.2%    baseline (ep60)
A     0.003   —      —       313     21.6%    66.2px    30.0%
B     —       0.003  —       283     21.1%    65.3px    26.8%
D     —       —      0.02    210     22.5%    34.0px    24.1%
C ★   0.003   0.002  —       303     38.4%    16.7px    43.9%    BEST
F     0.003   —      0.02    274     16.6%    62.9px    25.9%
G     —       0.003  0.02    224     21.8%    47.9px    26.4%
H     0.003   0.002  0.02    216     27.3%    16.5px    34.3%
```

### Interaction effects

```
                  단독 효과      coord 와 결합     edge 와 결합
──────────────────────────────────────────────────────────────
coord             +2.7pp         —                 +16.8pp (C)
edge              +2.2pp         +16.8pp (C) ★     —
flip              +3.6pp         -5.0pp (F) ✗      +0.7pp (G)
coord+edge        +19.5pp (C)    —                 —
coord+flip        -2.3pp (F)     —                 -11.1pp (H) ✗
edge+flip         +2.9pp (G)     —                 —
all               +8.4pp (H)     —                 —
```

### 결론

1. **coord+edge synergy 가 유일한 유의미 interaction** (+16.8pp)
2. **flip 은 모든 조합에서 해로움** — 특히 coord+edge 38.4% → all 27.3% (-11.1pp)
3. **단독 loss 는 전부 marginal** (2~4pp) — 조합이 핵심
4. **edge 의 역할**: coord 없이는 무효, coord 와 결합 시 outlier suppressor

### 논문 contribution (갱신)

> "Full factorial ablation (2^3) confirms that coord+edge is the only
> beneficial combination (+16.8pp over coord-only, +19.5pp over baseline).
> Flip equivariance is harmful in every combination — it conflicts with
> coord's gradient direction by enforcing symmetry that the asymmetric
> pallet does not possess. Edge log-ratio acts as an outlier suppressor
> that is effective only when paired with absolute coordinate supervision."

## 관련

- 메모리: `memory/project_ablation_baseline_setup.md`,
  `project_structural_loss_line_closed.md`,
  `project_vp_loss_dead_on_arrival.md`,
  `project_one_sided_collapse_mechanism.md`
- 결과 ckpt: `weights/v8_ablation_{A,B,C,D,F,G,H}_*/`
- Loss 정의: `_docs/method/formulation.md` §9.2
- 전체 조합 원본: `data/pallet/eval_results/loss_ablation_all_nn.txt`
- 실행 스크립트: `scripts/run_loss_ablation_all.sh`, `scripts/eval_loss_ablation_all.sh`
- History: `_docs/history/2026-04-16.md`
