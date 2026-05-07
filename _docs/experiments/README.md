# Experiments

팔레트 6D 포즈 연구의 모든 실험은 이 폴더에서 독립 파일로 관리한다. 각
실험 파일은 **하나의 Table 또는 Figure 단위**로 분할되어 있고, 완료되면
그 파일만 업데이트한다.

> 2026-04-12 분할: 기존 단일 `experiments.md` 를 실험 단위 16 파일 → 5 개
> 분야 서브폴더 (filter / loss / self_training / eval / synthetic) 로 재편.

## 폴더 구조

```
_docs/experiments/
├── README.md                    # 인덱스 (this)
├── model_catalog.md             # 모델 카탈로그 (cross-cutting)
├── related_work.md              # T10 Related Work 비교 (cross-cutting)
│
├── filter/                      # 필터 관련 실험
│   ├── ablation.md              # T1 Filter Ablation main
│   ├── selection.md             # T3 Filter Selection P/R
│   └── consensus_sweep.md       # T7 RANSAC consensus sweep
│
├── loss/                        # Loss / coord 관련 실험
│   ├── ablation.md              # T2 Loss Ablation — coord contribution
│   └── coord_strategy.md        # T4 Coord Loss 학습 전략 비교
│
├── self_training/               # Self-Training 루프 관련 실험
│   ├── rounds.md                # F1 ST Round 수렴 Figure
│   ├── alpha.md                 # T6 α (pseudo-label weight) 민감도
│   └── forgetting.md            # T8 Catastrophic Forgetting
│
├── eval/                        # 최종 평가 / 실측 관련
│   ├── seen_unseen.md           # T5 Real Seen vs Unseen
│   ├── inference_speed.md       # Inference Speed breakdown
│   └── qualitative.md           # Qualitative Failure Analysis
│
└── synthetic/                   # 합성 데이터 축
    ├── multisource.md           # T9 Multi-source synthetic 비교 (legacy)
    └── sigma_sensitivity.md     # Sigma Sensitivity (optional)
```

## 진행 상태 인덱스

```
#    파일                                    내용                                     상태
──────────────────────────────────────────────────────────────────────────────────────────────
—    README.md                               인덱스 + 평가 프로토콜 + 정책             —
T1   filter/ablation.md                      Filter Ablation (T1 + F4/F5)              ★ 완료 (2026-04-14)
T2   loss/ablation.md                        Loss Ablation — coord contribution         ★ 완료
T3   filter/selection.md                     Filter Selection P/R (23 후보)             ★ 완료
T4   loss/coord_strategy.md                  Coord Loss 학습 전략 비교                  ★ 완료 (v8 3-way)
F1   self_training/rounds.md                 Single-Round Real-Only ft (F4/F5)         ★ 완료 (2026-04-14)
T5   eval/seen_unseen.md                     Real Seen vs Unseen                       촬영 대기
T6   self_training/alpha.md                  α (pseudo-label weight) 민감도             예정 (Phase 4)
T7   filter/consensus_sweep.md               RANSAC consensus threshold sweep           ★ 완료
T8   self_training/forgetting.md             Catastrophic Forgetting                    ★ 완료 (9 모델, forgetting 없음)
—    eval/metric_validation.md               NN matching pipeline 무결성 검증            ★ 완료 (2026-04-14)
—    eval/inference_speed.md                 Inference Speed breakdown                  ★ 완료 (26.7 FPS, RTX 3080)
T9   synthetic/multisource.md                Multi-source synthetic 비교 (legacy)       부분 (v10 폐기, v8 기준)
—    eval/qualitative.md                     Qualitative Failure Analysis               예정 (Phase 5)
—    model_catalog.md                        모델 카탈로그                               갱신 (F3/F4/F5 추가)
T10  related_work.md                         Related Work 비교 (Knitt / Mueller / Ours) 예정 (Phase 5)
—    synthetic/sigma_sensitivity.md          Sigma Sensitivity                          optional
```

★ = 실측 완료

## 평가 프로토콜 요약

```
1. Filter selection screening (GT-based P/R, frame-level)          → filter/selection.md
   └─ capture0403middle 440 장 (2D projected_cuboid GT)

2. Pseudo-label pool / self-training (no GT)                       → self_training/rounds.md
   └─ capture0403noapril 188 장 (unlabeled pool)

3. Primary model evaluation (2D reproj + PnP rate)                 → filter/ablation.md
   └─ capture0403middle 440 장, 기준 reproj 50 px

4. Real Seen / Unseen (AprilTag GT, ADD + 5 cm 5°)                 → eval/seen_unseen.md
```

## 사용 Metric (상세 정의: `_docs/method/formulation.md` §10)

```
Primary (2026-04-14~)   NN matching (Hungarian) raw kp vs GT projected_cuboid, per-frame <20px
Primary (legacy)        PnP success rate, 2D mean reproj, Filter P/R/F1
Secondary               ADD, 5 cm 5° (Real Seen / Unseen 만)
Screening               PCK @ 3 px (synthetic val)
Appendix                Volume ratio (본문 미사용)
```

**Metric 변경 주의** (2026-04-13): 기존 PnP self-reproj / direct index 는 각각
self-referential / convention-sensitive 문제로 폐기. 현행 NN matching pipeline
의 무결성은 [`eval/metric_validation.md`](./eval/metric_validation.md) 에서
4 test (T1~T4) 로 검증 완료.

## 데이터셋 정책 (중요)

**capture0403middle limitation** — object frame convention mismatch 로 3D
ADD / 5 cm 5° 를 신뢰할 수 없다. 논문 본문 표 어디에도 이 데이터셋의 3D
수치를 넣지 않는다. Primary metric 은 **2D reproj ≤ 50 px** 와 **PnP
success rate**.

Real Seen / Unseen (촬영 완료 후) 에서만 ADD, 5 cm 5° 를 primary 로 사용.

## 공통 학습 설정 (ablation baseline)

모든 v8_* ablation 은 아래 설정을 공유 — `project_ablation_baseline_setup.md`:

```
anchor      = weights/mixed_v8/final_net_epoch_0060.pth
epochs      = 61 → 65  (5 epoch fine-tune)
lr          = 5e-5
batch       = 4
imagesize   = 448
sigma       = 4.0
struct_coord = 0.003  (coord Huber, δ = 0.03)
struct_edge  = 0.0
struct_flip  = 0.0
```

새 ablation 은 `struct_edge` / `struct_flip` / 새 항 만 변경하고 나머지는
유지 — 공정 비교 조건.

## 관련 폴더

- `_docs/filter/` — 필터 전용 실험 (selection, consensus sweep, design rationale)
- `_docs/method/` — 방법론 서술 (overview / step1~3 / formulation / implementation / generalization)
- `_docs/models/` — 모델 카탈로그 원본 (카드 파일들)
- `data/pallet/eval_results/` — 모든 평가 결과 JSON/CSV 원본
