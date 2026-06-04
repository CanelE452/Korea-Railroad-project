# Self6D++ vs Ours (Pallet Pose) — 방법론 & 실험 설계 비교

> 작성: 2026-06-01
> 비교 대상: `~/Documents/github/self6dpp` (Self6D++, TPAMI 2021) vs 본 프로젝트(pallet-pose)
> 목적: ① 두 방법론의 구조·철학 정리, ② 평가 메트릭/실험 설계 관점 비교 및 차용 가능 지점 도출

두 방법 모두 **"PBR/합성으로 pretrain → 라벨 없는 real에서 self-improve"** 라는 동일한 큰 골격을 공유한다.
차이는 ① real 적응 신호를 **무엇으로** 만드는가, ② pose를 **어떻게** 표현/복원하는가 두 축에 집중된다.

---

## 1. 방법론 정리

### 1.1 한눈 비교표

| 축 | **Self6D++ (논문)** | **Ours (pallet-pose)** |
|---|---|---|
| Pose 표현 | **Dense 2D-3D correspondence** (GDR-Net: XYZ map + region + mask) | **Keypoint** (DOPE: 9 belief + 16 affinity, cuboid 8코너+centroid) |
| Backbone | ResNeSt50d / ResNet34 (256×256 in) | VGG-19 + 6-stage CPM (448×448 in, 56×56 out) |
| Pose 복원 | PnP-Net(ConvPnPNet) **direct regression** (6D rot + SITE trans) | **EPnP + RANSAC** (closed-form, gradient 불필요) |
| Refiner | **DeepIM**(FlowNetS, render-and-compare iterative) | 없음 (PnP 결과가 최종) |
| Real 적응 신호 | **Differentiable rendering(DIBR) self-supervision** — render vs real 일관성 | **Geometric Filter(RANSAC consensus)로 pseudo-label 선별 → finetune** |
| Teacher-Student | **Mean Teacher (EMA)**, teacher가 pseudo pose/mask/xyz 생성 | Teacher = 직전 round 모델, hard pseudo-label만 사용 |
| 센서 | **RGB-D** (depth로 chamfer/geom loss) | **RGB only** (depth 미사용) |
| 합성 데이터 | PBR synthetic | Isaac Sim 4.5 + Replicator (NDDS, structured DR) |
| 대상 | LM / LMO / YCBV (소형 가정용 물체 13~21종) | KS T-11 팔레트 1종, 1100×1100×150mm |
| 학습 단계 | Stage I (det+pose+refiner) → Stage II (self-sup) | Step1(pretrain) → Step2(filter+pseudo) → Step3(finetune), 순환 |

### 1.2 Self6D++ 핵심 (근거: `self6dpp/`)

- **Stage I** — PBR synthetic으로 3개 모듈 각각 supervised 학습
  - Detector: YOLOv4 (`det/yolov4/train_yolov4.sh`)
  - Pose estimator: GDR-Net (`core/gdrn_modeling/train_gdrn.sh`) — dense XYZ/region/mask → ConvPnPNet → R(6D)+T(centroid_z, SITE)
  - Refiner: DeepIM (`core/deepim/train_deepim.sh`) — FlowNetS 기반 render-and-compare
- **Stage II** — self-supervised (`core/self6dpp/engine/self_engine.py`)
  - Teacher(EMA mean-teacher)가 real 이미지에서 pseudo pose/mask/xyz 생성 → DIBR 렌더 → real과 일관성 loss로 student 학습
  - **Loss 구성** (`self_engine_utils.py`, config `SELF_LOSS_CFG`):

    | Loss | weight(예) | 비교 대상 |
    |---|---|---|
    | MASK_INIT_REN | 1.0 | pseudo mask ↔ rendered mask (edge-weighted BCE/Dice) |
    | MS_SSIM | 1.0 | real RGB ↔ rendered RGB (structural) |
    | LAB | 0.2 | real ↔ render, Lab의 a,b채널 (조명 불변 photometric) |
    | PERCEPT | 0.15 | AlexNet perceptual |
    | **GEOM (chamfer)** | **100.0** | real depth ↔ rendered depth point cloud (**occlusion-aware**) |
    | SELF_PM | 10.0 | pseudo pose ↔ pred pose (point matching, sym/disentangled) |

  - **Occlusion-aware**: DIBR가 내는 per-pixel prob map + real depth로 가시 영역만 마스킹하여 loss 계산
  - **Renderer**: DIBR(VertexColorBatch), `lib/dr_utils/dib_renderer_x/` — color/depth/mask/xyz/prob 동시 출력
- **철학**: pseudo-label을 *버리지 않고*, 미분 가능 렌더링으로 real 관측과의 **광학적·기하학적 정합**을 직접 backprop. depth가 강한 supervision을 제공.

### 1.3 Ours 핵심 (근거: `scripts/self_training/`, `_docs/method/`)

- **Step1** — Isaac Sim 합성(~6k train) → DOPE scratch 학습 60ep (MSE, 6-stage intermediate sup, σ=4.0)
- **Step2** — real unlabeled inference → keypoint peak(sub-pixel) → **Geometric Filter 3-gate**
  - Pre: keypoint ≥ 5
  - Main: **RANSAC subset consensus** (n_iter=50, subset=5, reproj τ=5px, consensus ≥ 6)
  - Sanity: 복원 팔레트 너비 0.5~2.5m
  - 통과 프레임만 hard pseudo-label로 저장 (`geometric_filter.py`)
- **Step3** — synthetic(GT) + pseudo real(strong aug) 혼합 finetune, `L = L_syn + λ·L_real`, round 반복
  - 수렴: acceptance rate 변화 < 1% 3 round 연속
- **철학**: pseudo-label을 **선별(filter)** 하는 데 집중. 미분 불가능한 EPnP+기하 제약을 *게이트*로만 쓰고, 학습 신호는 여전히 keypoint MSE. depth 없이 RGB+기하 지식으로 신뢰 프레임을 고름.

### 1.4 장단점 대비

| | Self6D++ | Ours |
|---|---|---|
| 강점 | 모든 pseudo 프레임 활용(버림 없음), 미분 렌더로 정밀 정합, refiner로 추가 보정, occlusion 견고 | 단순/안정, RGB-only(센서 부담↓), 기하 제약이 잘못된 라벨을 원천 차단, gradient 불필요로 디버깅 쉬움 |
| 약점 | RGB-D 필요, 미분 렌더러+3 모듈로 파이프라인 무겁고 학습 불안정 위험, mesh 텍스처 품질 의존 | filter가 strict하면 acceptance rate 매우 낮음(초기 ~3%), 버린 프레임의 정보 손실, 미세 pose 보정 메커니즘 없음, EPnP가 keypoint 노이즈에 민감 |

---

## 2. 평가 메트릭 & 실험 설계 비교

### 2.1 메트릭 대응표

| 측정 의도 | Self6D++ | Ours | 비고 |
|---|---|---|---|
| 3D 자세 정확도 | **ADD(-S)** @0.02/0.05/0.1d | **ADD** < 0.1·diameter (real only) | 동일 계열. 우리는 단일 임계(0.1d)만 사용 |
| 회전+병진 동시 | **ReTe** (2°2cm/5°5cm/10°10cm) | **5cm-5°** | Self6D++가 다중 임계로 더 세분화 |
| 2D 투영 정확도 | **Proj** @2/5/10px | **Reproj error (mean px)** | 우리는 비율(%) 아닌 평균값 → 임계 기반 %로 바꾸면 직접 비교 가능 |
| Keypoint 정확도 | (없음, dense) | **PCK@3/5/10px** | keypoint 표현 고유 메트릭 |
| 크기 타당성 | (없음) | **Volume Ratio** (pred/GT cuboid) | 우리 고유, 물리 sanity |
| GT 출처 | 데이터셋 GT pose | **AprilTag** real GT | 우리는 real GT를 직접 취득 |

근거: Self6D++ `core/self6dpp/engine/gdrn_custom_evaluator.py` (`add/adi/re/te/arp_2d`, `lib/pysixd/pose_error.py`) /
Ours `scripts/data_prep/eval/evaluate_on_val.py`, `evaluate_real.py`, `scripts/self_training/metrics.py`

### 2.2 실험 설계 관점 비교

| 관점 | Self6D++ | Ours |
|---|---|---|
| 벤치마크 | 공개 표준(LM/LMO/YCBV) → SOTA 직접 비교 가능 | 자체 팔레트 데이터 → 외부 비교 불가, ablation 중심 |
| 실험 축 | object별, with/without refiner, self-sup loss ablation | round(R0/R1/R2), filter type(ransac/none/loo), domain(night/outside/indoor), epoch(ep65/96) |
| 네이밍 | config 파일명에 인코딩 (`ss_v1_dibr_..._ape`) | 폴더명에 인코딩 (`pl_[domain]_R[round]_[filter]_[variant]`) |
| 결과 관리 | per-object metric 표 (config 주석에 baseline 기록) | `eval_summary.json` + `compare_experiments.py`로 표 생성 |
| 대표 수치 | APE(LM): AD@0.1d 75.7%, ReTe@5°5cm 95.5%, Proj@2px 86.7% | f5 best: PCK@3px 60.5% / syn-only baseline 18.9% (self-training으로 +41.6%p) |

### 2.3 우리 프로젝트에 일반화 가능한 점 (차용 후보)

> 아래는 "방법 차용"이 아니라 **실험 설계/평가 차원**에서 우리 repo에 바로 적용 가능한 것들.

1. **다중 임계 metric 도입** — 현재 5cm-5°, ADD<0.1d는 단일 임계. Self6D++처럼 ADD@0.02/0.05/0.1d, ReTe(2/5/10) 다단계로 보고하면 모델 개선 폭을 더 민감하게 추적 가능. (`metrics.py`에 임계 배열만 추가)
2. **Proj을 %-기반으로** — 현재 Reproj는 mean px. Self6D++의 Proj@2/5/10px처럼 임계 통과율로 함께 보고하면 outlier에 덜 휘둘리는 지표 확보.
3. **AUC(ADD) 추가** — 단일 임계 대신 0~0.1d 구간 AUC를 쓰면 임계 선택 편향 제거, 논문류 비교에 유리.
4. **per-difficulty 분해** — Self6D++의 LMO(occlusion) 분리 평가처럼, 우리도 domain(night/outside/indoor)·가림 정도별로 metric 분해 보고 (이미 실험은 domain별 분리되어 있으니 평가 표만 분해).
5. **(방법 차용, 선택) Soft self-supervision 신호** — filter로 *버리는* 프레임이 많은 게 약점(초기 acceptance ~3%). Self6D++식 mask/silhouette 일관성을 *보조 loss*로 추가하면 버려진 프레임도 약한 신호로 활용 가능. 단 미분 렌더러 도입 비용 큼 → PCK 정체 시에만 검토.
6. **EMA mean-teacher** — 현재 teacher=직전 round 체크포인트(hard switch). EMA로 부드럽게 갱신하면 self-training 안정성↑, round 간 진동↓ (구현 비용 낮음, 우선 검토 권장).

---

## 3. 결론 요약

- **같은 가족, 다른 신호**: 둘 다 self-supervised domain adaptation이지만, Self6D++는 *미분 렌더링 정합*(soft, depth 활용, 전 프레임 사용)이고 우리는 *기하 필터 선별*(hard, RGB-only, 신뢰 프레임만).
- **우리 설계의 정체성**: RGB-only + 기하/물리 제약 + keypoint는 센서·구현 부담이 낮고 잘못된 라벨을 원천 차단하는 게 강점. 대신 정보 손실(낮은 acceptance)과 미세 보정 부재가 약점.
- **즉시 적용 권장**: 메트릭 다단계화(2.3-1~3) + EMA teacher(2.3-6)는 방법론을 안 바꾸고도 비교성·안정성을 올리는 저비용 개선.
- **장기 검토**: PCK/ADD가 정체되면 Self6D++식 보조 silhouette/photometric loss로 "버린 프레임"을 약한 신호로 재활용하는 hybrid를 고려.

---

### 근거 파일 색인

**Self6D++** (`~/Documents/github/self6dpp`)
- 학습 엔트리: `core/self6dpp/engine/self_engine.py`, `main_self6dpp.py`, `train_self6dpp.sh`
- Loss: `core/self6dpp/engine/self_engine_utils.py`, `losses/depth_bp_chamfer_loss.py`, `losses/pm_loss.py`
- Renderer: `lib/dr_utils/dib_renderer_x/renderer_dibr.py`, `configs/_base_/renderer_base.py`
- 평가: `core/self6dpp/engine/gdrn_custom_evaluator.py`, `lib/pysixd/pose_error.py`
- Config: `configs/self6dpp/ssLM|ssLMO|ssYCBV/`

**Ours** (`~/Documents/github/pallet-pose`)
- Self-training: `scripts/self_training/self_train.py`, `self_train_pseudo.py`, `geometric_filter.py`, `pnp_solver.py`
- 모델: `Deep_Object_Pose/common/models.py`
- 평가: `scripts/data_prep/eval/evaluate_on_val.py`, `evaluate_real.py`, `scripts/self_training/metrics.py`
- 비교 유틸: `scripts/compare_experiments.py`
- 문서: `_docs/survey/survey-6d-pose-estimation.md`, `_docs/method/overview.md`
