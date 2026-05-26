# Pallet 6D Pose — Geometry-aware Self-Training

팔레트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 DA 프레임워크.
Python + PyTorch + Isaac Sim + DOPE.

## 연구 가이드

최신 연구 설계: `_docs/` (README.md에서 목차 확인)
도메인 서베이: `_docs/survey/survey-6d-pose-estimation.md` (방법론/학습전략/메트릭 비교)

## Commands

### Step 1: 합성 데이터 생성 + DOPE Pretrain
- 합성 데이터 생성 (단일): Isaac Sim에서 `scripts/data_prep/isaac_sim/gen_replicator_data.py` 실행
- 합성 데이터 생성 (배치): `bash scripts/data_prep/isaac_sim/generate_all.sh` (64프레임/배치, 자동 재시작)
- DOPE pretrain: `bash scripts/train_dope.sh` (config/default.yaml 기반)
- DOPE fine-tune: `bash scripts/train_dope.sh --finetune`

### Step 2-3: Self-Training + 평가
- Self-training: `python scripts/self_training/self_train.py` (설정: `config/stage3_selftrain.yaml`)
- Synthetic val 평가: `python scripts/data_prep/eval/evaluate_on_val.py --weights <path> --val_dir <path>` (PCK + PnP Reproj + Volume Ratio)
- Real test 평가: `python scripts/data_prep/eval/evaluate_real.py --weights <path> --test_dir <path>` (ADD + 5cm5° + Reproj)
- 실험 비교: `python scripts/compare_experiments.py`

### Real Data + AprilTag GT
- AprilTag GT 생성: `python scripts/data_prep/apriltag/apriltag_gt.py --image <tag_image> --visualize`
- Real data split: `data/pallet/real_data/{real_test_seen,real_test_unseen,real_unlabeled,real_dev}/`
- 촬영 프로토콜: `data/pallet/real_data/README.md`

### 유틸리티
- Annotation 시각화: `python scripts/data_prep/visualize/visualize_annotations.py`
- 추론 시각화: `python scripts/data_prep/visualize_inference.py --weights <path> --num_syn 10 --num_real 10`
- 데이터 검증: `python scripts/data_prep/validate/merge_and_validate.py`, `python scripts/data_prep/validate/verify_keypoints.py`
- 실시간 추론 (native): `python scripts/dope/run_dope_live.py --realsense --weights <path>` (RealSense SDK + `pip install pyrealsense2` 필요)

## Architecture

- **Pose 표현**: Keypoint-based (DOPE) — 팔레트는 비대칭 직육면체로 keypoint 방식에 적합
- 3단계 파이프라인: Step 1 (합성데이터+DOPE학습) → Step 2 (Geo Filter+Pseudo-label) → Step 3 (Finetuning) → 반복
- DOPE 모델: `Deep_Object_Pose/` (VGG-19 backbone, 9 belief maps + 16 affinity fields)
- **PnP**: EPnP + RANSAC (`scripts/self_training/pnp_solver.py`) — keypoint → 6D 포즈 복원
- 합성 데이터: Isaac Sim 4.5.0 + Omniverse Replicator, NDDS 포맷 JSON annotation
- USD 모델: `data/pallet/models_usd/scene*.usd` (4종 팔레트)
- Geometric Filter: **RANSAC subset consensus** (n_iter=50, k=5, τ=5px, c≥6) + size sanity. 2026-04-11 GT 기반 P/R 분석으로 canonical A/B/C/D 중 1등 선정. 필터 전용 문서: `_docs/filter/` (selection, ablation, sweep 등 누적). canonical 필터는 `scripts/data_prep/canonical_filters.py`에 ablation 용으로 보존.
- Keypoint convention: Y=UP, 8 cuboid corners + centroid (memory 참조)
- **평가 (Synthetic val)**: PCK@3/5/10px + PnP Reproj + Volume Ratio — `scripts/data_prep/eval/evaluate_on_val.py`
- **평가 (Real test)**: ADD + 5cm5° + Reproj — `scripts/data_prep/eval/evaluate_real.py` (AprilTag GT 기반)
- **Real data**: seen/unseen/unlabeled/dev split — `data/pallet/real_data/`
- 실시간 추론: `scripts/dope/run_dope_live.py` + RealSense D435i (native, pyrealsense2)
- 팔레트 규격: KS T-11형 1100×1100×150mm (config에서 관리)

## Code Style

- conda env: `pallet-pose`
- 설정은 `config/default.yaml`에서 중앙 관리 (`train_dope.sh`가 yaml 읽음)
- Isaac Sim 스크립트는 standalone 실행 (Isaac Sim 내장 Python), 모듈화: `scripts/data_prep/isaac_sim/sdg_*.py`
- DOPE 데이터 로더: CleanVisiiDopeLoader (`{i:06d}.png` + `{i:06d}.json` 쌍)

## Gotchas

- Isaac Sim DLL 충돌: `CUDA_MODULE_LOADING=LAZY` + `PYTHONUNBUFFERED=1` 설정 필요
- Isaac Sim ~2분/프레임, 64프레임마다 재시작 (메모리 누수)
- Replicator `rep.distribution.choice()`는 머티리얼 생성 시 1회만 평가됨 → USD API로 직접 변경
- 팔레트 기울기 없이 바닥 수평 고정 (tilt=0)
- 어려운 케이스(낮은 대비, 유사 색상)는 유지 — 모델 로버스트니스에 필수
- Belief map sigma=4.0 유지 (sigma<1은 gradient vanishing 발생, `_docs/method/step1_synthetic_data.md` 3.6절 참조)
- **3D/2D 작업은 항상 `3d-expert` agent 에 먼저 위임** (2026-05-22 사용자 규칙). 좌표계 변환 / cuboid 라벨링 / camera convention (OpenCV vs USD vs ROS) / projection / rendering / annotation 시각화. 직접 trial-and-error 금지. v1/v2/v3 keypoint 변환 4 회 시행착오 후 3d-expert 가 한 번에 v4 해결한 경험. 자세한 근거: `feedback_3d_expert_first.md` 메모리.
- 장시간 작업(데이터 생성, 학습 등) 실행 중에는 주기적으로 로그/프로세스를 확인하여 정상 진행 여부를 모니터링한다

## Self-Verification

- [ ] 연구 설계 변경 시 `_docs/`의 해당 문서도 함께 업데이트했는가?
- [ ] Isaac Sim 스크립트 수정 시 ORIENTATION_OVERRIDES 건드리지 않았는가?
- [ ] 새 스크립트 추가 시 재현성을 위한 config/argparse 지원이 있는가?
- [ ] Geometric Filter 임계값 변경 시 `config/stage3_selftrain.yaml`에 반영했는가?
- [ ] 학습 설정(sigma, batch, lr) 변경 시 `scripts/train_dope.sh`와 `_docs/method/step1_synthetic_data.md` 3.6절 동기화했는가?
- [ ] 생성된 이미지/overlay 확인 시 각 프레임을 개별 로드하여 변경 사항(적재물, 카메라 높이, 배경 등)이 실제 반영되었는지 눈으로 검증했는가? 로그만 보고 판단하지 않는다.
