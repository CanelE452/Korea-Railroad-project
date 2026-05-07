# 작업 기록

## 개요

팔레트 6D 포즈 추정 프로젝트의 주요 작업 이력. 최신순 정렬.

---

## 2026-04-13

### T1 Filter Ablation 파이프라인 디버그 + 실행
- `self_train.py` end-to-end 검증: YAML encoding(UTF-8), belief_map_size(56), logging 섹션, pseudo-loader drop_last 수정
- `config/stage3_selftrain.yaml` 에 camera/pallet/logging 섹션 추가
- Dry-run 성공 (ransac 1r×1ep, 6/188 accepted, loss 수렴)
- **4-filter ablation 런치**: ransac / bc / conf / none × 1 round × 5 epochs

### 실험 ③ 결과 — Joint scratch 실패 확정
- `v8_exp3_coord_scratch` (warmup=10, 60ep): PnP 28.2% — pretrain(48.4%) 보다 나쁨
- Sequential ft(`v8_ablation_A_coord`): PnP 72.3% — 압도적 승자
- 원인: DOPE 50×50 belief 에서 coord gradient 가 belief 형성 간섭

### v8_coord_ft 재현 성공
- seed 9587 로 재현: noapril PnP 75.5% (원본 72.3%) — 결과 일치 확인
- coord ft 효과 실재 확정

### T8 Forgetting ★ 완료
- 9 모델 × synthetic val 200장 평가
- coord ft 후 syn PCK@10 +4pp, syn PnP +7pp — forgetting 없음, 오히려 개선
- self-training R1 도 syn PnP +12pp — 전혀 forgetting 없음

### Inference Speed ★ 완료
- RTX 3080: 총 37.4ms, **26.7 FPS** (DOPE forward 22.5ms, PnP 4.5ms)

### docs 전체 λ_coord 0.005 → 0.003 일괄 수정
- 10개 stale reference + 3개 memory 파일 수정
- `_docs/experiments/loss/coord_strategy.md` ★ 완료 (3-way 비교)
- `_docs/experiments/self_training/forgetting.md` ★ 완료
- `_docs/experiments/eval/inference_speed.md` ★ 완료
- `_docs/experiments/model_catalog.md` 갱신 (v8_ablation_A_coord = primary anchor)

---

## 2026-04-11 ~ 12

### GT Convention 수정
- capture0403middle gt_final 의 projected_cuboid top/bottom 반전 발견
- Permutation `[3,2,1,0,7,6,5,4]` 검증 (0.00 px exact match)
- `gt_final_isaac/` 440 파일 생성

### evaluate_real.py + pnp_solver.py 수정
- PalletPnPSolver: `keypoints_3d` override 파라미터 추가
- evaluate_real.py: Isaac dims (1.1×1.3×0.11) + Isaac corner ordering 사용
- metrics.py: ADD-S 통합 (PoseEvaluator)

### mixed_v10 실패 + 원인 확정
- test_indoor_v1 annotation: 3D cuboid 0.8mm scale (정상 1100mm) — degenerate
- v10 pretrain: noapril 0/9 검출 66% — 학습 전반 오염
- v10 계열 전부 폐기

### v8_ablation_A_coord 생성 (baseline)
- mixed_v8/ep60 → coord ft 5ep (lr=5e-5, λ=0.003) → v8 ablation A 슬롯 완성
- noapril: PnP 71.3% (pretrain 48.4% 대비 +22.9pp)
- middle: PnP 44.1%, 0/8 28.9% (pretrain 56.6% → 28.9% 반감)

### Filter dispatch 구현 + smoke test
- `self_train.py` 에 filter_type 분기 (ransac/bc/conf/none)
- `extract_peaks()` 3-tuple (u,v,conf) 반환
- 4개 filter 10프레임 smoke test 통과

### RANSAC Filter 재선정 (GT P/R)
- 23개 필터 후보 × 2 모델 F1 비교
- F11 RANSAC subset consensus (c≥6): F1=0.833 (ep68) / 0.722 (r1)
- Canonical B∧C: F1=0.235 / 0.069 → reject

---

## 2026-04-02 ~ 10

### Structural Loss Line 종결
- coord/edge/flip/VP loss ablation → coord-only 가 global optimum
- VP loss = dead on arrival (gradient ≈ 0, belief MSE 가 이미 projective prior 학습)
- One-sided collapse mechanism 정리

### Self-Training Round 1 (legacy filter)
- v8_A_control + canonical B∧C filter → selftrain_r1
- capture0403middle good=27/440 @ 50px (이전 세션 기준)

### AprilTag GT 생성
- capture0403middle 440장 multi-tag GT
- gt_manual → gt_final 정제 파이프라인

---

## 2026-03-30

### Blender 합성 데이터 학습 + 실험 관리 체계
- Blender 생성 데이터(`blender_train_v1`, 4000장) 학습 파이프라인 구축
- `train_dope.sh`에 `--exp_name`, `--train_dir`, `--val_dir` CLI 오버라이드 추가
- `scripts/compare_experiments.py` 신규 — `weights/*/eval_results/` 스캔하여 메트릭 비교 테이블 출력
- YAML 파서 인코딩 수정 (UTF-8), 절대경로 처리, `PYTHONUNBUFFERED=1` 추가

### 멀티소스 학습 실험
- **blender_v1**: Blender only 3600장 학습 → PCK@3px=0.624, PnP 99.5%
- **combined_v1**: Isaac Sim 6000 + Blender 3600 = 9600장 → PCK@3px=0.498, PnP 79.0%
- **mixed_v1** (진행 중): Isaac Sim 기본 2000 + far 2000 + Blender 4000 = 8000장 (1:1 비율)
- 결과 시각화를 `data/pallet/eval_results/{exp_name}/`에, eval_summary.json을 `weights/{exp_name}/eval_results/`에 저장

### Real Test 파이프라인 구축
- `data/pallet/real_data/` split 구조: real_test_seen / real_test_unseen / real_unlabeled / real_dev / qualitative_panel
- `metadata.csv` 템플릿 — frame_id, split, pallet_id, distance_bin, occlusion, load, lighting 등
- `scripts/data_prep/apriltag_gt.py` — AprilTag detection → pallet GT pose → NDDS 포맷 저장 → overlay 검증
- `scripts/data_prep/evaluate_real.py` — Real test 전용 평가 (ADD, 5cm5°, Reproj) AprilTag GT 기반
- 촬영 프로토콜 문서 (`data/pallet/real_data/README.md`) — tag-on/tag-off, 조건 분포표
- CLAUDE.md 업데이트 — real test 명령어, 평가 체계 반영

### 3D 부피 비교 메트릭 추가
- `evaluate_on_val.py`에 `compute_volume_from_keypoints()` 추가
- 예측 2D keypoint를 PnP depth로 back-project → 3D cuboid 부피 계산 → GT 부피(0.1815m³)와 비교
- Volume ratio (mean/median/std), |ratio-1| < 20%/50% 메트릭
- `compare_experiments.py`에 Vol Ratio, Vol<20% 컬럼 추가

---

## 2026-03 초

### config 통합 및 TensorBoard 설정
- `config/default.yaml` 생성 — 모든 하이퍼파라미터 단일 소스로 통합
- `train_dope.sh`를 `default.yaml`에서 값을 읽도록 리팩토링
- `scripts/launch_tensorboard.py` 추가 — 실험별 loss 비교, 요약 출력
- TensorBoard 설치 (`pallet-pose` conda env)
- scripts/ 하위 디렉토리에 CLAUDE.md 추가 (data_prep, dope, self_training)

### docs 구조 재편
- `_docs/` 하위를 preprocessing, method, experiments, survey, history로 분리
- 키포인트 정의 문서 복원 (`preprocessing/keypoint_definition.md`)
- 합성 데이터 파이프라인 문서 추가 (`preprocessing/data_pipeline.md`)

### 상태줄 최적화
- `statusline.sh` Windows 최적화: jq 8회 → 1회 호출, 12초 → 0.15초
- ccusage 직접 호출 제거 (Claude Code 자식 프로세스 대기 문제) → JSON 내장 비용 데이터 사용

---

## 2026-02 ~ 2026-03 (합성 데이터 v11)

### 렌더링 파이프라인 안정화
- Isaac Sim 4.5.0 DLL 충돌 해결: `CUDA_MODULE_LOADING=LAZY`
- 200프레임/배치 재시작 전략 (메모리 누수 대응)
- DLSS 완전 비활성화: `anti_aliasing=0` + `/rtx/post/dlss/enabled=False`

### Domain Randomization 개선 (iter1 → iter13)
- **iter3-5**: 팔레트 색상 override — `diffuse_texture` disconnect + opacity/metallic 고정
- **iter4**: Distractor 색상 — per-distractor material 생성 (`stage.Traverse` 방식 폐기)
- **iter13**: 바닥/벽 텍스처 — USD API `Sdf.AssetPath` 직접 변경 (Replicator API 불가 확인)
- **v11**: 조명 상한 하향 (DomeLight 5000→3500, Main 400K→300K), brightness skip 240으로 완화

### 키포인트 Convention 확립 (Y=UP, v2)
- 메시 노멀 분석 방식 폐기 (팔레트에서 불안정)
- Canonical bbox 방식 도입: `R_canonical = R_yz_swap @ euler(base_rot)`
- ORIENTATION_OVERRIDES 4개 모델 전수 검증 완료
- 검증 스크립트: `verify_keypoints.py`, `DIAGNOSE_MODELS=1` 환경변수 진단 모드

### Nucleus Props 대체
- Isaac Sim 4.5에서 `Simple_Warehouse/Props/` 에셋 없음 확인
- Enhanced primitive fallback: cube 60% + cylinder 20% + cone 10% + sphere 10%

---

## 2026-01 ~ 2026-02 (DOPE 학습)

### Pretrain (pallet_category)
- 합성 데이터 ~2,000장으로 60 epoch 학습
- sigma=4.0 설정 (sigma<1은 gradient vanishing)
- 최종 loss: belief=0.043, affinity=0.004

### Fine-tune (pallet_v11)
- pretrain weight에서 91 epoch 추가 학습 (lr=5e-5)
- 최종 loss: total=0.044 (epoch 117)

### 평가 체계 구축
- `evaluate_on_val.py`: PCK@3/5/10px + PnP reproj error + ADD + 5cm-5°
- `visualize_inference.py`: belief heatmap + keypoint + cuboid wireframe
- `visualize_pretrain.py`: 종합 시각화 (belief + cuboid + PnP yaw/pitch/roll)

---

## 2025-12 ~ 2026-01 (프로젝트 초기)

### 아키텍처 결정
- DOPE (keypoint-based) 선택 — 팔레트는 비대칭 직육면체로 keypoint 방식에 적합
- Depth-free (RGB only) 접근 — 산업 현장 카메라 호환성
- 3단계 파이프라인 설계: 합성데이터 → Geometric Filter → Self-Training

### 환경 구축
- conda env `pallet-pose` (PyTorch 2.10 + CUDA 12.6)
- Isaac Sim 4.5.0 standalone 실행 환경
- Docker 실시간 추론 환경 (RealSense D435i)
- Deep_Object_Pose 서브모듈 통합 + Windows 호환 패치

### Self-Training 모듈 개발
- `scripts/self_training/` 5개 모듈 구현
- Geometric Filter 3단계 설계 (Augmentation Consistency + Edge Consistency + Pallet Ratio)
- PnP solver: EPnP + RANSAC wrapper
- FixMatch augmentation: photometric-only (좌표 불변)
