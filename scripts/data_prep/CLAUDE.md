# scripts/data_prep/

합성 데이터 생성 + 검증 + AprilTag GT + 평가 + 필터 분석 파이프라인.
2026-04-12 에 역할별 서브폴더로 전면 재편. 루트에는 **shared library 2 개**
만 유지.

## 폴더 구조

```
scripts/data_prep/
├── canonical_filters.py        ★ 공유 lib (filter A/B/C/D 함수, 7+ 파일이 import)
├── visualize_inference.py      ★ 공유 lib (load_model/infer/extract_keypoints, 8+ 파일이 import)
│
├── isaac_sim/                  Isaac Sim SDG (Isaac 내장 Python 전용)
├── blender/                    Blender 합성 데이터 생성
│
├── apriltag/                   AprilTag GT 생성 + 디버그 / 정제 (8 files)
├── eval/                       모델 평가 + 필터 P/R (4 files)
├── visualize/                  Annotation / Pretrain 시각화 (2 files)
├── validate/                   데이터 검증 / 병합 (2 files)
├── inference/                  Pseudo-label 생성 (v4 canonical + 보조 2, 3 files)
└── ablation/                   Loss / pnp ablation 분석 (3 files)
```

## Shared Library (루트)

| 파일 | 설명 |
|------|------|
| `canonical_filters.py` | A / B / C / D 필터 함수 (filter_A/B/C/D, prior_depth/tilt). self_train.py + data_prep/* 다수 파일이 sibling import |
| `visualize_inference.py` | DOPE 추론 + belief + keypoint + cuboid overlay. 다수 파일이 `load_model`, `extract_keypoints`, `infer` 를 sibling import |

## Isaac Sim (`isaac_sim/`)

Isaac Sim 내장 Python 으로만 실행. conda 환경 아님.

| 모듈 | 역할 |
|------|------|
| `gen_replicator_data.py` | 메인 진입점 (SimulationApp, generate_data) |
| `sdg_config.py` | 상수 / 설정값 (경로, 카메라, 색상, 에셋) |
| `sdg_math.py` | 수학 헬퍼 (euler, quat, bbox, camera matrix) |
| `sdg_annotation.py` | NDDS JSON 작성, visibility 계산 |
| `sdg_usd_xform.py` | USD xformOp 제어, prim path resolve |
| `sdg_scene.py` | 씬 구성 (warehouse, props, 조명, 텍스처) |
| `sdg_distractors.py` | 디스트랙터 / 적재물 배치, 카메라 포즈 |
| `generate_all.sh` | 배치 생성 (64프레임 / 배치, Isaac Sim 재시작) |
| `run_iter_verification.sh` | iter 단위 배치 검증 |
| `debug_pallet_orientation.py` | USD 모델별 orientation 진단 |
| `list_isaac_assets.py` | Isaac Sim Nucleus / S3 에셋 경로 탐색 |

## `apriltag/` — AprilTag GT 생성 + 디버그 / 정제

| 파일 | 설명 |
|------|------|
| `apriltag_gt.py` | ★ AprilTag single-tag GT 생성 진입점 |
| `apriltag_gt_multitag.py` | ★ AprilTag multi-tag GT 생성 진입점 (신뢰도 향상) |
| `apriltag_debug_brute.py` | AprilTag 인식 brute-force 디버그 |
| `apriltag_debug_id4.py` | 특정 tag ID 디버그 |
| `apriltag_debug_single.py` | 단일 이미지 tag 디버그 |
| `refine_tag_config.py` | tag 설정 fine-tune |
| `finalize_gt.py` | GT 최종 정리 |
| `gt_editor.py` | GT 수동 편집 |

```bash
python scripts/data_prep/apriltag/apriltag_gt.py --image <path> --visualize
python scripts/data_prep/apriltag/apriltag_gt_multitag.py --input_dir <path>
```

## `eval/` — 모델 평가 + 필터 P/R

| 파일 | 설명 |
|------|------|
| `evaluate_on_val.py` | Synthetic val 평가 (PCK@3/5/10, PnP, ADD, 5 cm 5°) |
| `evaluate_real.py` | Real test 평가 (ADD + 5 cm 5° + Reproj, AprilTag GT 필요) |
| `filter_pr_eval.py` | ★ 23 필터 후보 × 모델 GT 기반 P/R 평가 (필터 선정 스크립트) |
| `quick_screen.py` | 빠른 모델 screening (1 ~ 2 batch 추론) |

```bash
python scripts/data_prep/eval/evaluate_on_val.py --weights <path> --val_dir <path>
python scripts/data_prep/eval/evaluate_real.py --weights <path> --test_dir <path>
python scripts/data_prep/eval/filter_pr_eval.py --weights <path> --gt_dir <path>
```

## `visualize/` — 시각화

| 파일 | 설명 |
|------|------|
| `visualize_annotations.py` | NDDS 9-point cuboid keypoint + pose axis overlay |
| `visualize_pretrain.py` | Pretrain 결과 종합 시각화 (multi-panel) |

※ `visualize_inference.py` 는 shared library 라 루트에 유지.

## `validate/` — 데이터 검증 / 병합

| 파일 | 설명 |
|------|------|
| `merge_and_validate.py` | 배치 병합 + visibility 필터링 |
| `verify_keypoints.py` | Keypoint 기하학 자동 검증 (Y=UP convention) |

## `inference/` — Pseudo-label 생성 파이프라인

RANSAC 필터 도입 (2026-04-11) 이전 세대. 현재는 `scripts/self_training/
self_train.py` 로 통합됨. legacy v1~v3 및 `extract_pl_v5.py` 는 2026-07-23 정리 삭제.

| 파일 | 설명 |
|------|------|
| `infer_and_filter_v4.py` | v4 — B / C 만 (canonical 최종) |
| `infer_test_data.py` | 테스트 데이터 추론 |
| `generate_pseudo_labels.py` | batch pseudo-label 생성 |

## `ablation/` — Loss / pnp ablation 분석

| 파일 | 설명 |
|------|------|
| `compare_struct_ablation.py` | v8_A / B2 / B3 PnP rate + B∧C 비교 (Loss Ablation T2) |
| `measure_vp_scale.py` | VP loss magnitude 측정 (λ 캘리브레이션) |
| `calibrate_pnp_thresholds.py` | PnP threshold 교정 |

## 주의사항

- `isaac_sim/` 내 스크립트는 Isaac Sim standalone 으로만 실행 (conda 환경 아님)
- Isaac Sim ~2 분 / 프레임, 200 프레임마다 재시작 필수 (메모리 누수)
- `ORIENTATION_OVERRIDES` 절대 수정 금지 (검증 완료된 값)
- 어려운 케이스 (낮은 대비, 유사 색상) 는 의도된 것 — 제거하지 말 것
- **`canonical_filters.py` / `visualize_inference.py` 는 루트 유지 — 서브
  폴더 다수 파일이 import**. 이동 시 import chain 전체 깨짐.
- 서브폴더 파일에서 루트 shared lib import 하려면 상단에
  `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` 필요.
