# scripts/

팔레트 6D 포즈 추정 파이프라인의 실행 스크립트 모음.

## 디렉토리 구조 (2026-04-12 재편)

```
scripts/
├── data_prep/                       ← PyTorch 데이터 준비 + 평가 + 시각화
│   │   (루트: shared library 2 개만 — 상세는 data_prep/CLAUDE.md)
│   ├── canonical_filters.py         공유 lib (filter A/B/C/D)
│   ├── visualize_inference.py       공유 lib (load_model, infer, extract_keypoints)
│   │
│   ├── isaac_sim/                   Isaac Sim SDG (내장 Python)
│   ├── blender/                     Blender 합성 데이터 생성
│   ├── apriltag/                    AprilTag GT 생성 + 디버그 / 정제 (8 files)
│   ├── eval/                        모델 평가 + 필터 P/R (4 files)
│   ├── visualize/                   Annotation / Pretrain 시각화 (2 files)
│   ├── validate/                    데이터 검증 / 병합 (2 files)
│   ├── inference/                   Pseudo-label 생성 (v4 canonical + 보조 2)
│   └── ablation/                    Loss / pnp ablation 분석 (3 files)
│
├── self_training/                   ← PyTorch Self-Training (conda: pallet-pose)
│   ├── self_train.py                메인 루프 (filter_type dispatcher)
│   ├── geometric_filter.py          RANSAC subset consensus + size sanity
│   ├── pnp_solver.py                EPnP + RANSAC
│   ├── augmentations.py             FixMatch weak/strong aug
│   ├── metrics.py                   ADD, 5cm5°, Reproj
│   └── _smoke_test_filter_dispatch.py
│
├── dope/                            ← 실시간 추론 (native, RealSense)
│   └── run_dope_live.py
│
├── train_dope.sh                    ← DOPE 학습 진입점 (conda: pallet-pose)
├── launch_tensorboard.py
└── compare_experiments.py           실험 비교 유틸

※ 2026-07-23 정리: 미등록 일회성 분석/그림 스크립트 25 + legacy(inference v1~v3,
  filter_compare) 6 + upload/기타 유틸 9 = 40개 삭제.
```

**중요**: `data_prep/` 서브폴더 (apriltag / inference / ablation / filter_compare /
debug) 는 루트의 `canonical_filters.py` 와 `visualize_inference.py` 를 sibling
import 못 하므로 상단에 `sys.path.insert(0, os.path.join(__file__, ".."))` 로
루트를 추가해야 한다. 이미 이동된 파일들은 패치 완료 (2026-04-12).

## 환경 구분

| 환경 | 디렉토리/파일 | 실행 방법 |
|------|--------------|-----------|
| **Isaac Sim** | `data_prep/isaac_sim/` | Isaac Sim 내장 Python으로 standalone 실행 |
| **PyTorch** | `data_prep/*.py`, `self_training/`, `train_dope.sh`, `dope/run_dope_live.py` | `conda activate pallet-pose` (실시간 추론은 추가로 `pip install pyrealsense2`) |

## 설정 파일 참조

- 모든 하이퍼파라미터: `config/default.yaml`
- Self-Training 전용: `config/stage3_selftrain.yaml`
