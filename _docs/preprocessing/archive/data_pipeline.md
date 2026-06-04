# 합성 데이터 파이프라인

## 1. 개요

Isaac Sim 4.5.0 + Omniverse Replicator로 DOPE 학습용 합성 데이터를 생성하고, 검증 후 학습에 투입하는 파이프라인.

```
USD 모델 준비 → Isaac Sim 렌더링 → Annotation 생성 → 검증 → 병합 → 학습
```

## 2. 데이터 생성

### 실행

```bash
# 단일 배치 (Isaac Sim 내장 Python으로 실행)
scripts/data_prep/gen_replicator_data.py --num_frames 200 --output_dir data/pallet/training_data/batch_01

# 전체 배치 (200프레임 × 13배치 = 2,600프레임)
bash scripts/data_prep/generate_all.sh
```

### 출력 형식 (NDDS)

```
batch_01/
├── 000000.png          # RGB 이미지 (640×480)
├── 000000.json         # Annotation (keypoint 2D/3D, pose, bbox)
├── 000001.png
├── 000001.json
└── ...
```

### 제약사항

- ~2분/프레임 (렌더러 무관, bottleneck은 Isaac Sim step/orchestrator)
- 200프레임마다 Isaac Sim 재시작 필요 (메모리 누수)
- 시작 시간 ~5분 (extension 로딩 + DLL 초기화)
- `CUDA_MODULE_LOADING=LAZY`, `PYTHONUNBUFFERED=1` 설정 필수

## 3. Domain Randomization

매 프레임마다 아래 항목을 랜덤화:

```
항목            방법
──────────────────────────────────────────────────────────────────────────
팔레트 색상     프리셋 8색 (60%) + HSV 연속 (40%), diffuse_texture disconnect 필수
바닥/벽 텍스처  USD API로 Sdf.AssetPath 직접 변경 (Replicator API 불가)
적재물          팔레트 위 박스/원통 3~8개, 80% 프레임에 적용
Distractors     per-distractor material_omnipbr 생성 (stage.Traverse 금지)
조명            DomeLight 2000~3500 + RectLight 3개 (Main/Fill1/Fill2)
카메라          Mode A (60%, 리프터), Mode B (25%, 높은 시점), Mode C (15%, 바닥 레벨)
팔레트 배치     바닥 수평 고정 (tilt=0°), yaw만 자유 회전
```

## 4. 품질 검증

### 자동 검증

```bash
# Keypoint 기하학 검증 (edge 길이, 직교성, 회전행렬)
python scripts/data_prep/validate/verify_keypoints.py

# Annotation overlay 시각화
python scripts/data_prep/visualize/visualize_annotations.py --data_dir data/pallet/training_data/train
```

### 검증 항목

- Brightness: mean < 40 또는 mean > 240이면 skip
- Edge 길이: 0→1 ≈ 1.0m, 0→3 ≈ 0.15m, 0→4 ≈ 1.2m
- 회전행렬: det(R) ≈ 1, R·Rᵀ ≈ I

### 주의

- 어려운 케이스(팔레트-배경 유사 색상, 어두운 팔레트)는 **제거하지 말 것** — 모델 로버스트니스에 필수
- 생성된 이미지는 **로그가 아닌 실제 이미지를 개별 확인**해야 함

## 5. 데이터 병합

```bash
python scripts/data_prep/validate/merge_and_validate.py
```

배치별 출력을 `train/`과 `val/` 디렉토리로 병합하고, 파일명을 `{i:06d}` 형식으로 재번호 매김.

## 6. 최종 데이터 구성

```
data/pallet/training_data/
├── train/          # ~2,000 프레임 (10 배치 × 200)
│   ├── 000000.png
│   ├── 000000.json
│   └── ...
└── val/            # ~600 프레임 (3 배치 × 200)
    ├── 000000.png
    ├── 000000.json
    └── ...
```
