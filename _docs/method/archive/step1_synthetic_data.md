# Step 1: Isaac Sim 합성 데이터로 DOPE 학습

## 3.1 워크플로우 개요

NVIDIA Developer 공식 자료에서 제시한 워크플로우를 기반으로 한다:

- **자료 1 (팔레트 감지 모델):** OpenUSD + Omniverse Replicator로 팔레트 합성 데이터를 생성하고 반복적으로 다양성을 늘려 감지 모델을 개선하는 워크플로우
  - URL: https://developer.nvidia.com/ko-kr/blog/developing-a-pallet-detection-model-using-openusd-and-synthetic-data/
  - GitHub: https://github.com/NVIDIA-AI-IOT/sdg_pallet_model

- **자료 2 (팔레트 잭 감지):** Isaac Sim + Replicator로 domain randomization을 단계적으로 적용하는 구체적 코드 예시
  - URL: https://developer.nvidia.com/ko-kr/blog/how-to-train-autonomous-mobile-robots-to-detect-warehouse-pallet-jacks-using-synthetic-data/
  - GitHub: https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow

## 3.2 시뮬레이터 환경

- **시뮬레이터:** NVIDIA Isaac Sim 4.5.0 (Omniverse 기반)
- **데이터 생성 API:** Omniverse Replicator
- **스크립트:** `scripts/data_prep/gen_replicator_data.py`

## 3.3 3D 모델

4종 USD 팔레트 모델 사용 (`data/pallet/models_usd/scene*.usd`).
각 모델의 ORIENTATION_OVERRIDES와 좌표계 정규화 → [keypoint_definition.md](../preprocessing/keypoint_definition.md) 참조.

## 3.4 Domain Randomization 요약

상세 구현 및 파라미터 → [data_pipeline.md](../preprocessing/data_pipeline.md) Section 3 참조.

주요 항목: 배경(창고/실내/야외), 팔레트 색상(프리셋+HSV), 조명(DomeLight+RectLight×3), Distractors(54종), 가림(80% 적재물), 카메라(3 Mode), 팔레트 배치(tilt=0°, yaw 자유).

## 3.5 DOPE 모델 구조

```
입력: RGB 이미지 (448 × 448)
      ↓
VGG-19 Backbone (ImageNet pre-trained)
→ feature map (50 × 50 × 512)
      ↓
Multi-Stage CNN Heads (Stage 1~6)
→ Belief maps: 9채널 (8 꼭짓점 + 1 centroid)
→ Affinity fields: 16채널
```

## 3.6 학습 설정

```yaml
Step1_Training:
  optimizer: Adam
  learning_rate: 1e-4
  weight_decay: 1e-4
  batch_size: 4
  epochs: 60
  lr_scheduler: StepLR (step=20, gamma=0.1)
  input_size: 448 × 448
  output_size: 50 × 50
  sigma: 4.0  # belief map Gaussian std (DOPE 공식 기본값)
  loss: MSE(predicted_belief, GT_belief) + MSE(predicted_affinity, GT_affinity)
  data: 5,000~15,000장 합성 이미지
```

> **sigma 설정**: belief map GT 생성 시 각 keypoint에 sigma=4.0인 Gaussian을 찍는다.
> 50×50 output에서 ~25×25 픽셀 영역(전체의 25%)을 커버하여 충분한 gradient signal을 제공한다.
> sigma=0.5는 거의 1픽셀 peak만 생성하여 gradient vanishing 문제를 일으킨다.

## 3.7 Annotation 형식

NDDS 호환 포맷. 각 이미지에 대해 JSON 파일 자동 생성:
- `projected_cuboid`: 8개 꼭짓점 2D 좌표
- `projected_cuboid_centroid`: 중심 2D 좌표
- `cuboid`: 8개 꼭짓점 3D 좌표
- `pose_transform`: 4×4 포즈 행렬

## 3.8 평가 메트릭

`scripts/data_prep/eval/evaluate_on_val.py`로 종합 평가. 주요 메트릭: PCK@3/5/10px, PnP 성공률, Reproj error, ADD, 5cm-5°.

수학적 정의 → [formulation.md](formulation.md) Section 10 참조.

> Keypoint 추출: DOPE 공식 sub-pixel 방식 (Gaussian filter + NMS + 11×11 weighted average)

## 3.9 반복적 개선 프로세스

```
Round 1: 1000장 생성 → 학습 → 실제 이미지 테스트 → 실패 분석
Round 2: 1000장 추가 (DR 보강) → 재학습 → 재테스트
Round 3: 1000장 추가 (distractors 강화) → 재학습
Round 4: 편향 확인 → 보완 데이터 추가
최종: 총 5,000~15,000장
```
