# DOPE 모델 구조 및 학습 설정

## 모델 구조

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

## 학습 설정

```
파라미터           값              비고
──────────────────────────────────────────────────────────────────────
optimizer          Adam            DOPE 기본
learning_rate      1e-4 (pretrain) finetune 시 5e-5
                   5e-5 (finetune)
weight_decay       0               DOPE 기본 (no weight decay)
batch_size         4               GPU 메모리 제약
epochs             60 (pretrain)   finetune은 용도에 따라 조정
input_size         448 × 448       정사각형 리사이즈
output_size        50 × 50         belief map 해상도
sigma              4.0             belief map Gaussian std
loss               MSE             belief + affinity (상세: training_loss.md)
```

> **sigma 설정**: belief map GT 생성 시 각 keypoint에 sigma=4.0인 Gaussian을 찍는다.
> 50×50 output에서 ~25×25 픽셀 영역(전체의 25%)을 커버하여 충분한 gradient signal을 제공한다.
> sigma=0.5는 거의 1픽셀 peak만 생성하여 gradient vanishing 문제를 일으킨다.

## Annotation 형식

NDDS 호환 포맷. 각 이미지에 대해 JSON 파일 자동 생성:

```
필드                         내용
──────────────────────────────────────────────────────────────────────
projected_cuboid             8개 꼭짓점 2D 좌표
projected_cuboid_centroid    중심 2D 좌표
cuboid                       8개 꼭짓점 3D 좌표
pose_transform               4×4 포즈 행렬
```

데이터 로더: `Deep_Object_Pose/common/utils.py` CleanVisiiDopeLoader
- `{i:06d}.png` + `{i:06d}.json` 쌍으로 읽음

## Keypoint 추출 (Inference)

DOPE 공식 sub-pixel 방식: Gaussian filter + NMS + 11×11 weighted average

## 평가 메트릭

`scripts/data_prep/eval/evaluate_on_val.py`로 종합 평가:

```
메트릭              설명                              용도
──────────────────────────────────────────────────────────────────────
PCK@3/5/10px       keypoint 위치 정확도              Synthetic val screening
PnP 성공률         EPnP+RANSAC 성공 비율             기본 감지 성능
Reproj error       PnP 재투영 오차 (px)              Pose 품질
Volume Ratio       3D cuboid 부피 비 (1.0=perfect)   Pose 정밀도
ADD                3D 모델 포인트 평균 거리           Real test 최종 평가
5cm-5°             병진 5cm + 회전 5° 이내 비율       Real test 최종 평가
```

수학적 정의 → `_docs/method/formulation.md` Section 10 참조

## 코드 위치

```
파일                                    역할
──────────────────────────────────────────────────────────────────────
Deep_Object_Pose/train/train.py         학습 루프
Deep_Object_Pose/train/geo_loss.py      Geometric loss (soft-argmax + BPnP)
Deep_Object_Pose/common/models.py       DopeNetwork 모델 정의
Deep_Object_Pose/common/utils.py        데이터 로더 (CleanVisiiDopeLoader)
scripts/train_dope.sh                   학습 실행 스크립트
config/default.yaml                     설정 중앙 관리
```
