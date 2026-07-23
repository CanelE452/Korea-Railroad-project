# student DLA-18 no-DCN 640x480 inference_v4 bundle

이 번들은 `inference_v4` 데이터셋을 포함한 student 모델 추론용 패키지입니다. 압축 해제 후 외부 데이터셋 경로를 따로 지정하지 않아도 바로 `inference_v4` 전체 9430장을 추론할 수 있습니다.

## 포함 파일

- `weights/model_final.pth`: validation sweep에서 선택된 best student checkpoint
- `datasets/inference_v4/`: 추론용 `inference_v4` 데이터셋
- `SMOKE-master/`: DLA-18 no-DCN, geometry-v2 추론 코드
- `SMOKE-master/configs/smoke_geometry_v2_real_v4_distill_dla18_nodcn_640x480.yaml`: student 설정
- `run_inference_included_v4.sh`: 포함된 `inference_v4`를 바로 추론하는 스크립트
- `run_inference.sh`: 외부 KITTI/SMOKE 스타일 데이터셋 추론 스크립트
- `results_meta/`: checkpoint ranking, FPS 비교, validation 추론 메타

## 바로 실행

```bash
unzip student_dla18_nodcn_640x480_fp32_with_inference_v4_20260723.zip
cd student_dla18_nodcn_640x480_fp32_with_inference_v4_20260723
./run_inference_included_v4.sh
```

기본 실행은 FP16을 사용하지 않는 FP32 추론입니다. 결과는 아래 경로에 저장됩니다.

```text
outputs/inference_v4_student_fp32_YYYYmmdd_HHMMSS/inference/kitti_test/data/
outputs/inference_v4_student_fp32_YYYYmmdd_HHMMSS/inference/kitti_test/test_predictions_student_dla18_nodcn_False.zip
```

## 출력 경로 지정

```bash
./run_inference_included_v4.sh /path/to/output_dir
```

## 외부 데이터셋 추론

```bash
./run_inference.sh /path/to/dataset_root /path/to/output_dir
```

외부 데이터셋은 아래 구조를 기대합니다.

```text
dataset_root/
  testing/
    image_2/
    calib/
    ImageSets/
      test.txt
```

## 자주 쓰는 옵션

GPU가 아닌 CPU에서 실행:

```bash
DEVICE=cpu ./run_inference_included_v4.sh
```

FP16 autocast로 실행:

```bash
FP16=1 ./run_inference_included_v4.sh
```

KITTI metric 평가까지 실행하려면 `label_2`가 있는 split에서:

```bash
RUN_KITTI_EVAL=1 ./run_inference.sh /path/to/dataset_root
```

## 모델 정보

- 입력 크기: `640x480`
- 모델: SMOKE `geometry_v2`
- backbone: `DLA-18-NODCN`
- head channels: `128`
- checkpoint: `model_final.pth`
- validation selection score: Car 3D moderate AP `91.5485`
- inference_v4 FP32 prediction files: `9430`
- inference_v4 FP32 non-empty prediction files: `7864`

주의: 이 번들은 데이터셋을 포함하므로 압축파일 크기가 큽니다. PyTorch autocast FP16은 현재 GPU benchmark에서 FP32보다 빠르지 않았고, 기본값은 `FP16=0`입니다.
