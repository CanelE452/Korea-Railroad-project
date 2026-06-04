# 02. 모델 export (ONNX / TensorRT / INT8) + FPS

## 왜 TensorRT
```
- PyTorch FP32/half(=True): 작은 모델(3M)이라 batch1에선 호출 오버헤드가 병목
  → half 켜도 효과 거의 없음 (3080 기준 126→118fps).
- TensorRT engine: 커널 fusion으로 오버헤드 제거 → 3~5배 빨라짐 (작은 모델에 특히 효과).
- ★ engine은 하드웨어 종속 — 3080에서 만든 engine은 Jetson에서 못 씀.
  Jetson은 ONNX를 가져가 그 보드에서 engine 빌드해야 함.
```

## 산출물 (`pallet_jetson_deploy/models/`)
```
pallet_pose_cropaug_v2.pt    원본 (engine 빌드 소스)
pallet_pose_640.onnx         portable, Jetson 이식용 (imgsz 640)
pallet_pose_320.onnx         portable, 저해상도(더 빠름)
best_fp16_640.engine         3080 전용 (참고/로컬용, Jetson 미사용)
best_fp16_320.engine         3080 전용
best_int8_640.engine         3080 전용
```

## 벤치마크 (RTX 3080, forklift 100프레임, pad+predict)
```
모델                  FPS    ms     FP32 대비 keypoint 편차
──────────────────────────────────────────────────────────
PyTorch FP32 @640     121    8.27   기준
TRT FP16 @640         360    2.77   0.66px (무손실 수준)
TRT FP16 @320         604    1.66   -
TRT INT8 @640         385    2.60   1.64px (양자화 오차)
```
- FP16: 정확도 거의 무손실. INT8: 1.6px 편차로 약간 trade (작은 모델이라 640에선 FP16과 속도 비슷, INT8 이득은 Jetson에서 더 큼).
- PnP(SQPnP) 0.012ms, RANSAC 포함 0.067ms — 무시 가능. 병목은 모델.

## PnP = SQPnP (속도 아닌 정확도 목적)
```
EPnP 단독        0.030 ms
SQPnP            0.012 ms  ← 더 빠르고 globally optimal
EPnP+RANSAC(현)  0.067 ms
```
- YOLO 추론 경로(`eval_ab_crop.solve_pnp`, `infer_fps.py`, `yolo_inference.py`)는 **SQPnP** 사용.
- 검증(holdout 42): EPnP+RANSAC 대비 reproj median 5.27→3.12px, ADD median 96.6→90.7mm 개선.

## Jetson에서 engine 빌드 + 추론
```bash
# 1) ONNX/pt → engine (Jetson에서, 그 보드 전용)
bash pallet_jetson_deploy/build_engine.sh 640 fp16   # 더 빠르게: 320

# 2) 추론 + FPS 측정 (카메라 없이 mp4로도 OK)
python pallet_jetson_deploy/infer_fps.py \
  --model models/pallet_pose_cropaug_v2.engine \
  --source data/forklift_sample.mp4 --cam-k data/cam_K.txt --imgsz 640 --conf 0.5 --show
# 종료 시 inference-only FPS / full-pipeline FPS / ms breakdown / 검출·PnP율 요약 출력
```
- engine 실행 시 `LD_LIBRARY_PATH`에 `<env>/lib/python3.10/site-packages/torch/lib` + `tensorrt_libs` 필요.
- infer_fps.py 는 standalone(repo 의존 0), 검증: 기존 파이프라인과 검출/PnP 100% 일치.

## imgsz와 Jetson
```
3080에선 imgsz 줄여도 FPS 그대로(오버헤드 병목). 그러나 Jetson은 연산 병목이라
imgsz 320 + INT8 가 큰 효과. Jetson Nano(구형)에서 실시간 노릴 땐 320/INT8 권장.
```

세부: `_docs/history/2026-06-02.md`.
