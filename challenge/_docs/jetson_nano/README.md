# Jetson Nano 배포 — 팔레트 6D pose → 리프터 포크 정렬/삽입

YOLO26-pose 기반 팔레트 6D 추정을 Jetson에 올려 FSM(정렬 상태머신)으로 리프터가
포크를 끼우게 하기까지의 전 과정 문서. (작업 기간: 2026-06-02 ~ 06-04)

## 목표
```
forklift 영상/카메라 → 팔레트 6D pose 추론 → FSM 정렬(ψ, 횡, 거리) → CAN → 리프터 포크 삽입
                       (DOPE 또는 YOLO, depth fusion 으로 거리 보정)
```

## 문서 목차
```
01_yolo_pose_models.md      YOLO26 pose 모델 (crop-aug truncation), 데이터 전략, 모델 비교
02_export_tensorrt_fps.md   ONNX / TensorRT(FP16,INT8) export + FPS 벤치마크
03_fsm_integration.md       DOPE/YOLO backend, depth fusion, real-mode gate, extrinsic
04_deployment_checklist.md  Jetson 셋업 / preflight / CAN 포팅 / 남은 실물작업
```

## 핵심 요약 (한눈에)
```
항목                     상태        비고
──────────────────────────────────────────────────────────────────────
YOLO truncation 강건     ✅ 완료     crop-aug(측면 잘림)로 잘린 팔레트 검출 100%/PnP 83%
TensorRT 최적화          ✅ 완료     FP16 3080 360fps, ONNX는 Jetson 이식용
FSM에 YOLO backend       ✅ 완료     POSE_BACKEND=dope|yolo, 같은 경로 (R_fix 주의)
거리 정확도(depth fusion) ✅ 완료     monocular ~0.5m 오차 → RealSense depth로 보정
real-mode 안전 게이트    ✅ 완료     GATE_PROFILE=real
Jetson 배포 패키지       ✅ 완료     jetson_fsm_deploy/ (README+preflight+CAN가이드)
──────────────────────────────────────────────────────────────────────
pallet 실측 dims         ⬜ 필요     줄자 실측 → config
CAM_TO_FORK extrinsic    ⬜ 필요     카메라 장착위치 실측 → config
CAN Kvaser→SocketCAN     ⬜ 필요     Jetson 실보드 검증
librealsense/torch 빌드  ⬜ 필요     Jetson에서 소스 빌드
```

## 관련 코드/산출물 경로
```
모델/배포(YOLO 추론+FPS)  : pallet_jetson_deploy/
Jetson FSM 배포 가이드     : jetson_fsm_deploy/
FSM + 양 backend 코드     : 25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam/
  - calib/yolo_inference.py     YoloPoseEstimator
  - calib/dope_inference.py     DopePoseEstimator
  - calib/pose6d_adapter.py     depth_scale_correct, apply_cam_to_fork, pose6d_to_align_vars
  - calib/config.py             POSE_BACKEND, GATE_PROFILE, CAM_TO_FORK_*
crop 증식 스크립트         : challenge/scripts/gen_truncation_crops.py
추론 FPS 스크립트          : pallet_jetson_deploy/infer_fps.py
```
세부 작업일지: `_docs/history/2026-06-02.md`, `_docs/history/2026-06-04.md`
