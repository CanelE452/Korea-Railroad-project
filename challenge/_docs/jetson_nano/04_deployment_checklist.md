# 04. Jetson 배포 체크리스트 + 남은 실물 작업

배포 가이드 폴더: `jetson_fsm_deploy/` (README.md, preflight.py, CAN_PORTING.md, build_engine.sh)

## 실행 위치
```
<repo>/25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam/
POSE_BACKEND=yolo GATE_PROFILE=real python main_rec.py
```

## 셋업 순서
```
1. repo 를 Jetson 에 git clone (DOPE 가중치+Deep_Object_Pose 커서 번들 대신 clone)
2. JetPack(torch/cuDNN/TensorRT 포함) + Jetson용 torch/torchvision wheel 설치
   pip install ultralytics opencv-python numpy pyrr
3. librealsense 소스 빌드 (pyrealsense2 ARM wheel 없음, BUILD_PYTHON_BINDINGS=ON)
4. CAN: Kvaser ARM 확인 또는 SocketCAN 포팅 (CAN_PORTING.md)
5. YOLO engine 빌드: bash pallet_jetson_deploy/build_engine.sh 640 fp16  (또는 320)
6. python jetson_fsm_deploy/preflight.py  ← 무엇이 빠졌는지 진단
7. POSE_BACKEND=yolo GATE_PROFILE=real python main_rec.py
```

## preflight.py 가 점검하는 것
```
import numpy/cv2/torch/torchvision/ultralytics/pyrr, CUDA, RealSense(장치),
CAN(mock 여부), YOLO/DOPE weights 존재, config 안전설정(GATE_PROFILE/extrinsic).
❌ = 실행 막힘, ⚠️ = 기능 제한.
```

## ⬜ 남은 실물/측정 작업 (Jetson·리프터·줄자 필요 — 코드로 끝낼 수 없음)
```
(1) pallet 실측 dims
    config.py PALLET_WIDTH/DEPTH/HEIGHT_M (현재 1.10×1.30×0.12 가정).
    PnP scale·삽입 기하의 기준 → 실제 팔레트 줄자 실측값으로.

(2) 카메라→포크 extrinsic 실측
    config.py CAM_TO_FORK_T(m), CAM_TO_FORK_RPY_DEG(deg). 현재 0(항등).
    카메라 장착 위치·각도 실측해 채워야 d_lateral/d_forward 가 포크 기준.

(3) CAN 실동작
    Kvaser canlib 가 Jetson ARM 에서 안 되면 SocketCAN 포팅(python-can).
    frame ID/data/bitrate/확장ID 는 리프터 약속이라 유지, 인터페이스만 교체.
    preflight 의 CAN 항목이 MOCK 이면 리프터 실제 안 움직임.

(4) librealsense / torch Jetson 빌드
    ARM wheel 부재 → 소스 빌드. JetPack 버전 맞춤.
```

## 실삽입 안전 수칙
```
□ GATE_PROFILE=real        나쁜 프레임 거부
□ depth fusion 동작         RealSense depth 유효 (거리=depth 기반, monocular 단독 ~0.5m 오차)
□ 첫 시험은 빈 공간/저속    사람·구조물 없는 데서 정렬 먼저
□ 비상정지 손에            STOP_SEC=1.2 인터록은 보조일 뿐
□ 리프터 띄운 채 1차 테스트  (바퀴/포크 공중) 후 실삽입
```

## 성능 기대치
```
환경              YOLO26n pose (pad+SQPnP)         권장 설정
─────────────────────────────────────────────────────────────
RTX 3080(개발)    ~360 fps (TRT FP16 640)         참고용
Jetson Nano 구형  ~10~15 fps (TRT FP16/INT8 320)  imgsz320 + INT8 + nvpmodel -m0 + jetson_clocks
Jetson Orin Nano  30+ fps                          FP16 640 무난
DOPE(VGG-19)      Nano엔 무거움                    → YOLO backend 권장
FSM               snapshot(STOP_SEC=1.2)라 낮은 FPS 허용
```
세부: `_docs/history/2026-06-04.md`.
