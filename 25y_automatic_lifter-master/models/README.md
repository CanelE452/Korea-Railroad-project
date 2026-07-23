# models — 리프터 6D pose 가중치 폴더

`depth_cam/main_rec.py` 통합 루프(RealSense + DOPE 6D 추론 → FSM → CAN)가 로드하는
DOPE 가중치(.pth)를 여기에 둔다.

## 사용법

이 폴더에 `.pth` 파일을 하나 넣으면 `depth_cam/calib/config.py` 가 자동 탐색한다.
```
25y_automatic_lifter-master/models/<아무이름>.pth
```

가중치 경로 우선순위 (`config.py:_resolve_model_path_6d`):
1. 환경변수 `MODEL_PATH_6D` (지정 시 이 값 우선 — HF 등 다른 위치 사용 시 권장)
2. 이 폴더(`models/`) 내 유일한 `.pth`
3. 여러 개면 이름순 첫 `.pth`

## 참고

- `.pth` 파일 자체는 `.gitignore`(`*.pth`) 대상이라 커밋되지 않는다 (가중치는 로컬/외부 보관).
- YOLO backend(`POSE_BACKEND=yolo`) 가중치는 별도로 `MODEL_PATH_6D_YOLO` 환경변수 또는
  `pallet_jetson_deploy/models/` 참조.
