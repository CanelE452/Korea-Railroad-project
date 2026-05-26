# depth_cam — 6D Pose 통합 reference

이 폴더(`depth_cam/_docs/`)는 **25y_automatic_lifter-master/depth_cam**을 FoundationPose 안에 가져오면서, DOPE 6D pose 추론 결과를 기존 FSM 입력단(YOLO seg + RANSAC plane)에 어떻게 연결할지 한 곳에 정리한 문서다.

원본 repo: `C:/Users/minjae/Documents/github/25y_automatic_lifter-master/depth_cam`
복사 시점: 2026-05-20
변경 없이 그대로 복사된 상태에서, 6D pose로 perception을 교체하기 위한 reference 문서만 추가했다.

## 문서 인덱스

| 파일 | 내용 |
|------|------|
| [`module_inventory.md`](module_inventory.md) | 각 `.py` 파일이 무엇을 하고, 6D 통합에서 어떻게 다뤄지는지 (수정/유지/삭제) |
| [`integration_map.md`](integration_map.md) | 기존 perception 파이프라인 → 6D pose 매핑 (입출력 단위, 변환 식, 변경할 파일 위치) |
| [`coordinate_conventions.md`](coordinate_conventions.md) | RealSense / DOPE / FSM yaw·offset 부호 규칙과 검증 절차 |
| [`known_issues.md`](known_issues.md) | 가져온 직후 식별된 문제들 (ui.diagram 누락, fsm.py↔fsm/ 충돌 등) |
| [`runtime_dataflow.md`](runtime_dataflow.md) | 프레임 한 장이 들어왔을 때 모듈 간 호출 순서와 자료의 흐름 |

## 통합 한 줄 요약

`main_rec.py`의 **9~14단계** (`perception.infer_front` + `robust_points_from_mask_or_roi` + `fit_plane_yaw_from_points`)를 **DOPE 6D pose 추론**으로 교체한다. FSM(`calib/fsm/`), HUD(`calib/hud.py`), CAN(`calib/control.py`), IMU rel_yaw 적분은 **변경 없이 그대로 사용**한다.

## 환경 가정

- conda env: `pallet-pose` (FoundationPose 기존 환경)
- DOPE 모델: `challenge/weights/baseline_v8_A.pth`
- DOPE 코드: `Deep_Object_Pose/common/` (sys.path append)
- CAN: Kvaser CANlib (`canlib`) — 없으면 `[MOCK SEND]` 로 동작 (control.py 자동 fallback)
- IMU: RealSense D435i 내장 gyro/accel — `main_rec.py`의 `RelYawEstimator`가 그대로 처리
