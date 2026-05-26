# Module Inventory

depth_cam 안의 모든 파이썬 파일을 (a) 무엇을 하는 모듈인지, (b) 6D pose 통합 때 어떻게 다룰지로 정리한 표.

표기:
- **수정**: 6D 통합을 위해 코드 변경이 필요
- **유지**: 변경 없이 그대로 사용
- **삭제 후보**: 6D 통합 후 더 이상 호출되지 않으므로 정리 가능 (단, 비교/롤백을 위해 일단 보존)
- **누락**: import 되는데 파일이 없음 — 별도 처리 필요

---

## 진입점

### `main_rec.py`  [수정]

`pipeline.start` → 매 프레임 RGB-D + IMU → 1)perception 2)geometry 3)FSM step 4)HUD draw 5)녹화. 단일 진입점.

- **수정 포인트**:
  - `from calib.perception import Perception` → DOPE 추론기로 교체 (perception.py 자체를 수정하거나 신규 클래스로)
  - `from calib.geometry import robust_points_from_mask_or_roi, fit_plane_yaw_from_points` → 호출 제거. 대신 6D pose에서 `offset/yaw/width/dist_z`를 직접 산출하는 헬퍼 호출
  - `from ui.diagram import draw_fsm_diagram_panel` → **모듈이 존재하지 않음**. 빈 stub 패널로 대체 또는 import 제거 (`known_issues.md` 참조)
  - 244~280줄 (`# 1) 감지` ~ `# 2) 3D 포인트/기하`) 블록을 통째로 교체
- **유지**:
  - RealSense pipeline 구성, IMU(rel_yaw) 처리, CAN init/close, 녹화(VideoWriter), 키 핸들링, HUD 호출
  - `fsm.step(det_ok, detected_length, dist_z, yaw_smooth, offset_smooth, rel_yaw)` 인터페이스 — 입력만 6D 기반으로 채워주면 됨

---

## perception (교체 대상)

### `calib/perception.py`  [수정]

`Perception.infer_front(color_img) -> (det_ok, mask_bin, bbox_xyxy)` — YOLO seg 모델(`runs/segment/y11n_seg_finetune/weights/last.pt`)을 로드해 "front" 클래스 마스크/박스를 반환.

- **수정 방향**: 클래스를 그대로 두되 내부 구현을 DOPE 추론으로 교체. 반환 형식도 6D pose 정보를 담도록 확장:
  ```python
  Detection6D = {
      "ok": bool,           # gate + temporal confirm 통과 여부
      "R_pallet": (3,3),    # 팔레트 → 카메라 좌표 회전
      "t_pallet_m": (3,),   # 팔레트 centroid (카메라 좌표, m)
      "raw_points": [...]   # 9개 keypoint 픽셀좌표 (디버그/시각화용)
      "proj_points": [...]  # PnP cuboid 8 corner projection (HUD overlay용)
      "reason": str         # NOT DETECTED 시 어디서 떨어졌는지
  }
  ```
- **제거할 것**:
  - `from ultralytics import YOLO` 등 YOLO 의존성
  - `_resolve_front_idx`, TARGET_AR, `cv2.resize(masks[best_idx])`
- **참고**: `challenge/scripts/run_live.py` 가 동일한 DOPE 파이프라인을 이미 구현했으므로 그것을 모듈화해서 옮기면 됨

### `calib/geometry.py`  [수정 (또는 보존+신규 추가)]

- `compute_yaw_deg_from_plane(a, b)` / `fit_plane_yaw_from_points` — RANSAC plane fit으로 yaw 추정. **6D로 교체되면 호출 안 됨**. 일단 보존하되 호출 제거.
- `robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask, ...)` — depth 픽셀을 3D로 deproject. **6D로 교체되면 호출 안 됨**. 보존.
- `clamp_bbox` — 유틸. 보존.
- `compute_offset_and_width(pts_in)` — `pts_in[:,0]` 의 median/min-max로 offset_x/width 계산. **6D로 교체되면 호출 안 됨**. 보존.
- **추가할 함수 (6D 전용)**:
  ```python
  def fsm_inputs_from_pose(R_pallet, t_pallet_m, pallet_dim_m):
      """6D pose → FSM이 받는 (offset_smooth, yaw_deg, width_m, dist_z) 튜플."""
      ...
  ```
  → `integration_map.md` 참조

---

## FSM (변경 없음)

### `calib/fsm.py`  [삭제 후보 — 일단 보존]

Single-file 버전 legacy FSM. **Python import 규칙상 `calib/fsm/` 패키지가 우선**되므로 이 파일은 dead code다 (실행되지 않음). 옛 reference로 보존해도 무방하나 혼동 방지를 위해 `fsm_legacy.py.bak`로 rename하는 것이 더 안전.

### `calib/fsm/__init__.py`  [유지]

`from .top import CalibrationFSM` — 패키지의 public API.

### `calib/fsm/top.py`  [유지]

상위 FSM: `SEARCH → DETECTED → (ALIGN | RECOVER) → CHECK → DONE`.

- `step(det_ok, detected_length, dist_z, yaw_smooth, offset_smooth, rel_yaw)` — **6D 통합에서 이 시그니처를 그대로 채워주면 끝**.
- `cmd_status` property → HUD가 사용.

### `calib/fsm/align.py`  [유지 — 단, INSERT_FORWARD config 추가 권장]

상태 전이:
```
DIST_CHECK → YAW_CHECK → OFFSET_CHECK
   ├─ |offset|>tol → 90° 회전 체인 (RIGHT or LEFT)
   └─ |offset|≤tol & |yaw|≤tol → INSERT_FORWARD → READY_TO_DONE
```

- **INSERT_FORWARD가 이미 존재**: `_fwd_sec_for_insertion(dist_z) = (dist_z + PALLET_POCKET_M) / INSERT_FWD_MPS` 만큼 전진. 즉 정렬 완료 후 자동으로 포크가 팔레트 안으로 들어감.
- `PALLET_POCKET_M`, `INSERT_FWD_MPS`, `INS_FWD_MIN_SEC`, `INS_FWD_MAX_SEC`는 `cfg`에 없으면 default(`0.0/0.25/0.5/10.0`) 사용. **`calib/config.py`에 명시적으로 추가하는 게 안전**.

### `calib/fsm/recover.py`  [유지]

폭 부족(`detected_length < WIDTH_MIN_FULL`) 시 시야 확보용 회전. 현재 `WIDTH_MIN_FULL=0.00`이라 사실상 거의 진입 안 됨.

### `calib/fsm/commands.py`  [유지]

`CommandExecutor.exec("FWD"/"BACK"/"ROT_LEFT"/"ROT_RIGHT"/"STOP"/...)` — `calib.control.issue_command_*`로 위임. 중복 명령 송신 억제 + 마지막 회전 방향 기억.

### `calib/fsm/status_helper.py`  [유지]

`CommandStatus` 인스턴스 보유 + `start_timed`/`start_until`/`update_until_metric` 래퍼.

### `calib/fsm/utils.py`  [유지]

`within_band`, `Stabilizer` (N프레임 연속 같은 tag 확인), `SimpleTimer`.

---

## 보조 모듈 (전부 유지)

### `calib/config.py`  [수정 — 항목 일부 추가/조정]

- **수정**: `MODEL_PATH = 'runs/segment/.../last.pt'` → DOPE weight 절대경로
- **수정**: `WIDTH_MIN_FULL = 0.00` — 6D detection만 통과하면 사실상 OK
- **추가**: `PALLET_POCKET_M = 1.0`, `INSERT_FWD_MPS = 0.25` (align.py INSERT_FORWARD용)
- **추가**: DOPE 추론 threshold (challenge/config/task.yaml과 일치시킬 값들)
- **유지**: FSM thresholds (YAW_TOL_DEG, OFF_TOL_M, ALIGN_DIST_M, ...), FWD time fit 파라미터, HUD 색상

### `calib/control.py`  [유지]

Kvaser CAN init/close + heartbeat + movement/control frame 송신. **CANlib 없으면 `[MOCK SEND]` 출력만**. 변경 없음.

### `calib/command_status.py`  [유지]

HUD용 dataclass: code/label/mode("timed"/"until")/진행률. 변경 없음.

### `calib/hud.py`  [유지]

PIL+OpenCV로 한글 텍스트 패널 + 진행바 렌더. 변경 없음.

### `calib/utils.py`  [유지]

`fmt_m/fmt_deg/dir_to_text/log_cmd` 포맷 헬퍼. 변경 없음.

### `calib/motion_models.py`  [유지]

`fwd_sec_from_offset_piecewise(offset_m)` — 가속→정속 piecewise 모델로 offset 보정용 FWD 시간 산출. 변경 없음.

### `logger.py`  [유지 / 무관]

`drive forward until wall` 데이터 수집 유틸 (FWD 시간 fit용 로그 수집). 6D 통합과 무관. 그대로 둠.

---

## 누락 모듈

### `ui/diagram.py`  [누락 — main_rec.py가 import]

`main_rec.py:23` `from ui.diagram import draw_fsm_diagram_panel`. 폴더 자체가 원본 repo에 없음. 두 가지 선택:

1. **stub 만들기**: `depth_cam/ui/diagram.py`를 신규 생성, `draw_fsm_diagram_panel(fsm, panel_size)`가 검은 패널만 반환하는 dummy로 (FSM 상태 텍스트만 표시).
2. **import/호출 제거**: main_rec.py에서 해당 import + `cv2.hconcat([vis, diag])` 부분 삭제.

→ `known_issues.md` 참조.

---

## 부속물 (코드 외)

| 폴더/파일 | 내용 | 6D 통합 시 |
|----------|------|-----------|
| `runs/` | YOLO 학습 산출물 (149MB, `last.pt`/`best.pt` 포함) | **삭제 가능** (DOPE로 교체되어 더 이상 안 씀). 일단 보존 후 `.gitignore` 추가 권장 |
| `docs/` | demo 이미지/비디오 | 유지 |
| `README.md` | 기존 YOLO 기반 사용 설명 | 6D 통합 후 별도 섹션 추가 또는 `_docs/`로 일원화 |
| `requirements.txt` | numpy/opencv/torch/canlib/ultralytics 등 | `ultralytics` 제거, `pyrr`(quaternion) 추가 |
