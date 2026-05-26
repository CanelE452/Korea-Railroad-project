# Integration Map — 기존 perception → 6D pose

YOLO seg + RANSAC plane으로 산출하던 FSM 입력값을, DOPE 6D pose 한 번의 추론에서 모두 뽑아내는 매핑 표.

## FSM이 요구하는 입력 (변경 없음)

`calib/fsm/top.py::CalibrationFSM.step()`의 시그니처:

```python
step(det_ok:       bool,
     detected_length: Optional[float],    # 팔레트 전면 폭 (m)
     dist_z:        Optional[float],      # 팔레트까지의 z 거리 (m, 카메라 좌표)
     yaw_smooth:    Optional[float],      # 팔레트 정면 방향 yaw (deg, 부호: + → 오른쪽으로 돌아간 상태)
     offset_smooth: Optional[tuple],      # (offset_x, offset_y, offset_z) 카메라 좌표, m
     rel_yaw:       Optional[float])      # IMU gyro-Y 적분 (deg) — 회전 체인 종료 조건용
```

> 단위와 부호는 *반드시* 아래 매핑대로 채워야 함. 부호가 뒤집히면 lifter가 반대로 돌면서 잘못 정렬됨.

---

## 기존 (YOLO seg + RANSAC) — `main_rec.py:234~278`

```
color_img
   ↓
Perception.infer_front(color_img) → det_ok, mask_bin, bbox_now
   ↓
robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask_bin, ...)
   → pts_in: (N, 3) — 카메라 좌표 deproject 후 z-median inlier
   ↓
ex = median(pts_in[:, 0])      # offset_x
ey = mean(pts_in[:, 1])         # offset_y
ez = mean(pts_in[:, 2])         # offset_z
fit_plane_yaw_from_points(pts_in)
   → ok_plane, yaw_deg, a, b
width_now = max(pts_in[:, 0]) - min(pts_in[:, 0])
c3d_mean   = mean(pts_in, axis=0)
dist_z     = c3d_mean[2]
dist_euclid = ||c3d_mean||
   ↓
EMA smoothing (alpha=0.4)
   ↓ offset_smooth, yaw_smooth, width_smooth
fsm.step(det_ok, detected_length=width_smooth, dist_z=dist_z, yaw_smooth, offset_smooth, rel_yaw)
```

## 신규 (DOPE 6D pose)

```
color_img (+ depth_frame for gate)
   ↓
DOPEPerception.infer(color_img, depth_frame)
   → {ok, R_pallet, t_pallet_cm, raw_points, proj_points, reason, confirmed}
   ↓
fsm_inputs_from_pose(R_pallet, t_pallet_cm, pallet_dim_m, ema_state)
   → det_ok, detected_length, dist_z, yaw_smooth, offset_smooth
   ↓
fsm.step(det_ok, detected_length, dist_z, yaw_smooth, offset_smooth, rel_yaw)
```

---

## 변환 식 (DOPE 출력 → FSM 입력)

`t_pallet_cm` 은 DOPE `CuboidPNPSolver.location` 출력 (cm 단위, 카메라 좌표). 반드시 `/100`으로 m 변환.

### `det_ok`
```python
det_ok = pose.ok  # gate(min_kp/reproj/edge/depth_pnp_z) + temporal confirm 통과
```

### `dist_z`  (m)
```python
t_m = t_pallet_cm / 100.0
dist_z = float(t_m[2])
```

### `offset_smooth`  (m, tuple)
```python
offset_now = (float(t_m[0]), float(t_m[1]), float(t_m[2]))
offset_smooth = ema_update(offset_smooth, offset_now, alpha=EMA_ALPHA_OFFSET)
```

> 부호 검증: `t_m[0]` = 팔레트 centroid의 카메라 X 좌표. RealSense 카메라 좌표계는 +X = 카메라 시점에서 오른쪽. 팔레트가 카메라 화각의 오른쪽에 있으면 `offset_x > 0` → FSM의 `OFFSET_CHECK`가 `ALIGN_ROTATE_RIGHT` 분기 시작. ✅ FSM 부호 규칙과 일치.

### `yaw_smooth`  (deg)
```python
front_cam = R_pallet @ np.array([0.0, 0.0, 1.0])  # 팔레트 +Z(정면)을 카메라 좌표로
# 팔레트 keypoint convention: Y=UP, +Z=front
# XZ 평면에 사영 (수평면; Y=UP에서 Y 성분 무시)
yaw_rad = np.arctan2(front_cam[0], front_cam[2])
yaw_deg_raw = np.degrees(yaw_rad)
# 보정: 팔레트가 카메라를 마주볼 때(이상적 정면) yaw=0이 되도록
# → coordinate_conventions.md 의 "yaw 부호 검증" 절차로 실측 후 결정
yaw_deg = wrap_to_180(yaw_deg_raw - YAW_CONVENTION_OFFSET_DEG)
yaw_smooth = ema_update(yaw_smooth, yaw_deg, alpha=EMA_ALPHA_YAW)
```

> `YAW_CONVENTION_OFFSET_DEG`는 smoke 테스트로 정함. 팔레트가 카메라 정면에 정렬된 시퀀스(예: `data/outside/capture*` 중 정렬된 프레임)에서 `yaw_deg_raw`를 측정해 그 값을 offset으로 박는다.

> 부호 규칙: 팔레트가 카메라 시점에서 오른쪽으로 회전했을 때 → `front_cam[0] > 0` (정면 벡터의 +X 성분) → `yaw_deg > 0`. FSM의 `YAW_CHECK`는 `yaw > 0 → ROT_RIGHT` 처방. ✅

### `detected_length`  (m)
DOPE PnP 결과의 8 cuboid corner를 카메라 좌표로 변환 후 X 범위:
```python
corners_local_cm = pnp_solver._cuboid3d.get_vertices()  # 9개 (8 corner + center), cm
corners_local_m = np.array(corners_local_cm[:8]) / 100.0  # 8 corner만
corners_cam_m = (R_pallet @ corners_local_m.T).T + t_pallet_cm / 100.0  # (8, 3)
width_now = float(corners_cam_m[:, 0].max() - corners_cam_m[:, 0].min())
# detection이 ok이면 항상 약 1.1m (팔레트 width). EMA smoothing 권장.
detected_length = ema_update(width_smooth, width_now, alpha=EMA_ALPHA_WIDTH)
```

> 사실 `WIDTH_MIN_FULL = 0.00`이라 detection만 되면 0보다 큰 값 아무거나로 충분. 단 RECOVER 진입을 방지하려면 안정적으로 큰 값을 반환해야 함. PALLET_WIDTH_M (1.1m) 상수로 고정해도 됨.

### `rel_yaw`  (deg)
변경 없음. `main_rec.py`의 `RelYawEstimator.update_from_frames(accel, gyro, ts_ms)`가 RealSense IMU에서 그대로 산출. 6D pose와 무관.

---

## 코드 변경 위치 요약

### 1. `calib/perception.py` — 전면 교체

```python
class DOPEPerception:
    def __init__(self, weights_path, dim_cm, K, gates, temporal):
        # DOPE 모델 로드 + PnP solver 초기화
        ...

    def infer(self, color_img_bgr, depth_frame=None):
        """
        Returns dict:
          ok:           bool  (gate + temporal confirm 통과)
          R_pallet:     (3,3) float
          t_pallet_cm:  (3,)  float
          raw_points:   List[Optional[tuple]] of length 9
          proj_points:  List[Optional[tuple]] of length 9
          reason:       str
        """
        ...
```

### 2. `calib/geometry.py` — 함수 추가

```python
def fsm_inputs_from_pose(R_pallet, t_pallet_cm, pallet_width_m=1.1,
                         yaw_convention_offset_deg=0.0):
    """6D pose → (det_meta, dist_z, yaw_deg, offset_xyz_m, width_m)."""
    ...
```

### 3. `calib/config.py` — 항목 갱신

```python
# DOPE
MODEL_PATH = "C:/Users/minjae/Documents/github/FoundationPose/challenge/weights/baseline_v8_A.pth"
DOPE_CONFIG_PATH = "C:/Users/minjae/Documents/github/FoundationPose/challenge/config/task.yaml"
YAW_CONVENTION_OFFSET_DEG = 0.0   # smoke 측정 후 설정

# INSERT_FORWARD 명시화
PALLET_POCKET_M = 1.0
INSERT_FWD_MPS = 0.25
INS_FWD_MIN_SEC = 0.5
INS_FWD_MAX_SEC = 10.0
```

### 4. `main_rec.py` — perception 호출 블록 교체

수정 전 (244~278):
```python
det_ok, mask_bin, bbox_now = perception.infer_front(color_img)
if det_ok and (mask_bin is not None) and (bbox_now is not None):
    ok_pts, pts_in = robust_points_from_mask_or_roi(...)
    if ok_pts:
        ex = float(np.median(pts_in[:, 0])); ...
        ok_plane, yaw_deg, a, b = fit_plane_yaw_from_points(pts_in)
        ...
```

수정 후:
```python
pose = perception.infer(color_img, depth_frame=depth_frame)
det_ok = pose["ok"]
if det_ok:
    dist_z, yaw_deg, offset_now, width_now = fsm_inputs_from_pose(
        pose["R_pallet"], pose["t_pallet_cm"],
        pallet_width_m=1.1,
        yaw_convention_offset_deg=YAW_CONVENTION_OFFSET_DEG,
    )
    # EMA smoothing (기존과 동일)
    offset_smooth = ema(offset_smooth, offset_now, EMA_ALPHA_OFFSET)
    yaw_smooth    = ema_scalar(yaw_smooth, yaw_deg, EMA_ALPHA_YAW)
    width_smooth  = ema_scalar(width_smooth, width_now, EMA_ALPHA_WIDTH)
```

`mask_bin`/`bbox_now`/`pts_in` 변수를 참조하던 HUD/시각화 코드는 `pose["proj_points"]`로 대체.

---

## 통합 후 사라지는 호출

- `Perception.infer_front()`
- `robust_points_from_mask_or_roi()`
- `fit_plane_yaw_from_points()`, `compute_yaw_deg_from_plane()`
- `compute_offset_and_width()` (geometry.py)
- YOLO 관련 import (`ultralytics`)

이 함수들은 코드에 남겨두되 호출되지 않는다. 추후 정리 가능.
