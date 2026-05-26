# Coordinate Conventions & Sign Rules

부호 한 자 잘못 잡으면 lifter가 반대로 돌면서 팔레트를 들이받는다. 통합 시 반드시 검증할 것.

## 1. 좌표계 정의

### RealSense camera (depth_cam 코드 전반 + DOPE 모두 사용)

OpenCV convention과 동일:
- **+X**: 카메라 시점에서 오른쪽
- **+Y**: 아래
- **+Z**: 앞 (카메라가 바라보는 방향)

### 팔레트 로컬 (DOPE / Y=UP convention)

`challenge/config/task.yaml: pallet.keypoint_convention = "y_up"`:
- **+X**: width (앞면 폭 ≈ 1.1m, 짧은 면)
- **+Y**: height (두께 ≈ 0.11m, 위쪽)
- **+Z**: depth/front (긴 면 ≈ 1.3m, 팔레트가 "정면을 바라보는" 방향)

`fork_target.py`의 `front_dir = R_pallet @ [0,0,1]` 도 +Z를 front로 정의.

### Y-UP 회전 행렬 정리 (참고)

`R_pallet`은 **팔레트 로컬 → 카메라 좌표** 변환. 즉 `R_pallet @ [0,0,1]` = 팔레트의 +Z 축이 카메라 좌표계에서 어디를 향하는가.

이상적 정렬(팔레트가 카메라를 정확히 마주봄)일 때:
- 팔레트 +Z(정면)는 카메라의 −Z 방향(카메라 쪽으로 다가옴)
- 따라서 `R_pallet[:,2] ≈ (0, 0, -1)`, 즉 `R_pallet @ [0,0,1] ≈ (0, 0, -1)`

→ `arctan2(R[0,2], R[2,2]) = arctan2(0, -1) = ±π`. 이상적 정렬 시 yaw가 0이 아니라 ±180°!

## 2. yaw 부호 — FSM 약속과 일치시키기

FSM(`calib/fsm/align.py: YAW_CHECK`)의 규칙:
```python
if yaw > +YAW_TOL_DEG:   ROT_RIGHT  처방  (팔레트가 오른쪽으로 회전된 상태를 풀어내기)
if yaw < -YAW_TOL_DEG:   ROT_LEFT   처방
if |yaw| ≤ YAW_TOL_DEG:  OFFSET_CHECK 진입
```

→ **`yaw_deg` 정의: 카메라가 팔레트를 정면에서 봤을 때 0, 팔레트가 시계방향(카메라 시점에서 오른쪽)으로 회전한 만큼 양수**.

### 변환식 (권장)

팔레트 정면을 가리키는 단위벡터를 `−R_pallet @ [0,0,1]`로 잡는다 (정면이 카메라 쪽으로 가는 방향). 그러면 이상적 정렬 시 `(0,0,+1)` → `arctan2(0, 1) = 0`.

```python
front_to_cam = -(R_pallet @ np.array([0.0, 0.0, 1.0]))   # 정면이 카메라로 향하는 방향
yaw_rad = np.arctan2(front_to_cam[0], front_to_cam[2])    # XZ 평면 사영
yaw_deg = np.degrees(yaw_rad)
# 부호 검증: 팔레트가 카메라 시점에서 오른쪽으로 돌아간 경우, 정면벡터가 카메라 +X 쪽으로 기움 → yaw > 0
```

대안 (`run_live.py` 와 동일하게 가려면):
```python
yaw_rad = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])
yaw_deg = np.degrees(yaw_rad)
# 이 정의에서는 정렬 시 yaw≈±180°. FSM에 넘기기 전 wrap_to_180(yaw_deg − 180) 또는 wrap_to_180(yaw_deg + 180) 필요.
```

→ **권장은 첫 번째 방법** (front_to_cam = -R_pallet @ [0,0,1]). 깔끔하고 wrap 보정 불필요.

### 잠재적 함정

DOPE가 학습 시 사용한 keypoint label의 정면 방향(+Z) 정의를 따라간다. `_docs/method/step1_synthetic_data.md` 또는 학습 데이터의 NDDS JSON을 확인. 만약 라벨에서 −Z가 정면이라면 위 식의 부호가 모두 뒤집힘.

검증: `data/outside/capture02` 등의 정렬된 프레임을 `run_live.py`로 돌려 화면의 yaw 표시값(현재 `run_live.py` 597줄)을 측정. 정렬된 프레임에서 0°가 나오면 두 번째 방법(+180 보정 필요), -180° 근방이 나오면 첫 번째 방법(부호 그대로).

## 3. offset 부호

```python
t_pallet_m = t_pallet_cm / 100.0
offset_x = t_pallet_m[0]   # 팔레트 centroid의 카메라 X 좌표
```

- 카메라 좌표 +X = 오른쪽. 팔레트가 카메라 화각의 오른쪽에 있으면 `offset_x > 0`.
- FSM(`OFFSET_CHECK`): `ox > +OFF_TOL_M` → `ALIGN_ROTATE_RIGHT` (오른쪽으로 90° 돌고 전진하고 다시 왼쪽으로 90° → "옆으로 평행이동"). ✅ 직관적 일치.

## 4. width 부호

길이이므로 항상 양수. detection이 ok이면 ~1.0~1.2m 범위에서 측정됨. 작아질 일은 거의 없음.

## 5. dist_z

```python
dist_z = t_pallet_m[2]   # 항상 양수 (카메라 앞에 있는 한)
```

FSM(`DIST_CHECK`):
- `dist_z > ALIGN_DIST_M + ALIGN_BAND_M` → `FWD`
- `dist_z < ALIGN_DIST_M - ALIGN_BAND_M` → `BACK`
- `|dist_z - ALIGN_DIST_M| ≤ ALIGN_BAND_M` → `YAW_CHECK`

`ALIGN_DIST_M = 2.20`, `ALIGN_BAND_M = 0.30` (config.py). 즉 1.90~2.50m에서 yaw 보정 시작.

## 6. rel_yaw (IMU)

6D pose와 별개. RealSense gyro Y 적분(`main_rec.py: RelYawEstimator`):
- 첫 프레임에 0으로 초기화
- 그 후로 누적, `wrap_to_180(rel - init)`
- FSM이 `ALIGN_ROTATE_*` 체인에서 ±90° 회전 종료 판단에 사용

부호는 IMU 자체 정의에 의존 (gyro Y가 +면 어느 쪽 회전인지). `main_rec.py` 코드는 검증된 상태로 보임 (`_docs/history` 참조). **6D 통합에서는 건드릴 필요 없음**.

---

## 검증 절차 (smoke)

### Step 1. 정렬된 프레임에서 yaw_raw 측정

```bash
conda activate pallet-pose
cd C:/Users/minjae/Documents/github/FoundationPose
python challenge/scripts/run_live.py --seq data/outside/capture02  # 정렬 잘된 시퀀스
```

화면 yaw 표시가 정렬된 시점에서 어떤 값인지 기록:
- 0° 근방 → DOPE 출력 그대로 +X/+Z atan2 사용. yaw_deg = arctan2(R[0,2], R[2,2]) 그대로.
- ±180° 근방 → 변환식 적용 필요. `front_to_cam = -R_pallet @ [0,0,1]` 권장.

### Step 2. 카메라 정면에서 오른쪽으로 살짝 회전된 프레임 측정

`run_live.py`로 그런 프레임을 찾아 yaw 부호 확인:
- yaw > 0 ✓ → FSM 그대로 호환
- yaw < 0 ✗ → 식에 부호 반전 추가

### Step 3. offset_x 부호 확인

팔레트가 카메라 화각의 오른쪽에 있는 프레임에서:
- `t_pallet_cm[0] > 0` ✓
- `t_pallet_cm[0] < 0` ✗ → 카메라 캘리브레이션 또는 좌표축 정의를 재확인

### Step 4. 전체 시퀀스로 end-to-end smoke

```bash
python depth_cam/main_rec.py  # 6D 교체 후
```

CAN 없는 환경에서는 `[MOCK SEND]` 로그로 lifter 명령이 어떻게 떨어지는지 확인.
- 정렬된 시퀀스 → FSM이 `[YAW_CHECK→OFFSET_CHECK→READY_TO_DONE]` 순서로 진행하면 OK.
- 회전된 시퀀스 → `ROT_LEFT/RIGHT`가 정확한 방향으로 떨어지는지 확인.
