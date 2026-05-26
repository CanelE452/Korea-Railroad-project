# Runtime Dataflow — 프레임 한 장 처리 흐름

`python depth_cam/main_rec.py` 실행 시 매 프레임마다 호출되는 함수 체인. 6D 통합 후 변경되는 부분은 **★** 표시.

## 초기화 (단 1회)

```
realsense_check_or_exit()                           # 카메라 연결 확인
can_init()                                          # Kvaser CAN (실패시 MOCK)
  → start_heartbeat()                               # 200ms 주기 background thread

perception = Perception()                           # ★ DOPE 모델 로드
fsm = CalibrationFSM()                              # 상태기계 + 서브머신 초기화

pipeline.start(cfg)                                 # RGB 640x480x15 + Depth 640x480x15 + IMU(accel, gyro)
align = rs.align(rs.stream.color)                   # depth를 color에 정렬

video_writer = None  (첫 프레임에서 lazy init)
RelYawEstimator(alpha=0.98)                         # IMU 적분기
```

## 매 프레임 루프

```
frames = pipeline.wait_for_frames()
  ↓
[IMU 처리]
  for f in frames:
    if motion_frame:
      st = stream_type
      if st == accel: 기억
      if st == gyro:  rel_yaw = RelYawEstimator.update_from_frames(accel, gyro, ts)
  ↓
aligned = align.process(frames)
color_frame, depth_frame = aligned.get_color/depth_frame()
color_img = np.asanyarray(color_frame.get_data())
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # ★ 6D에서는 K 행렬로 변환 사용
  ↓
★ [Perception + Geometry] — 통합 시 여기가 통째로 바뀜
  ─────────────────────────────────────────────────
  기존:
    det_ok, mask_bin, bbox_now = perception.infer_front(color_img)
    if det_ok:
      ok_pts, pts_in = robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask_bin, ...)
      if ok_pts:
        ex = median(pts_in[:,0])
        ey = mean(pts_in[:,1])
        ez = mean(pts_in[:,2])
        offset_smooth = EMA(offset_smooth, (ex,ey,ez), α=0.4)
        ok_plane, yaw_deg, a, b = fit_plane_yaw_from_points(pts_in)
        if ok_plane: yaw_smooth = EMA(yaw_smooth, yaw_deg, α=0.4)
        width_now = max(pts_in[:,0]) - min(pts_in[:,0])
        width_smooth = EMA(width_smooth, width_now, α=0.4)
        c3d_mean = mean(pts_in, axis=0)
        dist_euclid = ||c3d_mean||
        dist_z      = c3d_mean[2]
  ─────────────────────────────────────────────────
  신규:
    pose = perception.infer(color_img, depth_frame=depth_frame)
    # pose = {ok, R_pallet, t_pallet_cm, raw_points, proj_points, reason}
    det_ok = pose["ok"]
    if det_ok:
      dist_z, yaw_deg, offset_now_xyz, width_now = fsm_inputs_from_pose(
          pose["R_pallet"], pose["t_pallet_cm"],
          pallet_width_m=1.1, yaw_convention_offset_deg=YAW_CONVENTION_OFFSET_DEG)
      offset_smooth = EMA(offset_smooth, offset_now_xyz, α=EMA_ALPHA_OFFSET)
      yaw_smooth    = EMA_scalar(yaw_smooth, yaw_deg, α=EMA_ALPHA_YAW)
      width_smooth  = EMA_scalar(width_smooth, width_now, α=EMA_ALPHA_WIDTH)
  ─────────────────────────────────────────────────
  ↓
[시각화 — vis 위에 overlay]
  기존:
    if bbox_now: cv2.rectangle(vis, ...)
    if mask_bin: contour 그리기
    cv2.drawMarker(center)
  ─────────────────────────────────────────────────
  신규:
    if pose["proj_points"]: draw_cuboid(vis, pose["proj_points"], ...)  # run_live.py 함수 재사용
    cv2.drawMarker(center)
  ↓
[HUD 텍스트 라인 구성]
  lines = []
  lines.append(("detected"/"no detection", ...))
  lines.append((f"pts: {len(pts_in)}", ...))                              # 6D에서는 keypoint 수로 대체
  lines.append((f"yaw(now): {yaw_deg}", ...))
  lines.append((f"yaw_smooth: {yaw_smooth}", ...))
  lines.append((f"offset_x(now): {ex}", ...))
  lines.append((f"offset_smooth: ({...})", ...))
  lines.append((f"pallet width: {width_now}/{width_smooth}", ...))
  lines.append((f"distance: euclid {dist_euclid} | z {dist_z}", ...))
  lines.append((f"rel_yaw(gyro-Y): {rel_yaw}", ...))
  ↓
★ [FSM step] — 입력 형식 동일, 6D에서 채워준 값들이 들어감
  guide_lines = fsm.step(
      det_ok=det_ok,
      detected_length=width_smooth,
      dist_z=dist_z,
      yaw_smooth=yaw_smooth,
      offset_smooth=offset_smooth,    # (ox, oy, oz) tuple
      rel_yaw=rel_yaw,
  )
  lines.extend(guide_lines)
  ↓
[FSM 내부에서 CAN 송신]
  CommandExecutor.exec("FWD"/"BACK"/"ROT_LEFT"/"ROT_RIGHT"/"STOP")
    → calib.control.issue_command_*()  → Kvaser canlib.Channel.write(Frame(id, data, flags))
  ↓
[HUD 렌더링]
  draw_panel(vis, lines, cmd_status=fsm.cmd_status)
  diag = draw_fsm_diagram_panel(fsm, panel_size=(h, 900))    # ★ ui.diagram 없으면 stub 필요
  show = cv2.hconcat([vis, diag])
  ↓
[녹화]
  if video_writer is None: lazy init (mp4v, 15fps)
  if recording: video_writer.write(show)
  ↓
cv2.imshow("Forklift HUD + FSM", show)
key = cv2.waitKey(1)
  ESC: break / 'r': 녹화 토글
```

## 종료

```
video_writer.release()  # mp4 파일 닫기
can_close()             # CAN 정지 신호 송신 + 버스 해제 + heartbeat 종료
pipeline.stop()
cv2.destroyAllWindows()
```

---

## 호출 빈도와 지연

| 단계 | 빈도 | 추정 지연 |
|------|------|----------|
| RealSense `wait_for_frames` | 15Hz (color frame rate) | ~66ms (블로킹) |
| DOPE inference | 매 프레임 | ~50ms (RTX 3060, FP32) |
| FSM step | 매 프레임 | <1ms |
| HUD draw_panel | 매 프레임 | ~2ms |
| FSM diagram | 매 프레임 | (stub면 ~0ms) |
| CAN 송신 | 매 프레임 (중복 억제) | <1ms |
| Heartbeat (별도 thread) | 5Hz | 비차단 |

총 처리는 ~120ms/frame → 약 8 FPS 정도. RealSense를 30Hz로 올려도 추론이 병목.

## 데이터 단위 정리 (혼동 주의)

| 변수 | 단위 | 좌표계 |
|------|------|--------|
| `pts_in` (legacy) | m | 카메라 |
| DOPE `location` | **cm** | 카메라 |
| `t_pallet_cm` | cm | 카메라 |
| `t_pallet_m = t_pallet_cm/100` | m | 카메라 |
| `dist_z` | m | 카메라 Z 성분 |
| `offset_smooth` | m | 카메라 (x,y,z) tuple |
| `yaw_smooth` | **deg** | 부호: 오른쪽 회전이 + |
| `rel_yaw` | deg | IMU 적분, wrap to [-180, +180] |
| `width_smooth` | m | 카메라 X 범위 |
| FSM thresholds | YAW_TOL_DEG(deg), OFF_TOL_M(m), ALIGN_DIST_M(m) | — |

DOPE location의 cm 단위는 `Deep_Object_Pose` 학습 시 cuboid_dim을 cm로 넣은 결과 (`challenge/config/task.yaml`의 `pallet.width*100` 참조).
