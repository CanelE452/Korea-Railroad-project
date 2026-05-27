# main_rec.py (실행 즉시 자동 녹화 시작 버전: 첫 프레임 생성 시 VideoWriter 자동 초기화)

import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import math
from datetime import datetime

from calib.config import (
    SAMPLE_STRIDE, Z_INLIER_THRESH, MIN_POINTS,
    EMA_ALPHA_OFFSET, EMA_ALPHA_YAW, EMA_ALPHA_WIDTH,
    COLOR_ALERT, COLOR_STATUS_OK, COLOR_META, COLOR_BOX, COLOR_CNT, COLOR_CENTER,
    COLOR_YAW, COLOR_OFFSET, COLOR_WIDTH,
    USE_6D_MODE, MODEL_PATH_6D, DOPE_INPUT_HEIGHT, DOPE_PEAK_THRESHOLD, DOPE_PEAK_SIGMA,
)
from calib.hud import draw_panel
from calib.perception import Perception
from calib.geometry import robust_points_from_mask_or_roi, fit_plane_yaw_from_points
from calib.fsm import CalibrationFSM
from calib.control import can_init, can_close
from calib.utils import fmt_deg, fmt_m
from calib.pose6d_adapter import keypoints9_to_align_vars
from ui.diagram import draw_fsm_diagram_panel


def setup_video_writer_filename():
    rec_dir = "./rec"
    os.makedirs(rec_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(rec_dir, f"forklift_recording_{ts}.mp4")


def realsense_check_or_exit():
    ctx = rs.context()
    devs = ctx.query_devices()
    if devs.size() == 0:
        print("❌ RealSense 카메라를 찾을 수 없습니다. USB3 포트/케이블을 확인하세요.")
        sys.exit(1)
    print(f"✅ RealSense 장치 {devs.size()}대 연결됨.")
    for d in devs:
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        sn   = d.get_info(rs.camera_info.serial_number) if d.supports(rs.camera_info.serial_number) else "N/A"
        print(f"  - {name} (S/N {sn})")


# === rel_yaw.py 로직 이식: gyro.Y 적분 기반 상대 yaw 추정기 ===
class RelYawEstimator:
    """
    rel_yaw 계산:
      - 자이로 y(rad/s) 적분 → yaw(deg)
      - 초기 기준 init_yaw = 0.0
      - [-180, +180]로 angle wrapping
    가속도는 roll/pitch 보정용 보조 신호로만 사용.
    """
    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.first = True
        self.last_ts_ms = None
        self.yaw_deg = 0.0     # 적분 누적 yaw(deg)
        self.init_yaw = None   # 기준 yaw(deg)

        # 보조: 가속도 기반 각도(roll/pitch)
        self.accel_angle_x = 0.0
        self.accel_angle_z = 0.0

        # 최근 rel_yaw 캐시(센서 드롭 시 유지)
        self.last_rel = 0.0

    def update_from_frames(self, accel, gyro, ts_ms: float) -> float:
        """
        accel: rs.motion_data (x,y,z) [m/s^2]
        gyro : rs.motion_data (x,y,z) [rad/s]
        ts_ms: 타임스탬프(ms)

        Returns:
            rel_yaw(deg) in [-180, +180]
        """
        if self.first:
            self.first = False
            self.last_ts_ms = ts_ms
            # 가속도 기반 각도(roll/pitch) 초기화
            self.accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            self.accel_angle_x = math.degrees(math.atan2(
                accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
            # 초기 yaw = 0 기준
            self.init_yaw = 0.0
            self.last_rel = 0.0
            return 0.0

        dt = max(0.0, (ts_ms - self.last_ts_ms) / 1000.0)
        self.last_ts_ms = ts_ms

        # gyro 적분 (rad/s * s -> rad -> deg)
        dangleY = math.degrees(gyro.y * dt)

        # accel 기반 roll/pitch(보조 정보) 계산 및 보정 (yaw에는 미적용)
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(math.atan2(
            accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
        totalgyroangleX = accel_angle_x
        totalgyroangleZ = accel_angle_z
        self.accel_angle_x = totalgyroangleX * self.alpha + accel_angle_x * (1 - self.alpha)
        self.accel_angle_z = totalgyroangleZ * self.alpha + accel_angle_z * (1 - self.alpha)

        # === 핵심: yaw는 "gyro.Y" 적분 ===
        self.yaw_deg += dangleY

        if self.init_yaw is None:
            self.init_yaw = self.yaw_deg

        rel_yaw = self.yaw_deg - self.init_yaw

        # [-180, +180]로 wrapping
        rel_yaw = (rel_yaw + 180.0) % 360.0 - 180.0

        self.last_rel = rel_yaw
        return rel_yaw


def main():
    realsense_check_or_exit()

    # CAN 초기화
    print("CAN 통신 초기화 중...")
    if can_init():
        print("✅ CAN 통신 초기화 성공(하트비트 자동)")
    else:
        print("❌ CAN 통신 초기화 실패: MOCK 모드로 동작합니다.")

    # 모듈
    perception = Perception()
    fsm = CalibrationFSM()

    # === 6D pose (DOPE) 추론기 ===
    dope_pose = None
    if USE_6D_MODE:
        try:
            from calib.dope_inference import DopePoseEstimator
            dope_pose = DopePoseEstimator(
                weights_path=MODEL_PATH_6D,
                input_height=DOPE_INPUT_HEIGHT,
                peak_threshold=DOPE_PEAK_THRESHOLD,
                peak_sigma=DOPE_PEAK_SIGMA,
            )
            print("✅ DOPE 6D pose estimator 초기화 완료")
        except Exception as e:
            print(f"❌ DOPE 6D pose estimator 초기화 실패: {e}")
            print("   기존 RGB-D RANSAC perception 모드로 fallback")
            dope_pose = None

    # === RealSense 파이프라인 구성 ===
    pipeline = rs.pipeline()
    cfg = rs.config()
    # RGB-D: 640x480 @ 15fps
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    # IMU: rel_yaw 계산에 필수
    cfg.enable_stream(rs.stream.accel)
    cfg.enable_stream(rs.stream.gyro)

    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    # === 녹화: 실행 즉시 자동 녹화 시작 ===
    video_filename = setup_video_writer_filename()
    video_writer = None
    recording = True              # ← 기본값: True (자동 녹화)
    print(f"📹 자동 녹화 대기: {video_filename} (첫 프레임 생성 시 시작)")
    print("📹 'r' 녹화 토글, 'ESC' 종료")

    # EMA 상태
    offset_smooth = None
    yaw_smooth    = None
    width_smooth  = None

    # === rel_yaw 추정기 ===
    rel_yaw_est = RelYawEstimator(alpha=0.98)
    rel_yaw = 0.0
    last_accel_md = None  # gyro와 짝지어 사용할 최근 accel 저장

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # --- IMU 프레임 수집/적분 ---
            accel_md = None
            gyro_md = None

            for f in frames:
                if hasattr(f, "is_motion_frame") and f.is_motion_frame():
                    md = f.as_motion_frame().get_motion_data()
                    st = f.get_profile().stream_type()

                    if st == rs.stream.accel:
                        accel_md = md
                        last_accel_md = md

                    elif st == rs.stream.gyro:
                        gyro_md = md
                        # === rel_yaw 업데이트: "gyro.Y + 최근 accel" 사용 ===
                        use_accel = accel_md if accel_md is not None else last_accel_md
                        if use_accel is not None:
                            rel_yaw = rel_yaw_est.update_from_frames(
                                accel=use_accel, gyro=gyro_md, ts_ms=f.get_timestamp()
                            )
                        # accel 데이터가 전혀 없다면, 이전 rel_yaw 유지

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            # 프레임 누락 시 안전 처리
            if not color_frame or not depth_frame:
                vis = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_panel(vis, [("프레임 없음", COLOR_ALERT)])
                diag = draw_fsm_diagram_panel(fsm, panel_size=(vis.shape[0], 720))
                show = cv2.hconcat([vis, diag])

                # ★ 자동 녹화: 첫 show가 생성되면 VideoWriter 자동 초기화
                if recording and video_writer is None:
                    h, w = show.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (w, h))
                    if video_writer.isOpened():
                        print(f"🔴 녹화 시작: {video_filename}")
                    else:
                        print("❌ VideoWriter 초기화 실패")
                        recording = False
                        video_writer = None

                if recording:
                    cv2.putText(show, "REC", (show.shape[1]-80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    if video_writer is not None:
                        video_writer.write(show)

                cv2.imshow("Forklift HUD + FSM", show)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('r'):
                    recording = not recording
                    print("🔴 녹화 시작" if recording else "⏹️ 녹화 중지")
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            # RealSense color intrinsic → DOPE PnP 용 3x3 K
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            camera_matrix = np.array([
                [color_intrin.fx, 0.0,             color_intrin.ppx],
                [0.0,             color_intrin.fy, color_intrin.ppy],
                [0.0,             0.0,             1.0],
            ], dtype=np.float64)
            vis = color_img.copy()
            H, W = vis.shape[:2]

            # 1) 감지
            det_ok, mask_bin, bbox_now = perception.infer_front(color_img)

            # 2) 3D 포인트/기하
            ok_plane = False
            yaw_deg = None
            ex = ey = ez = None
            width_now = None
            dist_euclid = None
            dist_z = None
            pts_in = None

            if det_ok and (mask_bin is not None) and (bbox_now is not None):
                ok_pts, pts_in = robust_points_from_mask_or_roi(
                    depth_frame=depth_frame, depth_intrin=depth_intrin,
                    mask_or_roi=mask_bin, stride=SAMPLE_STRIDE,
                    z_inlier_thresh=Z_INLIER_THRESH, min_points=MIN_POINTS
                )
                if ok_pts:
                    ex = float(np.median(pts_in[:, 0]))
                    ey = float(np.mean(pts_in[:, 1]))
                    ez = float(np.mean(pts_in[:, 2]))
                    cur_off = np.array([ex, ey, ez], dtype=np.float32)

                    if offset_smooth is None:
                        offset_smooth = cur_off.copy()
                    else:
                        offset_smooth = (1 - EMA_ALPHA_OFFSET) * offset_smooth + EMA_ALPHA_OFFSET * cur_off

                    ok_plane, yaw_deg, a, b = fit_plane_yaw_from_points(pts_in)
                    if ok_plane:
                        if yaw_smooth is None:
                            yaw_smooth = yaw_deg
                        else:
                            yaw_smooth = (1 - EMA_ALPHA_YAW) * yaw_smooth + EMA_ALPHA_YAW * yaw_deg

                    width_now = float(np.max(pts_in[:, 0]) - np.min(pts_in[:, 0]))
                    if width_smooth is None:
                        width_smooth = width_now
                    else:
                        width_smooth = (1 - EMA_ALPHA_WIDTH) * width_smooth + EMA_ALPHA_WIDTH * width_now

                    c3d_mean = np.mean(pts_in, axis=0).astype(np.float32)
                    dist_euclid = float(np.linalg.norm(c3d_mean))
                    dist_z = float(c3d_mean[2])

            # 2.5) 6D pose (DOPE) 추론 — det_ok 와 무관하게 시도하여
            #      YOLO segmentation 이 놓친 경우에도 keypoint 기반 검출 가능.
            psi_pallet_deg: float = None
            d_lateral_m: float = None
            d_forward_m: float = None
            kps9 = None
            if dope_pose is not None:
                try:
                    kps9 = dope_pose.infer_keypoints9(color_img)
                except Exception as e:
                    print(f"[DopePose] infer_keypoints9 error: {e}")
                    kps9 = None
                if kps9 is not None:
                    result6d = keypoints9_to_align_vars(kps9, camera_matrix, dist_coeffs=None)
                    if result6d is not None:
                        psi_pallet_deg, d_lateral_m, d_forward_m = result6d

            # 3) 시각화
            if bbox_now is not None:
                x1, y1, x2, y2 = bbox_now
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BOX, 2)
            if mask_bin is not None:
                mask_vis = (mask_bin * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, COLOR_CNT, 2)
            cv2.drawMarker(vis, (W // 2, H // 2), COLOR_CENTER, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # 4) HUD 텍스트
            lines = []
            lines.append(("detected" if det_ok else "no detection", COLOR_STATUS_OK if det_ok else COLOR_ALERT))
            lines.append((f"pts: {len(pts_in) if pts_in is not None else 0}", COLOR_META))

            # yaw(now/smooth): 평면기반
            if ok_plane and (yaw_smooth is not None) and (yaw_deg is not None):
                lines.append((f"yaw(now): {fmt_deg(yaw_deg)} deg", COLOR_YAW))
                lines.append((f"yaw_smooth: {fmt_deg(yaw_smooth)} deg", COLOR_YAW))
            else:
                lines.append(("yaw: N/A", COLOR_ALERT))

            # offset/width
            if ex is not None and (offset_smooth is not None):
                lines.append((f"offset_x(now): {fmt_m(ex)} m", COLOR_OFFSET))
                lines.append((f"offset_smooth: ({fmt_m(offset_smooth[0])}, {fmt_m(offset_smooth[1])}, {fmt_m(offset_smooth[2])})", COLOR_OFFSET))
            else:
                lines.append(("offset_smooth: N/A", COLOR_OFFSET))

            if (width_now is not None) and (width_smooth is not None):
                lines.append((f"pallet width(now): {width_now:.3f} m", COLOR_WIDTH))
                lines.append((f"pallet width_smooth: {width_smooth:.3f} m", COLOR_WIDTH))
            else:
                lines.append(("pallet width_smooth: N/A", COLOR_WIDTH))

            if (dist_euclid is not None) and (dist_z is not None):
                lines.append((f"distance: euclid {dist_euclid:.3f} m | z {dist_z:.3f} m", COLOR_META))
            else:
                lines.append(("distance: N/A", COLOR_META))

            # === HUD: rel_yaw(gyro-Y)만 표기 (IMU heading(Z-int) 제거) ===
            lines.append((f"rel_yaw(gyro-Y): {fmt_deg(rel_yaw)} deg", COLOR_META))
            lines.append((f"|rel_yaw|(gyro-Y): {abs(rel_yaw):6.2f} deg", COLOR_META))

            # === HUD: 6D pose (DOPE) ===
            if psi_pallet_deg is not None:
                lines.append((f"[6D] psi_pallet: {fmt_deg(psi_pallet_deg)} deg", COLOR_YAW))
                lines.append((f"[6D] d_lateral: {fmt_m(d_lateral_m)} m", COLOR_OFFSET))
                lines.append((f"[6D] d_forward: {fmt_m(d_forward_m)} m", COLOR_META))
            elif dope_pose is not None:
                lines.append(("[6D] no detection", COLOR_ALERT))

            # keypoint overlay (시각화)
            if kps9 is not None:
                for i, (u, v) in enumerate(kps9):
                    if not (np.isnan(u) or np.isnan(v)):
                        clr = (0, 0, 200) if i == 8 else (0, 200, 0)
                        cv2.circle(vis, (int(u), int(v)), 4 if i == 8 else 3, clr, 2)
                        cv2.putText(vis, f"{i}", (int(u) + 5, int(v) - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)

            # 5) FSM 구동 — rel_yaw + 6D pose 입력 전달
            #    6D 입력이 있으면 fsm 내부에서 yaw_smooth/offset_smooth 대신 우선 사용된다.
            guide_lines = fsm.step(
                det_ok=det_ok,
                detected_length=width_smooth if width_smooth is not None else None,
                dist_z=dist_z if dist_z is not None else None,
                yaw_smooth=yaw_smooth if yaw_smooth is not None else None,
                offset_smooth=offset_smooth if offset_smooth is not None else None,
                rel_yaw=rel_yaw,  # ★ OFFSET_CHECK 이후 회전 완료 조건 및 진행률에 사용
                psi_pallet_deg=psi_pallet_deg,
                d_lateral_m=d_lateral_m,
                d_forward_m=d_forward_m,
            )
            lines.extend(guide_lines)

            # 6) HUD 렌더
            draw_panel(
                vis, lines, origin=(18, 22), pad=(7, 6),
                line_h=18, font_scale=0.45, thickness=1, cmd_status=fsm.cmd_status
            )

            # 7) FSM 다이어그램 패널 합성
            diag = draw_fsm_diagram_panel(fsm, panel_size=(vis.shape[0], 900))
            if diag.shape[0] != vis.shape[0]:
                diag = cv2.resize(diag, (diag.shape[1], vis.shape[0]), interpolation=cv2.INTER_AREA)
            show = cv2.hconcat([vis, diag])

            # ★ 자동 녹화: 첫 show가 준비되면 즉시 VideoWriter 초기화
            if recording and video_writer is None:
                h, w = show.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (w, h))
                if video_writer.isOpened():
                    print(f"🔴 녹화 시작: {video_filename}")
                else:
                    print("❌ VideoWriter 초기화 실패")
                    recording = False
                    video_writer = None

            # 8) 녹화 표시/쓰기
            if recording:
                cv2.putText(show, "REC", (show.shape[1]-80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if video_writer is not None:
                    video_writer.write(show)

            # 9) 표시 & 키 입력
            cv2.imshow("Forklift HUD + FSM", show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                recording = not recording
                print("🔴 녹화 시작" if recording else "⏹️ 녹화 중지")

    finally:
        print("프로그램 종료 처리 중...")
        try:
            if video_writer is not None:
                video_writer.release()
                print(f"✅ 비디오 저장 완료: {video_filename}")
        except Exception:
            pass
        try:
            can_close()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ 리소스 정리 완료")


if __name__ == "__main__":
    main()
