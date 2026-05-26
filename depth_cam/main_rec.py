# main_rec.py
# - 라이브 모드 (RealSense + IMU + CAN, 기본): 실시간 추론 + 녹화
# - 시퀀스 모드 (--seq DIR): 저장된 RGB+Depth 시퀀스 재생 (RealSense/CAN 없이 dry-run)

import sys
# Windows cp949 환경에서 이모지 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import cv2
import numpy as np
import os
import math
import time
from datetime import datetime

# RealSense 는 시퀀스 모드에서는 import 불필요 (라이브 모드에서만 사용)
try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    rs = None
    _HAS_RS = False

from calib.config import (
    EMA_ALPHA_OFFSET, EMA_ALPHA_YAW, EMA_ALPHA_WIDTH,
    COLOR_ALERT, COLOR_STATUS_OK, COLOR_META, COLOR_BOX, COLOR_CNT, COLOR_CENTER,
    COLOR_YAW, COLOR_OFFSET, COLOR_WIDTH,
    PALLET_DEPTH_M, PALLET_WIDTH_M,
    YAW_OFFSET_DEG, DEPTH_FRONT_SIGN,
)
from calib.hud import draw_panel
from calib.perception import Perception
from calib.geometry import fsm_inputs_from_pose, ema_scalar, ema_tuple
from calib.fsm import CalibrationFSM
from calib.control import can_init, can_close
from calib.utils import fmt_deg, fmt_m
from ui.diagram import draw_fsm_diagram_panel


def setup_video_writer_filename():
    rec_dir = "./rec"
    os.makedirs(rec_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(rec_dir, f"forklift_recording_{ts}.mp4")


def realsense_check_or_exit():
    if not _HAS_RS:
        print("❌ pyrealsense2 가 설치되지 않았습니다.")
        sys.exit(1)
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


# ─────────────────────────────────────────────────────────────────────────────
# 시퀀스 재생 모드 (RealSense 없이 dry-run)
# ─────────────────────────────────────────────────────────────────────────────
def _import_seq_helpers():
    """run_live_io.py 의 load_seq, NpDepthFrame 재사용."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_here)
    sys.path.insert(0, os.path.join(_repo_root, "challenge", "scripts"))
    from run_live_io import load_seq, NpDepthFrame
    return load_seq, NpDepthFrame


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


def parse_args():
    ap = argparse.ArgumentParser(description="forklift HUD + FSM (live / seq dry-run)")
    ap.add_argument("--seq", nargs="?",
                    const="data/outside/capturepallet03",
                    default="data/outside/capturepallet03",
                    help="저장 시퀀스 디렉토리 (RGB+Depth). default: capturepallet03. "
                         "--seq 값 없이도 default 사용 가능.")
    ap.add_argument("--seq_fps", type=float, default=10.0,
                    help="시퀀스 재생 속도 (0=프레임당 키 대기)")
    ap.add_argument("--seq_loop", action="store_true", help="시퀀스 끝나면 처음으로")
    ap.add_argument("--realsense", action="store_true",
                    help="라이브 모드 (RealSense + CAN). default 는 시퀀스 모드.")
    ap.add_argument("--no_can", action="store_true",
                    help="라이브 모드에서도 CAN 호출 skip (MOCK SEND 출력만)")
    return ap.parse_args()


def main():
    args = parse_args()
    # --realsense 명시 안 하면 시퀀스 모드 (기본)
    is_seq = not args.realsense

    # ── 입력 소스 + CAN 초기화 분기 ────────────────────────────────────────────
    seq_frames = None
    K_seq = None
    pipeline = None
    align = None
    can_active = False

    if is_seq:
        # 시퀀스 모드: RealSense/CAN 모두 skip
        load_seq, NpDepthFrame = _import_seq_helpers()
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        seq_dir = args.seq if os.path.isabs(args.seq) else os.path.join(repo_root, args.seq)
        seq_frames, K_seq = load_seq(seq_dir)
        if K_seq is None:
            # config 의 default intrinsic 사용
            from calib.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
            K_seq = np.array([[CAMERA_FX, 0, CAMERA_CX],
                              [0, CAMERA_FY, CAMERA_CY],
                              [0, 0, 1]], dtype=np.float64)
        print(f"[Seq] {seq_dir} — {len(seq_frames)} frames, fx={K_seq[0,0]:.1f}")
        print(f"[Seq] RealSense/CAN 사용 안 함 (dry-run)")
        # NpDepthFrame 클래스를 main 루프에서 사용
        _NpDepthFrame_cls = NpDepthFrame
    else:
        realsense_check_or_exit()

        if args.no_can:
            print("⚠️  CAN 호출 skip 모드 (--no_can): MOCK SEND 만 출력")
        else:
            print("CAN 통신 초기화 중...")
            if can_init():
                print("✅ CAN 통신 초기화 성공(하트비트 자동)")
                can_active = True
            else:
                print("⚠️  CAN 통신 초기화 실패: MOCK 모드로 동작합니다.")

        # === RealSense 파이프라인 구성 (D435i = IMU 있음, D435 = IMU 없음) ===
        pipeline = rs.pipeline()
        has_imu = False
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            cfg.enable_stream(rs.stream.accel)
            cfg.enable_stream(rs.stream.gyro)
            pipeline.start(cfg)
            has_imu = True
            print("[RS] IMU stream OK (D435i) — rel_yaw 적분 사용")
        except RuntimeError as e:
            print(f"[RS] IMU 없음 (D435?) — RGB+Depth 만 사용 (rel_yaw=0 고정)")
            print(f"     상세: {e!s}")
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            pipeline.start(cfg)
            print("⚠️  FSM 90° 회전 체인은 rel_yaw 의존 → IMU 없으면 ALIGN_ROTATE_*_90 단계에서 무한 회전. 정렬 시나리오 단순화 필요.")
        align = rs.align(rs.stream.color)
        _NpDepthFrame_cls = None  # 시퀀스 모드 전용

    # 모듈 (두 모드 공통)
    perception = Perception()
    fsm = CalibrationFSM()

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

    # === rel_yaw 추정기 (D435i 라이브 모드만 사용; D435/시퀀스는 rel_yaw=0 유지) ===
    use_imu = (not is_seq) and locals().get("has_imu", False)
    rel_yaw_est = RelYawEstimator(alpha=0.98) if use_imu else None
    rel_yaw = 0.0
    last_accel_md = None

    # 시퀀스 모드 상태
    seq_idx = 0
    seq_paused = False
    seq_wait_ms = max(1, int(1000.0 / max(0.1, args.seq_fps))) if is_seq else 1

    try:
        while True:
            # ── 프레임 획득: 시퀀스 모드 vs 라이브 RealSense ──────────────────────
            if is_seq:
                if seq_idx >= len(seq_frames):
                    if args.seq_loop:
                        seq_idx = 0
                    else:
                        print("[Seq] end of sequence")
                        break
                rgb_path, depth_path = seq_frames[seq_idx]
                color_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if color_img is None:
                    seq_idx += 1
                    continue
                depth_frame = None
                if depth_path is not None:
                    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if d is not None and d.dtype == np.uint16:
                        depth_frame = _NpDepthFrame_cls(d)
                # 시퀀스용 가상 color_frame (intrinsic 만 필요)
                color_frame = None  # 라이브와 분기되도록 None
                K_override = K_seq
                if not seq_paused:
                    seq_idx += 1
            else:
                frames = pipeline.wait_for_frames()

                # --- IMU 프레임 수집/적분 (D435i 전용; D435 면 use_imu=False 라 skip) ---
                if use_imu:
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
                                use_accel = accel_md if accel_md is not None else last_accel_md
                                if use_accel is not None:
                                    rel_yaw = rel_yaw_est.update_from_frames(
                                        accel=use_accel, gyro=gyro_md, ts_ms=f.get_timestamp()
                                    )

                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                color_img = np.asanyarray(color_frame.get_data()) if color_frame else None
                K_override = None

            # 프레임 누락 시 안전 처리
            if (color_img is None) or (not is_seq and not depth_frame):
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

            vis = color_img.copy()
            H, W = vis.shape[:2]

            # K (3x3) — 시퀀스 모드는 K_seq, 라이브는 RealSense intrinsic
            if K_override is not None:
                K = K_override
            else:
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                K = np.array([
                    [color_intrin.fx, 0.0,             color_intrin.ppx],
                    [0.0,             color_intrin.fy, color_intrin.ppy],
                    [0.0,             0.0,             1.0],
                ], dtype=np.float64)

            # ── 1) DOPE 6D pose 추론 (depth gate 위해 depth_frame 전달) ──
            pose = perception.infer(color_img, depth_frame=depth_frame, K=K)
            det_ok = bool(pose["ok"]) and bool(pose["confirmed"])

            # ── 2) FSM 입력 산출 (무게중심 → 앞면 중앙, yaw_smooth, width) ──
            yaw_deg = None
            offset_now = None
            width_now = None
            dist_z = None
            dist_euclid = None

            if pose["ok"]:
                offset_now, dist_z, yaw_deg, width_now = fsm_inputs_from_pose(
                    R_pallet=pose["R_pallet"],
                    t_pallet_cm=pose["t_pallet_cm"],
                    pallet_depth_m=PALLET_DEPTH_M,
                    pallet_width_m=PALLET_WIDTH_M,
                    yaw_offset_deg=YAW_OFFSET_DEG,
                    depth_front_sign=DEPTH_FRONT_SIGN,
                )
                # EMA smoothing — geometry.py 의 helper 사용
                offset_smooth = ema_tuple(offset_smooth, offset_now, EMA_ALPHA_OFFSET)
                yaw_smooth    = ema_scalar(yaw_smooth, yaw_deg, EMA_ALPHA_YAW)
                width_smooth  = ema_scalar(width_smooth, width_now, EMA_ALPHA_WIDTH)
                dist_euclid   = float(np.linalg.norm(np.array(offset_smooth)))

            # ── 3) 시각화: DOPE cuboid wireframe + front face 강조 + yaw 화살표 ──
            proj = pose.get("proj_points")
            raw  = pose.get("raw_points")
            scale_factor = pose.get("proc_scale", 1.0)
            # proj/raw 는 전처리된 (proc_scale 적용) 좌표 → 원본 해상도로 역변환
            if proj is not None:
                pts_orig = []
                for pt in proj[:8]:
                    if pt is None:
                        pts_orig.append(None)
                    else:
                        pts_orig.append((int(pt[0] / scale_factor), int(pt[1] / scale_factor)))
                # 색상: confirmed 면 진초록, pending 면 연한 청록
                c_front = (0, 255, 0)   if det_ok else (0, 200, 200)
                c_back  = (140, 140, 140)                              # back face 항상 회색
                c_vert  = (200, 200, 0) if det_ok else (160, 160, 100) # vertical 노랑/탁한노랑
                # back face (4 edges) 먼저
                for a, b in [(4,5),(5,6),(6,7),(7,4)]:
                    if pts_orig[a] and pts_orig[b]:
                        cv2.line(vis, pts_orig[a], pts_orig[b], c_back, 2, cv2.LINE_AA)
                # vertical 연결 4 edges
                for a, b in [(0,4),(1,5),(2,6),(3,7)]:
                    if pts_orig[a] and pts_orig[b]:
                        cv2.line(vis, pts_orig[a], pts_orig[b], c_vert, 2, cv2.LINE_AA)
                # front face 강조 (corners 0-3) — 두꺼운 초록
                for a, b in [(0,1),(1,2),(2,3),(3,0)]:
                    if pts_orig[a] and pts_orig[b]:
                        cv2.line(vis, pts_orig[a], pts_orig[b], c_front, 4, cv2.LINE_AA)
                # corner index 표시 (디버그용)
                for i, p in enumerate(pts_orig):
                    if p:
                        cv2.putText(vis, str(i), (p[0] + 4, p[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_front, 1)
            if raw is not None and raw[8] is not None:
                cx_p, cy_p = int(raw[8][0] / scale_factor), int(raw[8][1] / scale_factor)
                cv2.circle(vis, (cx_p, cy_p), 7, (0, 0, 255), -1)  # centroid keypoint
            # yaw 화살표 — 팔레트 +Z (front) 방향을 카메라 좌표에 그림
            if pose["ok"]:
                R_p = pose["R_pallet"]
                t_p_cm = pose["t_pallet_cm"]
                K_proc = pose["K_proc"]
                rvec, _ = cv2.Rodrigues(R_p)
                tvec = t_p_cm.reshape(3, 1).astype(np.float64)
                # local +Z 방향 50cm 화살표
                f3d = np.array([[0, 0, 0], [0, 0, 50]], dtype=np.float64)
                pts_proj, _ = cv2.projectPoints(
                    f3d, rvec, tvec, K_proc, np.zeros((4, 1), dtype=np.float64)
                )
                pts_proj = pts_proj.reshape(-1, 2)
                p1 = (int(pts_proj[0, 0] / scale_factor), int(pts_proj[0, 1] / scale_factor))
                p2 = (int(pts_proj[1, 0] / scale_factor), int(pts_proj[1, 1] / scale_factor))
                # 화면 범위 안에 있을 때만 그림
                if 0 <= p1[0] < W and 0 <= p1[1] < H and 0 <= p2[0] < W and 0 <= p2[1] < H:
                    cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 4, tipLength=0.2)
                    cv2.putText(vis, "+Z(front)", (p2[0] + 6, p2[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.drawMarker(vis, (W // 2, H // 2), COLOR_CENTER,
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # ── 4) HUD 텍스트 ──────────────────────────────────────────────
            lines = []
            if det_ok:
                lines.append(("detected (CONFIRMED)", COLOR_STATUS_OK))
            elif pose["ok"]:
                lines.append((f"detected (PENDING, reason={pose['reason']})", COLOR_META))
            else:
                lines.append((f"no detection ({pose['reason']})", COLOR_ALERT))

            if yaw_deg is not None and yaw_smooth is not None:
                lines.append((f"yaw(now): {fmt_deg(yaw_deg)} deg", COLOR_YAW))
                lines.append((f"yaw_smooth: {fmt_deg(yaw_smooth)} deg", COLOR_YAW))
            else:
                lines.append(("yaw: N/A", COLOR_ALERT))

            if offset_now is not None and offset_smooth is not None:
                lines.append((f"offset_x(now): {fmt_m(offset_now[0])} m", COLOR_OFFSET))
                lines.append((f"offset_smooth: ({fmt_m(offset_smooth[0])}, "
                              f"{fmt_m(offset_smooth[1])}, {fmt_m(offset_smooth[2])})", COLOR_OFFSET))
            else:
                lines.append(("offset_smooth: N/A", COLOR_OFFSET))

            if width_now is not None and width_smooth is not None:
                lines.append((f"pallet width(now): {width_now:.3f} m", COLOR_WIDTH))
                lines.append((f"pallet width_smooth: {width_smooth:.3f} m", COLOR_WIDTH))
            else:
                lines.append(("pallet width_smooth: N/A", COLOR_WIDTH))

            if dist_euclid is not None and dist_z is not None:
                lines.append((f"distance: euclid {dist_euclid:.3f} m | z {dist_z:.3f} m", COLOR_META))
            else:
                lines.append(("distance: N/A", COLOR_META))

            # === HUD: rel_yaw(gyro-Y)만 표기 (IMU heading(Z-int) 제거) ===
            lines.append((f"rel_yaw(gyro-Y): {fmt_deg(rel_yaw)} deg", COLOR_META))
            lines.append((f"|rel_yaw|(gyro-Y): {abs(rel_yaw):6.2f} deg", COLOR_META))

            # 5) FSM 구동 — rel_yaw를 반드시 전달
            guide_lines = fsm.step(
                det_ok=det_ok,
                detected_length=width_smooth if width_smooth is not None else None,
                dist_z=dist_z if dist_z is not None else None,
                yaw_smooth=yaw_smooth if yaw_smooth is not None else None,
                offset_smooth=offset_smooth if offset_smooth is not None else None,
                rel_yaw=rel_yaw,  # ★ OFFSET_CHECK 이후 회전 완료 조건 및 진행률에 사용
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
            wait_ms = seq_wait_ms if (is_seq and not seq_paused) else 1
            if is_seq and seq_paused:
                wait_ms = 0  # 키 대기
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                print("🔴 녹화 시작" if recording else "⏹️ 녹화 중지")
            elif is_seq and key == ord(' '):
                seq_paused = not seq_paused
                print(f"[Seq] {'PAUSED' if seq_paused else 'RESUMED'}")
            elif is_seq and key == ord('n'):
                seq_paused = True
                seq_idx = min(seq_idx + 1, len(seq_frames) - 1)
            elif is_seq and key == ord('p'):
                seq_paused = True
                seq_idx = max(seq_idx - 1, 0)

    finally:
        print("프로그램 종료 처리 중...")
        try:
            if video_writer is not None:
                video_writer.release()
                print(f"✅ 비디오 저장 완료: {video_filename}")
        except Exception:
            pass
        if can_active:
            try: can_close()
            except Exception: pass
        if pipeline is not None:
            try: pipeline.stop()
            except Exception: pass
        cv2.destroyAllWindows()
        print("✅ 리소스 정리 완료")


if __name__ == "__main__":
    main()
