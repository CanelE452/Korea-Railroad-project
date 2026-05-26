"""
Real-time Pallet 6D Pose Estimation with DOPE + Intel RealSense D435i

시각화:
  - 회색 점 + 수치: belief map raw 피크 (항상 표시)
  - 초록 점: 감지된 코너 키포인트 (0~7)
  - 빨간 점: 감지된 중심점 (8)
  - 노란 화살표: Yaw axis + 각도

Belief map 클릭: 팔레트 코너/중심을 클릭하면 자동 threshold 튜닝

Keys: q=종료  s=저장  b=belief 토글  r=auto-tune 리셋
"""

import os
import sys
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from scipy.ndimage import gaussian_filter

sys.path.append("Deep_Object_Pose/common")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from pyrr import Quaternion, matrix33

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ── 공유 상태 (메인루프 ↔ 마우스 콜백) ──────────────────────────────────────
class State:
    vertex2 = None
    sigma = 3
    cell_w = 0
    cell_h = 0
    grid_scale = 1.0
    clicks = []           # [(channel, raw_value)]
    auto_threshold = None  # 클릭으로 설정된 threshold


def on_belief_click(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN or state.vertex2 is None:
        return
    if state.cell_w == 0 or state.cell_h == 0:
        return

    gx = int(x / state.grid_scale)
    gy = int(y / state.grid_scale)
    col = gx // state.cell_w
    row = gy // state.cell_h
    ch = row * 3 + col
    if not (0 <= ch < 9):
        return

    # 클릭 위치 → belief map 좌표 (÷8)
    bx = (gx - col * state.cell_w) // 8
    by = (gy - row * state.cell_h) // 8

    raw = state.vertex2[ch].cpu().numpy()
    sm = gaussian_filter(raw, sigma=state.sigma)
    bh, bw = sm.shape
    bx, by = np.clip(bx, 0, bw - 1), np.clip(by, 0, bh - 1)

    # 클릭 주변 5x5 영역 최대값
    r = 2
    patch = sm[max(0, by-r):min(bh, by+r+1), max(0, bx-r):min(bw, bx+r+1)]
    val = float(patch.max())

    state.clicks.append((ch, val))
    name = f"corner{ch}" if ch < 8 else "center"
    print(f"\n[Click] {name}: peak={val:.6f}")

    min_val = min(v for _, v in state.clicks)
    state.auto_threshold = min_val * 1.0
    chs = sorted(set(c for c, _ in state.clicks))
    print(f"[Auto-tune] min={min_val:.6f} → threshold≈{state.auto_threshold:.6f}")
    print(f"  Clicked channels: {chs}")


# ── 유틸리티 ─────────────────────────────────────────────────────────────────
def sample_depth(depth_frame, x, y, radius=3):
    if depth_frame is None:
        return None
    fw, fh = depth_frame.get_width(), depth_frame.get_height()
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < fw and 0 <= ny < fh:
                d = depth_frame.get_distance(nx, ny)
                if d > 0.05:
                    vals.append(d)
    return float(np.median(vals)) if vals else None


def scale_K(K, s):
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def run_forward(net, img_rgb):
    """DOPE forward pass 1회 → (vertex2, aff)"""
    t = _transform(img_rgb)
    with torch.no_grad():
        out, seg = net(Variable(t).cuda().unsqueeze(0))
    return out[-1][0], seg[-1][0]


def extract_peaks(vertex2, sigma=3):
    """각 채널(9개)에서 raw peak 위치+값 추출"""
    peaks = []
    for ch in range(vertex2.size(0)):
        raw = vertex2[ch].cpu().numpy()
        sm = gaussian_filter(raw, sigma=sigma)
        val = float(sm.max())
        idx = np.unravel_index(sm.argmax(), sm.shape)
        peaks.append({'val': val, 'x': idx[1] * 8, 'y': idx[0] * 8})
    return peaks


def build_belief_grid(vertex2, img_bgr):
    """3x3 belief map grid 시각화"""
    upsampling = nn.UpsamplingNearest2d(scale_factor=8)
    h, w = img_bgr.shape[:2]
    cells = []
    for ch in range(min(vertex2.size(0), 9)):
        b = vertex2[ch].clone()
        bmin, bmax = float(b.min()), float(b.max())
        if bmax > bmin:
            b = (b - bmin) / (bmax - bmin)
        b_up = upsampling(b.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
        b_np = cv2.resize(b_up.cpu().numpy(), (w, h))
        hm = cv2.applyColorMap((b_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        bg = (img_bgr.astype(np.float32) * 0.4).astype(np.uint8)
        ov = cv2.addWeighted(bg, 1, hm, 0.6, 0)
        lbl = f"c{ch}" if ch < 8 else "ctr"
        cv2.putText(ov, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(ov)
    while len(cells) < 9:
        cells.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[r*3:r*3+3]) for r in range(3)]
    return np.vstack(rows)


def _noop(x):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",   default="data/pallet/ndds3_pallet.pth")
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--no_depth",  action="store_true")
    ap.add_argument("--cam_id",    type=int, default=0)
    ap.add_argument("--width",     type=int, default=1280)
    ap.add_argument("--height",    type=int, default=720)
    ap.add_argument("--dim", nargs=3, type=float, default=[80.0, 14.4, 120.0],
                    metavar=("W", "H", "L"), help="팔레트 치수 [cm]")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out_dir",   default="debug/live_dope")
    args = ap.parse_args()

    use_depth = args.realsense and not args.no_depth
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 카메라 ────────────────────────────────────────────────────────────────
    pipeline = align = cap = None
    if args.realsense:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        rc = rs.config()
        rc.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
        if use_depth:
            rc.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
            align = rs.align(rs.stream.color)
        profile = pipeline.start(rc)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        print(f"[RealSense] fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    else:
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("카메라 열기 실패"); return
        K = np.array([[900, 0, args.width/2],
                      [0, 900, args.height/2],
                      [0, 0, 1]], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ── DOPE 모델 ─────────────────────────────────────────────────────────────
    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = args.threshold; softmax = 1000; thresh_angle = 0.5
        thresh_map = 0.05; sigma = 3; thresh_points = 0.05
    cfg = Cfg()

    print(f"[모델] {args.weights} 로딩...")
    model = ModelData(name="pallet", net_path=args.weights, parallel=True)
    model.load_net_model()
    pnp_solver = CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(args.dim))
    pnp_solver.set_dist_coeffs(dist_coeffs)

    # ── 슬라이더 ──────────────────────────────────────────────────────────────
    ctrl = "Controls"
    cv2.namedWindow(ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl, 500, 250)
    cv2.createTrackbar("threshold(x1000)", ctrl, int(args.threshold * 1000), 300, _noop)
    cv2.createTrackbar("thresh_map(x1000)", ctrl, 50, 200, _noop)
    cv2.createTrackbar("thresh_pts(x1000)", ctrl, 50, 200, _noop)
    cv2.createTrackbar("thresh_ang(x100)", ctrl, 50, 100, _noop)
    cv2.createTrackbar("sigma", ctrl, 3, 10, _noop)

    # ── Belief map 클릭 콜백 ──────────────────────────────────────────────────
    state = State()
    cv2.namedWindow("DOPE Belief Maps")
    cv2.setMouseCallback("DOPE Belief Maps", on_belief_click, state)

    print("[조작] q=종료  s=저장  b=belief토글  r=auto-tune리셋")
    print("[Belief] 팔레트 코너/중심을 클릭하면 자동 threshold 튜닝")

    frame_idx = 0
    show_belief = True

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    while True:
        depth_frame = None
        if args.realsense:
            frames = pipeline.wait_for_frames()
            if use_depth and align:
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame() or None
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
        else:
            ret, img = cap.read()
            if not ret:
                break

        # ── Auto-tune 적용 ────────────────────────────────────────────────
        if state.auto_threshold is not None:
            t = state.auto_threshold
            cv2.setTrackbarPos("threshold(x1000)", ctrl, max(0, min(300, int(t * 1000))))
            cv2.setTrackbarPos("thresh_map(x1000)", ctrl, max(0, min(200, int(t * 500))))
            cv2.setTrackbarPos("thresh_pts(x1000)", ctrl, max(0, min(200, int(t * 500))))
            state.auto_threshold = None

        # ── 슬라이더 읽기 ─────────────────────────────────────────────────
        cfg.threshold     = cv2.getTrackbarPos("threshold(x1000)", ctrl) / 1000.0
        cfg.thresh_map    = cv2.getTrackbarPos("thresh_map(x1000)", ctrl) / 1000.0
        cfg.thresh_points = cv2.getTrackbarPos("thresh_pts(x1000)", ctrl) / 1000.0
        cfg.thresh_angle  = cv2.getTrackbarPos("thresh_ang(x100)", ctrl) / 100.0
        cfg.sigma         = max(1, cv2.getTrackbarPos("sigma", ctrl))

        # ── 전처리 ────────────────────────────────────────────────────────
        h, w = img.shape[:2]
        proc_scale = 400.0 / h
        new_w = int(w * proc_scale) & ~7
        img_small = cv2.resize(img, (new_w, 400))
        K_small = scale_K(K, proc_scale)
        pnp_solver.set_camera_intrinsic_matrix(K_small)
        img_rgb = img_small[..., ::-1].copy()

        # ── 네트워크 1회 실행 ─────────────────────────────────────────────
        vertex2, aff = run_forward(model.net, img_rgb)
        state.vertex2 = vertex2
        state.sigma = cfg.sigma

        peaks = extract_peaks(vertex2, sigma=cfg.sigma)
        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, cfg)
        except (IndexError, Exception):
            results = []

        img_draw = img_small.copy()
        detected = False

        # ── (1) Raw 피크 항상 표시: 회색 점 + 수치 ────────────────────────
        for i, pk in enumerate(peaks):
            px, py = int(pk['x']), int(pk['y'])
            if not (0 <= px < img_draw.shape[1] and 0 <= py < img_draw.shape[0]):
                continue
            above = pk['val'] > cfg.threshold
            if i == 8:
                clr = (0, 0, 200) if above else (80, 80, 120)
                cv2.circle(img_draw, (px, py), 4, clr, 1 if not above else 2)
            else:
                clr = (0, 200, 0) if above else (80, 120, 80)
                cv2.circle(img_draw, (px, py), 3, clr, 1 if not above else 2)
            cv2.putText(img_draw, f"{pk['val']:.4f}", (px + 6, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

        # ── (2) DOPE 정식 감지 결과 (밝은 점 + PnP yaw) ──────────────────
        for result in results:
            raw_points = result.get("raw_points")
            if raw_points is None:
                continue

            for i, pt in enumerate(raw_points):
                if pt is None:
                    continue
                px, py = int(pt[0]), int(pt[1])
                if i == 8:
                    cv2.circle(img_draw, (px, py), 7, (0, 0, 255), -1)
                else:
                    cv2.circle(img_draw, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(img_draw, str(i), (px + 4, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                detected = True

            # PnP 성공 → 3D yaw
            if result["location"] is not None:
                loc = result["location"]
                ori = result["quaternion"]
                q = Quaternion(ori)
                rot_mat = matrix33.create_from_quaternion(q)
                yaw_rad = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])
                yaw_deg = np.degrees(yaw_rad)

                rvec, _ = cv2.Rodrigues(rot_mat)
                tvec = np.array(loc, dtype=np.float32).reshape(3, 1)

                # Depth 보정
                cr = raw_points[8] if raw_points[8] is not None else None
                if depth_frame is not None and cr is not None:
                    d = sample_depth(depth_frame, cr[0]/proc_scale, cr[1]/proc_scale)
                    if d is not None:
                        dcm = d * 100.0
                        pz = tvec[2, 0]
                        if pz > 0 and abs(dcm - pz) / pz < 0.5:
                            tvec[2, 0] = dcm

                c3d = np.array(pnp_solver._cuboid3d.get_vertices()[8], dtype=np.float32)
                f3d = c3d + np.array([0, 0, 50], dtype=np.float32)
                pts, _ = cv2.projectPoints(np.array([c3d, f3d]), rvec, tvec, K_small, dist_coeffs)
                p1 = tuple(pts[0].ravel().astype(int))
                p2 = tuple(pts[1].ravel().astype(int))
                cv2.arrowedLine(img_draw, p1, p2, (0, 255, 255), 3, tipLength=0.15)
                cv2.putText(img_draw, f"Yaw: {yaw_deg:.1f}deg",
                            (p1[0], p1[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                proj_pts = result.get("projected_points")
                if proj_pts is not None:
                    for i, pt in enumerate(proj_pts):
                        if i < 8:
                            cv2.drawMarker(img_draw, (int(pt[0]), int(pt[1])),
                                           (0, 255, 0), cv2.MARKER_SQUARE, 12, 2)

        # ── 상태 표시 ─────────────────────────────────────────────────────
        if not detected:
            cv2.putText(img_draw, "NOT DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

        info = f"thr={cfg.threshold:.3f}"
        if state.clicks:
            info += f" [auto x{len(state.clicks)}]"
        cv2.putText(img_draw, info, (10, img_draw.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("DOPE Pallet Live", img_draw)

        # ── Belief grid ───────────────────────────────────────────────────
        if show_belief:
            grid = build_belief_grid(vertex2, img_small)
            state.cell_w = new_w
            state.cell_h = 400
            max_w = 900
            if grid.shape[1] > max_w:
                s = max_w / grid.shape[1]
                grid_show = cv2.resize(grid, None, fx=s, fy=s)
                state.grid_scale = s
            else:
                grid_show = grid
                state.grid_scale = 1.0
            cv2.imshow("DOPE Belief Maps", grid_show)

        # ── 디버그 출력 ───────────────────────────────────────────────────
        if frame_idx % 60 == 0:
            names = [f"c{i}" for i in range(8)] + ["ctr"]
            print(f"\n[F{frame_idx}] thr={cfg.threshold:.4f} map={cfg.thresh_map:.4f} "
                  f"pts={cfg.thresh_points:.4f} ang={cfg.thresh_angle:.2f}")
            for i, pk in enumerate(peaks):
                m = ">" if pk['val'] > cfg.threshold else " "
                print(f"  {names[i]}: {pk['val']:.6f} {m}")
            print(f"  detection: {len(results)} objects")

        # ── 키 입력 ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = os.path.join(args.out_dir, f"live_{frame_idx:04d}.png")
            cv2.imwrite(path, img_draw)
            print(f"[저장] {path}")
        elif key == ord('b'):
            show_belief = not show_belief
            if not show_belief:
                try: cv2.destroyWindow("DOPE Belief Maps")
                except: pass
            print(f"[Belief] {'ON' if show_belief else 'OFF'}")
        elif key == ord('r'):
            state.clicks = []
            state.auto_threshold = None
            print("[Reset] Auto-tune 초기화")

        frame_idx += 1

    if pipeline: pipeline.stop()
    if cap: cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
