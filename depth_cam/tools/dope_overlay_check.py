"""DOPE 추론 결과를 RGB에 overlay → 학습 R 컨벤션 시각 확인.

cuboid wireframe (8 corner) + front face (강조) + centroid + yaw 화살표.

사용:
    python depth_cam/tools/dope_overlay_check.py \\
        --seq data/outside/capturepallet02 \\
        --frame_idx 0 \\
        --out depth_cam/tools/overlay.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DEPTH_CAM_DIR)

from calib.config import PALLET_WIDTH_M, PALLET_DEPTH_M
from calib.geometry import fsm_inputs_from_pose
from calib.perception import Perception


def load_frame(seq_dir, frame_idx):
    rgb_paths = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    rgb_path = rgb_paths[min(frame_idx, len(rgb_paths) - 1)]
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

    depth_frame = None
    if depth_paths:
        d = cv2.imread(depth_paths[min(frame_idx, len(depth_paths) - 1)], cv2.IMREAD_UNCHANGED)
        if d is not None and d.dtype == np.uint16:
            sys.path.insert(0, os.path.join(_DEPTH_CAM_DIR, "..", "challenge", "scripts"))
            from run_live_io import NpDepthFrame
            depth_frame = NpDepthFrame(d)

    K_path = os.path.join(seq_dir, "cam_K.txt")
    K = np.loadtxt(K_path).reshape(3, 3) if os.path.isfile(K_path) else None
    return img, depth_frame, K


def draw_overlay(img, pose, K, proc_scale):
    """cuboid wireframe + front face 강조 + centroid + yaw 화살표."""
    proj = pose.get("proj_points")
    raw  = pose.get("raw_points")
    if proj is None or raw is None:
        cv2.putText(img, "NO POSE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return img

    # proc_scale 역변환
    def to_orig(pt):
        if pt is None:
            return None
        return (int(pt[0] / proc_scale), int(pt[1] / proc_scale))

    pts8 = [to_orig(proj[i]) for i in range(8)]
    centroid = to_orig(raw[8]) if raw[8] is not None else None

    # back face (4-7) — 회색
    back_edges = [(4,5), (5,6), (6,7), (7,4)]
    for a, b in back_edges:
        if pts8[a] and pts8[b]:
            cv2.line(img, pts8[a], pts8[b], (140, 140, 140), 2, cv2.LINE_AA)

    # vertical edges (0-4, 1-5, ...) — 하늘색
    for i in range(4):
        if pts8[i] and pts8[i + 4]:
            cv2.line(img, pts8[i], pts8[i + 4], (200, 200, 0), 2, cv2.LINE_AA)

    # front face (0-3) — 초록 강조
    front_edges = [(0,1), (1,2), (2,3), (3,0)]
    for a, b in front_edges:
        if pts8[a] and pts8[b]:
            cv2.line(img, pts8[a], pts8[b], (0, 255, 0), 4, cv2.LINE_AA)

    # corner index 표시
    for i, p in enumerate(pts8):
        if p:
            cv2.circle(img, p, 5, (0, 200, 0), -1)
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # centroid
    if centroid:
        cv2.circle(img, centroid, 8, (0, 0, 255), -1)
        cv2.circle(img, centroid, 14, (0, 0, 255), 2)

    # yaw 화살표 (cuboid local +Z 방향을 카메라 좌표에 그림)
    R = pose["R_pallet"]
    t_cm = pose["t_pallet_cm"]
    f3d_origin = np.array([0, 0, 0], dtype=np.float64)
    f3d_axis   = np.array([0, 0, 50], dtype=np.float64)  # +Z 50cm
    pts_proj, _ = cv2.projectPoints(
        np.array([f3d_origin, f3d_axis], dtype=np.float64),
        cv2.Rodrigues(R)[0], t_cm.reshape(3, 1).astype(np.float64),
        K, np.zeros((4, 1), dtype=np.float64)
    )
    pts_orig_arrow = pts_proj.reshape(-1, 2).astype(int)
    cv2.arrowedLine(img, tuple(pts_orig_arrow[0]), tuple(pts_orig_arrow[1]),
                    (0, 255, 255), 4, tipLength=0.2)
    cv2.putText(img, "+Z(front)", tuple(pts_orig_arrow[1] + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 라벨
    offset, dist_z, yaw, _ = fsm_inputs_from_pose(R, t_cm, PALLET_DEPTH_M, PALLET_WIDTH_M)
    text_lines = [
        f"yaw_smooth: {yaw:+.1f} deg  (FSM uses this)",
        f"dist_z (front center): {dist_z:.3f} m",
        f"offset_x: {offset[0]:+.3f} m",
        f"centroid (cm): ({t_cm[0]:+.1f}, {t_cm[1]:+.1f}, {t_cm[2]:+.1f})",
        f"R[:,2] (+Z axis in cam): ({R[0,2]:+.3f}, {R[1,2]:+.3f}, {R[2,2]:+.3f})",
        f"  R[2,2] > 0 -> pallet +Z aligns with camera forward (back-facing?)",
        f"  R[2,2] < 0 -> pallet +Z opposes camera forward (front-facing)",
    ]
    y0 = 25
    for line in text_lines:
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        y0 += 22

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/outside/capturepallet02")
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--out", default="depth_cam/tools/overlay.png")
    args = ap.parse_args()

    repo_root = os.path.dirname(_DEPTH_CAM_DIR)
    seq_dir = args.seq if os.path.isabs(args.seq) else os.path.join(repo_root, args.seq)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)

    perception = Perception()
    img, depth_frame, K = load_frame(seq_dir, args.frame_idx)
    pose = perception.infer(img, depth_frame=depth_frame, K=K)

    if not pose["ok"]:
        print(f"[FAIL] {pose['reason']}")
        return

    overlay = draw_overlay(img.copy(), pose, K, pose["proc_scale"])
    cv2.imwrite(out_path, overlay)
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
