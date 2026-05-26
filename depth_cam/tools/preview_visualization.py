"""main_rec.py 의 새 시각화 (front face 강조 + yaw 화살표) 를 1 frame 으로 미리 보기."""
from __future__ import annotations
import glob, os, sys
import cv2
import numpy as np

_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DEPTH_CAM_DIR)
sys.path.insert(0, os.path.join(_DEPTH_CAM_DIR, "..", "challenge", "scripts"))

from calib.config import COLOR_CENTER
from calib.perception import Perception
from run_live_io import NpDepthFrame


def draw_full(vis, pose, det_ok):
    """main_rec.py 시각화 블록과 동일."""
    proj = pose.get("proj_points")
    raw  = pose.get("raw_points")
    scale_factor = pose.get("proc_scale", 1.0)
    H, W = vis.shape[:2]

    if proj is not None:
        pts_orig = []
        for pt in proj[:8]:
            if pt is None:
                pts_orig.append(None)
            else:
                pts_orig.append((int(pt[0] / scale_factor), int(pt[1] / scale_factor)))
        c_front = (0, 255, 0)   if det_ok else (0, 200, 200)
        c_back  = (140, 140, 140)
        c_vert  = (200, 200, 0) if det_ok else (160, 160, 100)
        for a, b in [(4,5),(5,6),(6,7),(7,4)]:
            if pts_orig[a] and pts_orig[b]:
                cv2.line(vis, pts_orig[a], pts_orig[b], c_back, 2, cv2.LINE_AA)
        for a, b in [(0,4),(1,5),(2,6),(3,7)]:
            if pts_orig[a] and pts_orig[b]:
                cv2.line(vis, pts_orig[a], pts_orig[b], c_vert, 2, cv2.LINE_AA)
        for a, b in [(0,1),(1,2),(2,3),(3,0)]:
            if pts_orig[a] and pts_orig[b]:
                cv2.line(vis, pts_orig[a], pts_orig[b], c_front, 4, cv2.LINE_AA)
        for i, p in enumerate(pts_orig):
            if p:
                cv2.putText(vis, str(i), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_front, 1)
    if raw is not None and raw[8] is not None:
        cx_p, cy_p = int(raw[8][0] / scale_factor), int(raw[8][1] / scale_factor)
        cv2.circle(vis, (cx_p, cy_p), 7, (0, 0, 255), -1)

    if pose["ok"]:
        R_p = pose["R_pallet"]
        t_p_cm = pose["t_pallet_cm"]
        K_proc = pose["K_proc"]
        rvec, _ = cv2.Rodrigues(R_p)
        tvec = t_p_cm.reshape(3, 1).astype(np.float64)
        f3d = np.array([[0, 0, 0], [0, 0, 50]], dtype=np.float64)
        pts_proj, _ = cv2.projectPoints(
            f3d, rvec, tvec, K_proc, np.zeros((4, 1), dtype=np.float64)
        )
        pts_proj = pts_proj.reshape(-1, 2)
        p1 = (int(pts_proj[0, 0] / scale_factor), int(pts_proj[0, 1] / scale_factor))
        p2 = (int(pts_proj[1, 0] / scale_factor), int(pts_proj[1, 1] / scale_factor))
        if 0 <= p1[0] < W and 0 <= p1[1] < H and 0 <= p2[0] < W and 0 <= p2[1] < H:
            cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 4, tipLength=0.2)
            cv2.putText(vis, "+Z(front)", (p2[0] + 6, p2[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.drawMarker(vis, (W // 2, H // 2), COLOR_CENTER,
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


def main():
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else "data/outside/capturepallet02"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    repo_root = os.path.dirname(_DEPTH_CAM_DIR)
    seq = seq_dir if os.path.isabs(seq_dir) else os.path.join(repo_root, seq_dir)

    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(seq, "depth", "*.png")))
    K_path = os.path.join(seq, "cam_K.txt")
    K = np.loadtxt(K_path).reshape(3, 3) if os.path.isfile(K_path) else None

    idx = min(frame_idx, len(rgb_paths) - 1)
    img = cv2.imread(rgb_paths[idx])
    df = None
    if idx < len(depth_paths):
        d = cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED)
        if d is not None and d.dtype == np.uint16:
            df = NpDepthFrame(d)

    perception = Perception()
    # Detection confirm 위해 같은 frame 2번 추론 (temporal_confirm=2)
    for _ in range(3):
        pose = perception.infer(img, depth_frame=df, K=K)
    det_ok = pose["ok"] and pose["confirmed"]

    vis = img.copy()
    draw_full(vis, pose, det_ok)

    out = os.path.join(_DEPTH_CAM_DIR, "tools", f"preview_{os.path.basename(seq)}_f{idx}.png")
    cv2.imwrite(out, vis)
    print(f"[OK] saved: {out}")
    print(f"  pose.ok={pose['ok']} confirmed={pose['confirmed']} reason={pose['reason']}")
    if pose["ok"]:
        print(f"  R[:,2]={pose['R_pallet'][:,2]}, t_cm={pose['t_pallet_cm']}")


if __name__ == "__main__":
    main()
