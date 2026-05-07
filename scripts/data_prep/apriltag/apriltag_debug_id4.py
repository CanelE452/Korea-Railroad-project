"""id4 단독 디버그: 4회전 × 2 X-부호 = 8가지 T_pallet_from_tag 케이스.

각 케이스로 cuboid overlay를 그려서 어떤 설정이 팔레트와 일치하는지 시각 비교.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d

from pupil_apriltags import Detector as AprilTagDetector

# ---- 설정 ----
IMAGE = "data/pallet/raw_data/capture0403middle/rgb/1775201190067278336.png"
CAM_K = "data/pallet/raw_data/capture0403middle/cam_K.txt"
OUT_DIR = "data/pallet/raw_data/capture0403middle/gt_pilot/debug_id4"

PALLET_DIMS = (1.10, 1.30, 0.11)  # (W=X, D=Z, H=Y)
TAG_INNER = 0.16
TAG_FAMILY = "tag36h11"

_W, _D, _H = PALLET_DIMS[0] / 2, PALLET_DIMS[1] / 2, PALLET_DIMS[2] / 2
_EDGE = 0.10  # 10 cm from edge to tag center


def rot_z(deg):
    """in-plane rotation around tag-Z axis (in tag's own frame)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def base_top_face_R():
    """Top face base rotation: tag-Z=-Y(out of top), tag-Y=-Z(reading up = +Z)."""
    return np.array([[-1, 0,  0],
                     [ 0, 0, -1],
                     [ 0,-1,  0]], dtype=np.float64)


def build_T(R_pallet_from_tag, t):
    T = np.eye(4)
    T[:3, :3] = R_pallet_from_tag
    T[:3, 3] = t
    return T


def project_cuboid(T_cam_from_pallet, K):
    kp3d = make_pallet_keypoints_3d(
        width=PALLET_DIMS[0], depth=PALLET_DIMS[1], height=PALLET_DIMS[2],
    )
    R = T_cam_from_pallet[:3, :3]
    t = T_cam_from_pallet[:3, 3]
    pts = (R @ kp3d.T).T + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uv = []
    for p in pts:
        if p[2] <= 0:
            uv.append([-1, -1])
        else:
            uv.append([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])
    return uv


def draw(img, uv, label):
    vis = img.copy()
    EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = [tuple(int(c) for c in p[:2]) for p in uv]
    for i, j in EDGES:
        if pts[i][0] >= 0 and pts[j][0] >= 0:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 2)
    colors = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),
              (255,255,0),(255,0,0),(255,0,128),(128,0,255)]
    for idx, pt in enumerate(pts[:8]):
        if pt[0] >= 0:
            cv2.circle(vis, pt, 5, colors[idx], -1)
            cv2.putText(vis, str(idx), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[idx], 1)
    # centroid
    if len(pts) > 8 and pts[8][0] >= 0:
        cv2.circle(vis, pts[8], 6, (255, 255, 255), -1)
    # label bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(vis, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    K = np.loadtxt(CAM_K)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    cam_params = (fx, fy, cx, cy)

    img = cv2.imread(IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = AprilTagDetector(families=TAG_FAMILY, nthreads=4)
    results = det.detect(gray, estimate_tag_pose=True,
                          camera_params=cam_params, tag_size=TAG_INNER)

    tag4 = None
    for r in results:
        if r.tag_id == 4:
            tag4 = r
            break
    if tag4 is None:
        print("Tag 4 not detected. Aborting.")
        return

    T_cam_from_tag = np.eye(4)
    T_cam_from_tag[:3, :3] = tag4.pose_R
    T_cam_from_tag[:3, 3] = tag4.pose_t.flatten()
    print(f"Tag 4 detected. margin={tag4.decision_margin:.1f}")

    # Two X-sign options for id4 position in pallet frame
    # Interpretation A: "left-back" means pallet -X, -Z (default)
    # Interpretation B: "left-back" means pallet +X, -Z (flipped)
    t_cases = {
        "Xneg": np.array([-_W + _EDGE, -_H, -_D + _EDGE]),
        "Xpos": np.array([+_W - _EDGE, -_H, -_D + _EDGE]),
    }

    R_base = base_top_face_R()
    angles = [0, 90, 180, 270]

    for x_name, t_pallet in t_cases.items():
        for ang in angles:
            R_inplane = rot_z(ang)
            # In-plane rotation applied in tag's own frame (right-multiply)
            R_pt = R_base @ R_inplane
            T_pallet_from_tag = build_T(R_pt, t_pallet)

            T_cam_from_pallet = T_cam_from_tag @ np.linalg.inv(T_pallet_from_tag)
            uv = project_cuboid(T_cam_from_pallet, K)

            label = f"id4 {x_name} rot={ang}"
            vis = draw(img, uv, label)
            out = os.path.join(OUT_DIR, f"id4_{x_name}_rot{ang:03d}.jpg")
            cv2.imwrite(out, vis)
            print(f"  {label} -> {out}")

    # grid 2x4
    tiles = []
    for x_name in ["Xneg", "Xpos"]:
        row = []
        for ang in angles:
            p = os.path.join(OUT_DIR, f"id4_{x_name}_rot{ang:03d}.jpg")
            row.append(cv2.imread(p))
        tiles.append(np.hstack(row))
    grid = np.vstack(tiles)
    grid_path = os.path.join(OUT_DIR, "grid_2x4.jpg")
    cv2.imwrite(grid_path, grid)
    print(f"\nGrid: {grid_path}")


if __name__ == "__main__":
    main()
