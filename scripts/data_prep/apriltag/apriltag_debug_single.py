"""단일 tag 디버그: 위치 후보 × 회전 4개 grid overlay.

사용법:
    python scripts/data_prep/apriltag_debug_single.py --tag 1 \
        --image data/pallet/raw_data/capture0403middle/rgb/<frame>.png
"""
import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d
from pupil_apriltags import Detector as AprilTagDetector

CAM_K = "data/pallet/raw_data/capture0403middle/cam_K.txt"
OUT_ROOT = "data/pallet/raw_data/capture0403middle/gt_pilot"
PALLET_DIMS = (1.30, 1.10, 0.11)  # (W=X=130cm, D=Z=110cm, H=Y=11cm)
TAG_INNER = 0.16
_W, _D, _H = PALLET_DIMS[0] / 2, PALLET_DIMS[1] / 2, PALLET_DIMS[2] / 2
_EDGE = 0.10


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# Base R for each face. Reading-up interpretation same as id4-confirmed:
# top face: tag reading-up -> pallet +X (after rot 270°)
# But we'll test all 4 rotations, so the base only needs axis orientation right.
def R_top():
    # tag-Z = -Y (out of top)
    return np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)


def R_front():
    # front face = +Z, tag-Z = +Z (out of front)
    return np.eye(3)


def R_left():
    # left face = -X, tag-Z = -X
    return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)


def R_right():
    # right face = +X, tag-Z = +X
    # tagY=(0,1,0), tagX = tagY x tagZ = (0,1,0)x(1,0,0)=(0,0,-1)
    return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)


def R_back():
    # back face = -Z, tag-Z = -Z
    # tagY=(0,1,0), tagX = tagY x tagZ = (0,1,0)x(0,0,-1)=(-1,0,0)
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)


FACE_R = {"top": R_top, "front": R_front, "left": R_left,
          "right": R_right, "back": R_back}


def build_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project(T_cam_from_pallet, K):
    kp = make_pallet_keypoints_3d(width=PALLET_DIMS[0], depth=PALLET_DIMS[1], height=PALLET_DIMS[2])
    R = T_cam_from_pallet[:3, :3]
    t = T_cam_from_pallet[:3, 3]
    pts = (R @ kp.T).T + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    out = []
    for p in pts:
        if p[2] <= 0:
            out.append([-1, -1])
        else:
            out.append([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])
    return out


def draw(img, uv, label):
    vis = img.copy()
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = [tuple(int(c) for c in p[:2]) for p in uv]
    for i, j in E:
        if pts[i][0] >= 0 and pts[j][0] >= 0:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 2)
    cols = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,128),(128,0,255)]
    for idx, pt in enumerate(pts[:8]):
        if pt[0] >= 0:
            cv2.circle(vis, pt, 4, cols[idx], -1)
    if len(pts) > 8 and pts[8][0] >= 0:
        cv2.circle(vis, pts[8], 5, (255, 255, 255), -1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=int, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--face", choices=["top", "front", "left", "right", "back"], default="top")
    # Candidate positions (comma-separated triples)
    ap.add_argument("--positions", required=True,
                    help="label1:x1,y1,z1;label2:x2,y2,z2")
    args = ap.parse_args()

    K = np.loadtxt(CAM_K)
    cam_params = (K[0,0], K[1,1], K[0,2], K[1,2])
    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = AprilTagDetector(families="tag36h11", nthreads=4)
    results = det.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=TAG_INNER)

    target = None
    for r in results:
        if r.tag_id == args.tag:
            target = r
            break
    if target is None:
        print(f"Tag {args.tag} not detected in {args.image}")
        print(f"Detected tags: {[r.tag_id for r in results]}")
        return

    T_ct = np.eye(4)
    T_ct[:3, :3] = target.pose_R
    T_ct[:3, 3] = target.pose_t.flatten()
    print(f"Tag {args.tag} margin={target.decision_margin:.1f}")

    # Parse positions
    positions = {}
    for item in args.positions.split(";"):
        label, coords = item.split(":")
        xyz = np.array([float(x) for x in coords.split(",")])
        positions[label] = xyz

    R_base = FACE_R[args.face]()
    out_dir = os.path.join(OUT_ROOT, f"debug_id{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for pos_label, t_pal in positions.items():
        row_imgs = []
        for ang in [0, 90, 180, 270]:
            R_pt = R_base @ rot_z(ang)
            T_pt = build_T(R_pt, t_pal)
            T_cp = T_ct @ np.linalg.inv(T_pt)
            uv = project(T_cp, K)
            label = f"id{args.tag} {pos_label} rot{ang}"
            vis = draw(img, uv, label)
            out = os.path.join(out_dir, f"{pos_label}_rot{ang:03d}.jpg")
            cv2.imwrite(out, vis)
            row_imgs.append(vis)
            print(f"  {label} -> {out}")
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)
    grid_path = os.path.join(out_dir, "grid.jpg")
    cv2.imwrite(grid_path, grid)
    print(f"\nGrid: {grid_path}")


if __name__ == "__main__":
    main()
