"""6 faces × 4 in-plane rotations = 24 cases brute force test."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d
from pupil_apriltags import Detector
import argparse

PALLET_DIMS = (1.30, 1.10, 0.11)
TAG_INNER = 0.16
_W, _D, _H = PALLET_DIMS[0]/2, PALLET_DIMS[1]/2, PALLET_DIMS[2]/2

# Face name -> (R_base, t_center)
FACES = {
    "front":  (np.eye(3),                                          (0,    0,    +_D)),  # +Z
    "back":   (np.array([[-1,0,0],[0,1,0],[0,0,-1]],float),        (0,    0,    -_D)),  # -Z
    "left":   (np.array([[ 0,0,-1],[0,1,0],[1,0,0]],float),        (-_W,  0,    0)),    # -X
    "right":  (np.array([[ 0,0,1],[0,1,0],[-1,0,0]],float),        (+_W,  0,    0)),    # +X
    "top":    (np.array([[-1,0,0],[0,0,-1],[0,-1,0]],float),       (0,    -_H,  0)),    # -Y (top in Y=down)
    "bottom": (np.array([[1,0,0],[0,0,1],[0,1,0]],float),          (0,    +_H,  0)),    # +Y
}


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)


def project(T, K):
    kp = make_pallet_keypoints_3d(width=PALLET_DIMS[0], depth=PALLET_DIMS[1], height=PALLET_DIMS[2])
    R, t = T[:3,:3], T[:3,3]
    pts = (R @ kp.T).T + t
    fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
    out=[]
    for p in pts:
        if p[2]<=0: out.append([-1,-1])
        else: out.append([fx*p[0]/p[2]+cx, fy*p[1]/p[2]+cy])
    return out


def draw(img, uv, label):
    vis = img.copy()
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = [tuple(int(c) for c in p[:2]) for p in uv]
    for i,j in E:
        if pts[i][0]>=0 and pts[j][0]>=0:
            cv2.line(vis, pts[i], pts[j], (0,255,0), 2)
    for idx,pt in enumerate(pts[:8]):
        if pt[0]>=0:
            cv2.circle(vis, pt, 3, (0,0,255), -1)
    if len(pts)>8 and pts[8][0]>=0:
        cv2.circle(vis, pts[8], 5, (255,255,255), -1)
    cv2.rectangle(vis, (0,0), (vis.shape[1], 24), (0,0,0), -1)
    cv2.putText(vis, label, (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=int, required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    K = np.loadtxt('data/pallet/raw_data/capture0403middle/cam_K.txt')
    cp = (K[0,0],K[1,1],K[0,2],K[1,2])
    img = cv2.imread(args.image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = Detector(families='tag36h11', nthreads=4)
    rs = det.detect(g, estimate_tag_pose=True, camera_params=cp, tag_size=TAG_INNER)
    target=None
    for r in rs:
        if r.tag_id == args.tag:
            target=r; break
    if target is None:
        print(f"Tag {args.tag} not found"); return
    print(f"Tag {args.tag} margin={target.decision_margin:.1f}")

    T_ct = np.eye(4); T_ct[:3,:3]=target.pose_R; T_ct[:3,3]=target.pose_t.flatten()

    out_dir = f"data/pallet/raw_data/capture0403middle/gt_pilot/debug_brute_id{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for face_name, (R_base, t_center) in FACES.items():
        row = []
        for ang in [0,90,180,270]:
            R_pt = R_base @ rot_z(ang)
            T_pt = np.eye(4); T_pt[:3,:3]=R_pt; T_pt[:3,3]=t_center
            T_cp = T_ct @ np.linalg.inv(T_pt)
            uv = project(T_cp, K)
            label = f"{face_name} rot{ang}"
            vis = draw(img, uv, label)
            row.append(vis)
            cv2.imwrite(f"{out_dir}/{face_name}_rot{ang:03d}.jpg", vis)
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(f"{out_dir}/grid.jpg", grid)
    print(f"Grid: {out_dir}/grid.jpg")


if __name__ == "__main__":
    main()
