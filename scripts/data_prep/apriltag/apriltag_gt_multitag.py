"""Multi-tag AprilTag → Pallet GT (capture0403middle 전용).

5개 AprilTag가 팔레트 여러 면에 붙어있는 데이터셋용.
각 tag ID 별 T_pallet_from_tag를 미리 정의하고, 프레임마다 감지된 tag 중
decision_margin이 가장 높은 것을 선택해 pallet pose를 복원.

사용법:
    # 파일럿 1장
    python scripts/data_prep/apriltag/apriltag_gt_multitag.py \
        --image data/pallet/raw_data/capture0403middle/rgb/1775201190067278336.png \
        --cam_k data/pallet/raw_data/capture0403middle/cam_K.txt \
        --output_dir data/pallet/raw_data/capture0403middle/gt_pilot

    # 배치
    python scripts/data_prep/apriltag/apriltag_gt_multitag.py \
        --input_dir data/pallet/raw_data/capture0403middle/rgb \
        --cam_k data/pallet/raw_data/capture0403middle/cam_K.txt \
        --output_dir data/pallet/raw_data/capture0403middle/gt
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_LIB = "pupil"
except ImportError:
    from dt_apriltags import Detector as AprilTagDetector
    APRILTAG_LIB = "dt"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d_isaac, make_camera_matrix


# ============================================================
# Pallet geometry (pnp_solver convention: Y=down)
#   X = medium (width)  = 1.10 m
#   Y = height (down)   = 0.11 m
#   Z = long   (depth)  = 1.30 m
#   Origin = geometric center
#   Front face = Z_max, Top face = Y_min (Y=down이므로 top=-h)
# ============================================================
PALLET_DIMS = (1.10, 1.30, 0.11)   # (width_X=110cm, depth_Z=130cm, height_Y=11cm)

# AprilTag geometry
TAG_FAMILY = "tag36h11"
TAG_INNER_SIZE_M = 0.16    # 20cm outer - 2cm×2 white border = 16cm inner

# ============================================================
# Tag configuration
#   T_pallet_from_tag: 4x4 (tag coord → pallet coord)
#   AprilTag coord: tag-X=right, tag-Y=down, tag-Z=out of plane
#   "숫자 읽을 수 있는 방향"이 각 tag의 -tag-Y (reading up)
# ============================================================
def _T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T

# Empirically derived from id4-anchor co-detection chain (see _locate_all_tags.py)
# Anchor: id4 visually verified via 8-case debug (Xneg rot270 with PALLET_DIMS=(1.30,1.10,0.11)).
# Other tags computed by transforming their detected pose into pallet frame using id4 anchor.
TAG_CONFIG = {
    # Anchor: id4 (top face, manual visual confirmation)
    4: _T(
        R=[[ 0, -1,  0],
           [ 0,  0, -1],
           [ 1,  0,  0]],
        t=(-0.55, -0.055, -0.45),
    ),
    # id0: empirical (244 frames, std X=17mm Y=15mm Z=7mm)
    0: _T(
        R=[[-0.012, -0.023,  1.000],
           [ 0.035, -0.999, -0.022],
           [ 0.999,  0.035,  0.013]],
        t=(-0.6718, -0.0693, 0.1047),
    ),
    # id1: empirical (55 frames, std X=20mm Y=19mm Z=18mm)
    1: _T(
        R=[[-0.008, -1.000,  0.005],
           [ 0.014, -0.005, -1.000],
           [ 1.000, -0.008,  0.014]],
        t=(-0.5368, -0.0482, 0.6524),
    ),
    # id5: empirical (29 frames, std X=18mm Y=11mm Z=6mm)
    5: _T(
        R=[[ 0.007, -1.000,  0.021],
           [-0.006, -0.021, -1.000],
           [ 1.000,  0.006, -0.006]],
        t=(0.3812, -0.0604, 0.1046),
    ),
    # id2: chain via id4 (63 frames, std X=13mm Y=11mm Z=7mm, margin low ~25)
    2: _T(
        R=[[-1.000,  0.019,  0.012],
           [-0.018, -0.998,  0.068],
           [ 0.013,  0.068,  0.998]],
        t=(-0.1153, -0.0714, -0.5725),
    ),
}


def load_refined_config(path):
    """tag_config_refined.json을 로드하여 TAG_CONFIG 덮어쓰기."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    refined = {}
    for tid_str, entry in d.items():
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(entry["R"], dtype=np.float64)
        T[:3, 3] = np.array(entry["t"], dtype=np.float64)
        refined[int(tid_str)] = T
    return refined


def create_detector():
    if APRILTAG_LIB == "pupil":
        return AprilTagDetector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            decode_sharpening=0.25,
        )
    return AprilTagDetector(families=TAG_FAMILY, nthreads=4)


def load_cam_k(path):
    K = np.loadtxt(path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return fx, fy, cx, cy, K


def detect_tags(gray, detector, cam_params):
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=TAG_INNER_SIZE_M,
    )
    out = []
    for r in results:
        if r.tag_id not in TAG_CONFIG:
            continue
        T_cam_from_tag = np.eye(4)
        T_cam_from_tag[:3, :3] = r.pose_R
        T_cam_from_tag[:3, 3] = r.pose_t.flatten()
        out.append({
            "tag_id": int(r.tag_id),
            "T_cam_from_tag": T_cam_from_tag,
            "corners": r.corners,
            "center": r.center,
            "decision_margin": float(r.decision_margin),
            "hamming": int(r.hamming),
            "pose_err": float(r.pose_err) if hasattr(r, "pose_err") else 0.0,
        })
    return out


def tag_to_pallet(T_cam_from_tag, tag_id):
    T_pallet_from_tag = TAG_CONFIG[tag_id]
    # T_cam_from_pallet = T_cam_from_tag @ inv(T_pallet_from_tag)
    return T_cam_from_tag @ np.linalg.inv(T_pallet_from_tag)


def project_cuboid(T_cam_from_pallet, K):
    """Returns (uv (list of 8 [x,y]), centroid [x,y]).
    Points behind camera get [-1,-1] sentinel.
    """
    kp3d = make_pallet_keypoints_3d_isaac(
        width=PALLET_DIMS[0], depth=PALLET_DIMS[1], height=PALLET_DIMS[2],
    )  # (9, 3) Isaac canonical ordering
    R = T_cam_from_pallet[:3, :3]
    t = T_cam_from_pallet[:3, 3]
    pts_cam = (R @ kp3d.T).T + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    proj = []
    for p in pts_cam:
        if p[2] <= 0:
            proj.append([-1.0, -1.0])
        else:
            u = fx * p[0] / p[2] + cx
            v = fy * p[1] / p[2] + cy
            proj.append([float(u), float(v)])
    return proj[:8], proj[8]


def draw_overlay(img, cuboid, centroid, tag_dets, chosen_tag_id):
    vis = img.copy()
    h, w = img.shape[:2]
    EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    # Valid = not the [-1,-1] sentinel (i.e., in front of camera)
    # On-screen position may still be negative (off-screen) — cv2.line handles clipping
    def valid(p):
        return not (p[0] == -1 and p[1] == -1)
    pts = [(int(p[0]), int(p[1])) if valid(p) else None for p in cuboid]
    for i, j in EDGES:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 1, cv2.LINE_AA)
    colors = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),
              (255,255,0),(255,0,0),(255,0,128),(128,0,255)]
    for idx in range(8):
        pt = pts[idx]
        if pt is None: continue
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(vis, pt, 4, (0, 0, 0), -1)
            cv2.circle(vis, pt, 3, colors[idx], -1)
            cv2.putText(vis, str(idx), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[idx], 1)
    if valid(centroid):
        cx, cy = int(centroid[0]), int(centroid[1])
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(vis, "C", (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # Draw all detected tags, highlight chosen
    for det in tag_dets:
        color = (0, 255, 255) if det["tag_id"] == chosen_tag_id else (255, 0, 255)
        corners = det["corners"]
        for i in range(4):
            p1 = tuple(int(c) for c in corners[i])
            p2 = tuple(int(c) for c in corners[(i+1) % 4])
            cv2.line(vis, p1, p2, color, 2)
        tcx, tcy = int(det["center"][0]), int(det["center"][1])
        cv2.putText(vis, f"id{det['tag_id']} m={det['decision_margin']:.0f}",
                    (tcx+5, tcy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def process_image(img_path, out_json, out_overlay, detector, cam_params, K):
    img = cv2.imread(img_path)
    if img is None:
        return None, "read_fail"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detect_tags(gray, detector, cam_params)
    if not dets:
        return None, "no_tag"

    # Pick best-margin tag → compute pallet pose
    best = max(dets, key=lambda d: d["decision_margin"])
    T_cam_from_pallet = tag_to_pallet(best["T_cam_from_tag"], best["tag_id"])
    cuboid, centroid = project_cuboid(T_cam_from_pallet, K)

    h, w = img.shape[:2]
    ann = {
        "camera_data": {
            "width": w, "height": h,
            "intrinsics": {"fx": cam_params[0], "fy": cam_params[1],
                           "cx": cam_params[2], "cy": cam_params[3]},
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "pose_transform": T_cam_from_pallet.tolist(),
            "projected_cuboid": cuboid,
            "projected_cuboid_centroid": centroid,
            "dimensions_m": {"width": PALLET_DIMS[0],
                             "depth": PALLET_DIMS[1],
                             "height": PALLET_DIMS[2]},
            "gt_source": "apriltag_multitag",
            "chosen_tag_id": best["tag_id"],
            "chosen_decision_margin": best["decision_margin"],
            "all_detected_tags": [d["tag_id"] for d in dets],
        }],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ann, f, indent=2)

    if out_overlay is not None:
        os.makedirs(os.path.dirname(out_overlay), exist_ok=True)
        vis = draw_overlay(img, cuboid, centroid, dets, best["tag_id"])
        cv2.imwrite(out_overlay, vis)
    return best, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="단일 이미지")
    p.add_argument("--input_dir", help="rgb 디렉토리 (배치)")
    p.add_argument("--cam_k", required=True, help="cam_K.txt 경로")
    p.add_argument("--output_dir", required=True, help="gt JSON + overlay 출력 루트")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--no_overlay", action="store_true")
    p.add_argument("--refined_config", default=None,
                   help="refined config JSON (tag_config_refined.json). 자동 탐색: output_dir/tag_config_refined.json")
    args = p.parse_args()

    fx, fy, cx, cy, K = load_cam_k(args.cam_k)
    cam_params = (fx, fy, cx, cy)
    detector = create_detector()

    # Try loading refined config (merges with hardcoded TAG_CONFIG)
    refined_path = args.refined_config or os.path.join(args.output_dir, "tag_config_refined.json")
    refined = load_refined_config(refined_path)
    if refined is not None:
        global TAG_CONFIG
        merged = dict(TAG_CONFIG)
        merged.update(refined)
        TAG_CONFIG = merged
        print(f"[refined] Merged {len(refined)} tags from {refined_path}: {sorted(refined.keys())}")
    else:
        print(f"[default] Using hardcoded TAG_CONFIG ({refined_path} not found)")

    print(f"Pallet dims (W,D,H) = {PALLET_DIMS}")
    print(f"Tag family = {TAG_FAMILY}, inner_size = {TAG_INNER_SIZE_M}m")
    print(f"Camera fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"Configured tag IDs: {sorted(TAG_CONFIG.keys())}")
    print(f"AprilTag lib: {APRILTAG_LIB}")
    print()

    gt_dir = os.path.join(args.output_dir, "gt")
    ov_dir = os.path.join(args.output_dir, "gt_overlay")

    if args.image:
        name = os.path.splitext(os.path.basename(args.image))[0]
        out_json = os.path.join(gt_dir, f"{name}.json")
        out_ov = None if args.no_overlay else os.path.join(ov_dir, f"{name}.jpg")
        best, status = process_image(args.image, out_json, out_ov, detector, cam_params, K)
        print(f"[{status}] {args.image}")
        if best:
            print(f"  chosen tag={best['tag_id']} margin={best['decision_margin']:.1f}")
            print(f"  JSON:    {out_json}")
            if out_ov:
                print(f"  Overlay: {out_ov}")
        return

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Processing {len(files)} files...")
    stats = {"ok": 0, "no_tag": 0, "read_fail": 0}
    tag_use = {}
    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        out_json = os.path.join(gt_dir, f"{name}.json")
        out_ov = None if args.no_overlay else os.path.join(ov_dir, f"{name}.jpg")
        best, status = process_image(f, out_json, out_ov, detector, cam_params, K)
        stats[status] += 1
        if best:
            tag_use[best["tag_id"]] = tag_use.get(best["tag_id"], 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] ok={stats['ok']} no_tag={stats['no_tag']}")
    print()
    print(f"Total: {len(files)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Chosen tag usage: {tag_use}")


if __name__ == "__main__":
    main()
