"""수동 보정된 GT에서 T_pallet_from_tag를 역산하여 refined TAG_CONFIG 생성.

흐름:
  1. gt_manual/*.json 에서 각 프레임의 T_cam_from_pallet 로드
  2. 해당 이미지에서 AprilTag 감지
  3. 각 감지된 tag에 대해 T_pallet_from_tag = inv(T_cp) @ T_ct 계산
  4. 여러 프레임에 걸쳐 평균 (quaternion averaging for R, mean for t)
  5. 결과를 JSON으로 저장 → apriltag_gt_multitag.py가 로드

사용법:
  python scripts/data_prep/refine_tag_config.py \
      --capture data/pallet/raw_data/capture0403middle \
      --out data/pallet/raw_data/capture0403middle/tag_config_refined.json
"""
import argparse
import json
import os

import cv2
import numpy as np
from pupil_apriltags import Detector

TAG_FAMILY = "tag36h11"
TAG_INNER = 0.16


def avg_quat(qs):
    """Simple quaternion averaging by eigendecomposition (Markley's method)."""
    A = np.zeros((4, 4))
    for q in qs:
        q = q / np.linalg.norm(q)
        A += np.outer(q, q)
    A /= len(qs)
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs[:, -1]  # largest eigenvalue


def R_to_quat(R):
    """Convert 3x3 R to quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y+z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x+y*y)],
    ])


def load_manual_pose(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["objects"][0]["pose_transform"], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rgb_dir = os.path.join(args.capture, "rgb")
    manual_dir = os.path.join(args.capture, "gt_manual")
    cam_k_path = os.path.join(args.capture, "cam_K.txt")

    if not os.path.isdir(manual_dir):
        print(f"ERROR: {manual_dir} not found. 수동 보정된 프레임이 하나도 없음.")
        return

    K = np.loadtxt(cam_k_path)
    cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    detector = Detector(families=TAG_FAMILY, nthreads=4)

    manual_files = sorted([f for f in os.listdir(manual_dir) if f.endswith(".json")])
    print(f"Found {len(manual_files)} manually-corrected frames")
    if len(manual_files) == 0:
        return

    # Collect T_pallet_from_tag samples per tag
    collected = {}  # tag_id -> list of T_pt (4x4)
    per_frame_stats = []
    for jf in manual_files:
        name = jf[:-5]
        img_path = os.path.join(rgb_dir, f"{name}.png")
        if not os.path.exists(img_path):
            continue
        T_cp = load_manual_pose(os.path.join(manual_dir, jf))
        T_pc = np.linalg.inv(T_cp)  # pallet_from_cam

        img = cv2.imread(img_path)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(g, estimate_tag_pose=True,
                                camera_params=cam_params, tag_size=TAG_INNER)

        tag_ids_here = []
        for r in dets:
            T_ct = np.eye(4)
            T_ct[:3, :3] = r.pose_R
            T_ct[:3, 3] = r.pose_t.flatten()
            T_pt = T_pc @ T_ct
            collected.setdefault(r.tag_id, []).append((T_pt, r.decision_margin))
            tag_ids_here.append(r.tag_id)
        per_frame_stats.append((name, tag_ids_here))

    print("\nPer-frame detected tags:")
    for name, ids in per_frame_stats:
        print(f"  {name}: {sorted(ids)}")

    print("\nAveraged TAG_CONFIG:")
    config_out = {}
    for tid in sorted(collected.keys()):
        items = collected[tid]
        ts = np.array([T[:3, 3] for T, _ in items])
        quats = np.array([R_to_quat(T[:3, :3]) for T, _ in items])
        # Ensure consistent hemisphere (flip sign if dot < 0 with first)
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        t_mean = ts.mean(axis=0)
        t_std = ts.std(axis=0)
        q_mean = avg_quat(quats)
        R_mean = quat_to_R(q_mean)

        print(f"\nid{tid}: {len(items)} samples")
        print(f"  t mean: {t_mean.round(4)}")
        print(f"  t std:  {t_std.round(4)} (mm: {(t_std*1000).round(1)})")
        print(f"  R =")
        for row in R_mean.round(4):
            print(f"    {row.tolist()}")

        T_final = np.eye(4)
        T_final[:3, :3] = R_mean
        T_final[:3, 3] = t_mean
        config_out[str(tid)] = {
            "t": t_mean.tolist(),
            "R": R_mean.tolist(),
            "T": T_final.tolist(),
            "n_samples": len(items),
            "t_std_mm": (t_std * 1000).tolist(),
        }

    # Save
    out_path = args.out or os.path.join(args.capture, "tag_config_refined.json")
    with open(out_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
