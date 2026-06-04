"""추가 필터 (spread+area+depth) 적용 후 PASS/REJECT 폴더로 overlay 저장."""
import argparse
import glob
import json
import os

import cv2
import numpy as np


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def metrics(jp):
    with open(jp) as f:
        d = json.load(f)
    obj = d["objects"][0]
    kps = np.array(obj["projected_cuboid"], dtype=np.float64)
    pose = np.array(obj["pose_transform"], dtype=np.float64)
    W = d["camera_data"]["width"]; H = d["camera_data"]["height"]
    valid = kps[:8][(kps[:8] >= 0).all(axis=1)]
    if len(valid) < 4:
        return 0.0, 0.0, 0.0, kps
    spread = float(valid.std(axis=0).sum())
    x0, y0 = valid.min(axis=0); x1, y1 = valid.max(axis=0)
    area = float((x1 - x0) * (y1 - y0) / (W * H))
    depth = float(pose[2, 3])
    return spread, area, depth, kps


def overlay(img, kps, color=(0, 255, 255)):
    out = img.copy()
    for a, b in CUBOID_EDGES:
        pa, pb = kps[a], kps[b]
        if pa[0] < 0 or pb[0] < 0:
            continue
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
    for i in range(8):
        x, y = kps[i]
        if x < 0:
            continue
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), 5, (255, 255, 0), -1)
        cv2.putText(out, str(i), (int(x) + 7, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--spread_min", type=float, default=80.0)
    p.add_argument("--area_min", type=float, default=0.05)
    p.add_argument("--area_max", type=float, default=0.50)
    p.add_argument("--depth_min", type=float, default=1.0)
    p.add_argument("--depth_max", type=float, default=5.0)
    p.add_argument("--ratio_min", type=float, default=0.2,
                   help="visible bbox h/w 또는 그 역 최소")
    p.add_argument("--ratio_max", type=float, default=5.0)
    p.add_argument("--fb_sep_min", type=float, default=8.0,
                   help="front-back face 평균 위치 분리 최소 (px)")
    p.add_argument("--hull_ratio_min", type=float, default=0.25,
                   help="convex hull / bbox 면적 비율 최소")
    args = p.parse_args()

    jsons = sorted([j for j in glob.glob(os.path.join(args.src, "*.json"))
                    if not os.path.basename(j).startswith("_")])

    pass_dir = os.path.join(args.out_root, "PASS_strictv2")
    rej_dir = os.path.join(args.out_root, "REJECT_strictv2")
    os.makedirs(pass_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)

    n_pass = 0; n_rej = 0
    for jp in jsons:
        base = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(args.src, base + ".png")
        if not os.path.exists(ip):
            continue
        s, a, d, kps = metrics(jp)
        ok_spread = s >= args.spread_min
        ok_area = args.area_min <= a <= args.area_max
        ok_depth = args.depth_min <= d <= args.depth_max

        # G: visible bbox aspect ratio
        valid = kps[:8][(kps[:8] >= 0).all(axis=1)]
        ok_ratio = True
        ok_fb = True
        ok_hull = True
        if len(valid) >= 4:
            x0, y0 = valid.min(axis=0)
            x1, y1 = valid.max(axis=0)
            w = max(1e-3, x1 - x0); h = max(1e-3, y1 - y0)
            ar = h / w
            ok_ratio = args.ratio_min <= ar <= args.ratio_max

        # H: front-back face separation
        front = kps[[0, 1, 2, 3]]
        back = kps[[4, 5, 6, 7]]
        fmask = (front >= 0).all(axis=1)
        bmask = (back >= 0).all(axis=1)
        if fmask.any() and bmask.any():
            fc = front[fmask].mean(axis=0)
            bc = back[bmask].mean(axis=0)
            fb_sep = float(np.linalg.norm(fc - bc))
            ok_fb = fb_sep >= args.fb_sep_min

        # J: convex hull / bbox ratio
        if len(valid) >= 3:
            try:
                hull = cv2.convexHull(valid.astype(np.float32).reshape(-1, 1, 2))
                hull_area = float(cv2.contourArea(hull))
                bbox_area = max(1.0, (x1 - x0) * (y1 - y0))
                ok_hull = (hull_area / bbox_area) >= args.hull_ratio_min
            except Exception:
                ok_hull = False

        all_ok = ok_spread and ok_area and ok_depth and ok_ratio and ok_fb and ok_hull

        img = cv2.imread(ip)
        color = (0, 255, 0) if all_ok else (0, 165, 255)
        ov = overlay(img, kps, color=color)
        tag = []
        if not ok_spread: tag.append(f"S{int(s)}")
        if not ok_area:   tag.append(f"A{a*100:.1f}p")
        if not ok_depth:  tag.append(f"D{d:.1f}m")
        if not ok_ratio:  tag.append("AR")
        if not ok_fb:     tag.append("FB")
        if not ok_hull:   tag.append("HL")
        suffix = "_".join(tag) if tag else "ok"

        if all_ok:
            cv2.imwrite(os.path.join(pass_dir, f"{base}_overlay.jpg"), ov)
            n_pass += 1
        else:
            cv2.imwrite(os.path.join(rej_dir, f"{base}_{suffix}_overlay.jpg"), ov)
            n_rej += 1

    print(f"PASS: {n_pass}  →  {pass_dir}")
    print(f"REJECT: {n_rej} →  {rej_dir}")


if __name__ == "__main__":
    main()
