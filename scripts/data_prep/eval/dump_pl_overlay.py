"""Phase 1 PL pool 의 모든 통과 frame 에 cuboid overlay 그려서 폴더 저장.

기존 data/pallet/eval_results/v8ablationa_noapril/BC/ 같은 형식.
"""
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


def overlay(img, kps):
    out = img.copy()
    for (a, b) in CUBOID_EDGES:
        pa, pb = kps[a], kps[b]
        if pa[0] < 0 or pb[0] < 0:
            continue
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 (0, 255, 255), 2)  # yellow
    for i in range(8):
        x, y = kps[i]
        if x < 0:
            continue
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), 5, (255, 255, 0), -1)  # cyan
        cv2.putText(out, str(i), (int(x) + 7, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    # centroid (idx 8)
    if len(kps) > 8 and kps[8][0] >= 0:
        cv2.circle(out, (int(kps[8][0]), int(kps[8][1])), 5, (0, 0, 255), -1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dirs", nargs="+", required=True,
                   help="PL pool dirs (e.g. output/pl_outside_r0_loo)")
    p.add_argument("--out_root", required=True,
                   help="Output root (e.g. data/pallet/eval_results/phase1_pl_overlays)")
    args = p.parse_args()

    for src in args.src_dirs:
        tag = os.path.basename(os.path.normpath(src))
        out_dir = os.path.join(args.out_root, tag, "ABC_pass")
        os.makedirs(out_dir, exist_ok=True)
        jsons = sorted(glob.glob(os.path.join(src, "*.json")))
        jsons = [j for j in jsons if not os.path.basename(j).startswith("_")]
        n = 0
        for jp in jsons:
            base = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(src, base + ".png")
            if not os.path.exists(ip):
                continue
            with open(jp) as f:
                ndds = json.load(f)
            kps = ndds["objects"][0]["projected_cuboid"]
            img = cv2.imread(ip)
            ov = overlay(img, kps)
            cv2.imwrite(os.path.join(out_dir, base + "_overlay.jpg"), ov)
            n += 1
        print(f"  {tag}: {n} overlays → {out_dir}")


if __name__ == "__main__":
    main()
