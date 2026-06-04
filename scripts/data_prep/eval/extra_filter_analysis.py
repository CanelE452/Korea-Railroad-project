"""기존 PL pool 에 추가 필터 (spread / area / depth) 적용한 통계 + 시각화."""
import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def compute_metrics(json_path):
    with open(json_path) as f:
        d = json.load(f)
    obj = d["objects"][0]
    kps = np.array(obj["projected_cuboid"], dtype=np.float64)
    pose = np.array(obj["pose_transform"], dtype=np.float64)
    W = d["camera_data"]["width"]; H = d["camera_data"]["height"]

    # corner spread (x_std + y_std) over 8 corners
    valid = kps[:8][(kps[:8] >= 0).all(axis=1)]
    if len(valid) >= 4:
        spread = float(valid.std(axis=0).sum())
    else:
        spread = 0.0

    # bounding box area as fraction of image
    if len(valid) >= 4:
        x0, y0 = valid.min(axis=0)
        x1, y1 = valid.max(axis=0)
        area_frac = float((x1 - x0) * (y1 - y0) / (W * H))
    else:
        area_frac = 0.0

    # depth (translation z)
    depth = float(pose[2, 3])
    return spread, area_frac, depth, kps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--output_root", default="_docs/figures")
    p.add_argument("--tag", default="phase2_outside_filter")
    args = p.parse_args()

    jsons = sorted([j for j in glob.glob(os.path.join(args.src, "*.json"))
                    if not os.path.basename(j).startswith("_")])
    print(f"Total PL: {len(jsons)}")

    spreads, areas, depths = [], [], []
    for jp in jsons:
        s, a, d, _ = compute_metrics(jp)
        spreads.append(s); areas.append(a); depths.append(d)
    spreads = np.array(spreads); areas = np.array(areas); depths = np.array(depths)

    # --- 통계 distribution figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(spreads, bins=40, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.axvline(80, color="red", linestyle="--", linewidth=2, label="threshold = 80 px")
    ax.set_xlabel("Corner spread (x_std + y_std, px)")
    ax.set_ylabel("# PL")
    ax.set_title("(a) Corner spread distribution")
    ax.legend()

    ax = axes[1]
    ax.hist(areas * 100, bins=40, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax.axvline(5,  color="red", linestyle="--", linewidth=2, label="threshold = [5%, 50%]")
    ax.axvline(50, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Cuboid area / image area (%)")
    ax.set_ylabel("# PL")
    ax.set_title("(b) Cuboid area fraction")
    ax.legend()

    ax = axes[2]
    ax.hist(depths, bins=40, color="#d62728", edgecolor="black", alpha=0.75)
    ax.axvline(1, color="red", linestyle="--", linewidth=2, label="threshold = [1, 5] m")
    ax.axvline(5, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Recovered depth (m)")
    ax.set_ylabel("# PL")
    ax.set_title("(c) Depth distribution")
    ax.legend()

    fig.suptitle("Phase 2 — extra filter analysis on existing 1432 outside PLs",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out1 = os.path.join(args.output_root, f"{args.tag}_dist.png")
    plt.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"Saved: {out1}")

    # --- 필터 적용 후 통과 수 ---
    pass_spread = spreads >= 80
    pass_area = (areas >= 0.05) & (areas <= 0.5)
    pass_depth = (depths >= 1.0) & (depths <= 5.0)
    all_pass = pass_spread & pass_area & pass_depth

    print()
    print(f"Filter pass counts (N={len(jsons)}):")
    print(f"  spread >= 80 px        : {pass_spread.sum():4d} ({pass_spread.mean()*100:.1f}%)")
    print(f"  area  in [5%, 50%]     : {pass_area.sum():4d} ({pass_area.mean()*100:.1f}%)")
    print(f"  depth in [1, 5] m      : {pass_depth.sum():4d} ({pass_depth.mean()*100:.1f}%)")
    print(f"  ALL three pass         : {all_pass.sum():4d} ({all_pass.mean()*100:.1f}%)")

    # --- 통과/제외 sample 시각화 ---
    pass_idx = np.where(all_pass)[0]
    reject_idx = np.where(~all_pass)[0]

    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        if i < len(pass_idx):
            jp = jsons[pass_idx[len(pass_idx) // 4 * (i + 1) - 1]]
            base = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(args.src, base + ".png")
            img = cv2.imread(ip)
            with open(jp) as f:
                ndds = json.load(f)
            kps = ndds["objects"][0]["projected_cuboid"]
            ax = axes2[0][i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for a, b in CUBOID_EDGES:
                pa, pb = kps[a], kps[b]
                if pa[0] < 0 or pb[0] < 0: continue
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "y-", linewidth=1.8)
            for k in range(8):
                x, y = kps[k]
                if x < 0: continue
                ax.scatter(x, y, c="cyan", s=35, edgecolor="black", linewidth=0.5, zorder=5)
            ax.set_title(f"PASS: {base[:14]}\nspread={spreads[pass_idx[len(pass_idx)//4*(i+1)-1]]:.0f}",
                         fontsize=8, color="green")
            ax.set_xticks([]); ax.set_yticks([])

        if i < len(reject_idx):
            j_idx = reject_idx[len(reject_idx) // 4 * (i + 1) - 1]
            jp = jsons[j_idx]
            base = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(args.src, base + ".png")
            img = cv2.imread(ip)
            with open(jp) as f:
                ndds = json.load(f)
            kps = ndds["objects"][0]["projected_cuboid"]
            ax = axes2[1][i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for a, b in CUBOID_EDGES:
                pa, pb = kps[a], kps[b]
                if pa[0] < 0 or pb[0] < 0: continue
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="orange", linewidth=1.8)
            for k in range(8):
                x, y = kps[k]
                if x < 0: continue
                ax.scatter(x, y, c="red", s=35, edgecolor="black", linewidth=0.5, zorder=5)
            reasons = []
            if not pass_spread[j_idx]: reasons.append(f"spread={spreads[j_idx]:.0f}")
            if not pass_area[j_idx]:   reasons.append(f"area={areas[j_idx]*100:.1f}%")
            if not pass_depth[j_idx]:  reasons.append(f"depth={depths[j_idx]:.1f}")
            ax.set_title(f"REJECT: {base[:14]}\n" + " ".join(reasons),
                         fontsize=8, color="red")
            ax.set_xticks([]); ax.set_yticks([])

    fig2.suptitle(f"Extra filter sample — top: PASS ({all_pass.sum()}/{len(jsons)}), bottom: REJECT",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    out2 = os.path.join(args.output_root, f"{args.tag}_samples.png")
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
