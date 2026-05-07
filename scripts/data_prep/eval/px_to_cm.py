"""20px 가 몇 cm 인지 계산 — middle 440 장 전체 기준.

Pallet 실물 long side = 110 cm (KS T-11형 1100 mm).
projected_cuboid 의 0→4 edge (long axis) 를 픽셀로 재서
px_per_cm 분포를 구한다.
"""

import json
from pathlib import Path
import numpy as np

GT_DIR = Path("data/pallet/raw_data/capture0403middle/gt_final_isaac")
PALLET_LONG_CM = 110.0  # 1100 mm

# Corner convention (Y=UP):
#   0→1 = X (medium, ~100cm)
#   0→3 = Y (height, ~15cm)
#   0→4 = Z (long, ~120cm)
# long side = 0→4
EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]  # 4 parallel long edges

def long_edge_pixels(cuboid):
    """4 개의 long edge 평균 pixel length."""
    c = np.array(cuboid)  # (9, 2) or (8,2)
    lens = [np.linalg.norm(c[j] - c[i]) for i, j in EDGES]
    return float(np.mean(lens))

def main():
    pxs = []
    for p in sorted(GT_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        objs = data.get("objects", [])
        if not objs:
            continue
        cub = objs[0].get("projected_cuboid")
        if cub is None or len(cub) < 8:
            continue
        pxs.append(long_edge_pixels(cub))

    pxs = np.array(pxs)
    px_per_cm = pxs / PALLET_LONG_CM
    cm_per_20px = 20.0 / px_per_cm

    print(f"Frames evaluated: {len(pxs)}")
    print(f"Long edge pixels   : mean={pxs.mean():.1f}  median={np.median(pxs):.1f}  min={pxs.min():.1f}  max={pxs.max():.1f}")
    print(f"px / cm            : mean={px_per_cm.mean():.3f}  median={np.median(px_per_cm):.3f}  min={px_per_cm.min():.3f}  max={px_per_cm.max():.3f}")
    print(f"20 px in cm        : mean={cm_per_20px.mean():.2f}  median={np.median(cm_per_20px):.2f}  min={cm_per_20px.min():.2f}  max={cm_per_20px.max():.2f}")
    print()
    print("Percentile (20px in cm):")
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q:02d} = {np.percentile(cm_per_20px, q):.2f} cm")

if __name__ == "__main__":
    main()
