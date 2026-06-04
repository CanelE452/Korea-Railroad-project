"""diag_pass_contact_sheet.py — per-domain contact sheet of diag-PASS pseudo-labels.

Goal: show what the diag filter actually LET THROUGH, per domain.
  - PL (predicted 9 keypoints + cuboid wireframe) is the MAIN overlay.
  - GT cuboid drawn faintly for comparison only.
  - sorted by reproj ascending (best PL first) so the user sees the quality
    distribution; passed-but-slightly-skewed frames stay in the grid as-is.
Reuses _full_{tag}.json from filter_domain_analysis.py (no inference).
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
# distinct color per keypoint index (BGR)
KP_COLORS = [
    (60, 60, 255), (60, 160, 255), (60, 255, 255), (60, 255, 60),
    (255, 255, 60), (255, 160, 60), (255, 60, 60), (255, 60, 200),
    (255, 255, 255),  # centroid (8)
]
PL_LINE = (0, 235, 235)     # cuboid wireframe (yellow-cyan) — PL main
GT_LINE = (90, 200, 90)     # GT cuboid, faint green
DOMAINS = ["outside", "night", "indoor"]
EXCLUDE_FILE = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")


def load_exclude():
    ids = set()
    if os.path.exists(EXCLUDE_FILE):
        for ln in open(EXCLUDE_FILE):
            ln = ln.split("#")[0].strip()
            if ln:
                ids.add(ln)
    return ids


def draw_cell(rec, good_thresh):
    img = cv2.imread(rec["img"])
    if img is None:
        return None
    h, w = img.shape[:2]

    # --- GT cuboid: faint, thin, for comparison only ---
    gt8 = np.array(rec["gt8"], float)
    for a, b in EDGES:
        cv2.line(img, tuple(gt8[a].astype(int)), tuple(gt8[b].astype(int)),
                 GT_LINE, 1, cv2.LINE_AA)

    # --- PL (prediction) = MAIN ---
    pts = [None if p is None else np.array(p, float) for p in rec["kp"]]
    for a, b in EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     PL_LINE, 2, cv2.LINE_AA)
    for i in range(9):
        if pts[i] is None:
            continue
        c = tuple(int(x) for x in KP_COLORS[i])
        p = tuple(pts[i].astype(int))
        r = 6 if i == 8 else 5
        cv2.circle(img, p, r, c, -1)
        cv2.circle(img, p, r, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    # --- header banner ---
    mm = rec.get("mean_match_px")
    good = bool(mm is not None and mm < good_thresh)
    banner_h = 30
    bar = img[:banner_h].copy()
    cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.35, img[:banner_h], 0.65, 0, img[:banner_h])
    gcol = (0, 230, 0) if good else (0, 170, 255)
    txt = f"reproj={mm}px  {'GOOD' if good else 'pass'}  ndet={rec['n_detected']}"
    cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                gcol, 2, cv2.LINE_AA)
    return img


def make_sheet(cells, cols, tile_w=520):
    if not cells:
        return None
    n = len(cells)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    scaled = []
    for c in cells:
        h, w = c.shape[:2]
        s = tile_w / w
        scaled.append(cv2.resize(c, (tile_w, int(h * s))))
    tile_h = max(im.shape[0] for im in scaled)
    pad = 6
    sheet = np.full((rows * tile_h + pad * (rows + 1),
                     cols * tile_w + pad * (cols + 1), 3), 30, np.uint8)
    for idx, im in enumerate(scaled):
        r, cc = divmod(idx, cols)
        y = pad + r * (tile_h + pad)
        x = pad + cc * (tile_w + pad)
        sheet[y:y + im.shape[0], x:x + tile_w] = im
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--in_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "filter_domain_analysis"))
    ap.add_argument("--good_thresh_px", type=float, default=10.0)
    ap.add_argument("--max_per_domain", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    full = json.load(open(os.path.join(args.in_dir, f"_full_{args.tag}.json")))
    out_dir = os.path.join(args.in_dir, "diag_pass_overlays")
    os.makedirs(out_dir, exist_ok=True)
    exclude = load_exclude()

    saved, report = [], {}
    for dom in DOMAINS:
        rows = full.get(dom, [])
        passed = [r for r in rows
                  if r["filters"]["diag"] and r["frame"] not in exclude]
        # best (lowest reproj) first; None reproj sinks to the end
        passed.sort(key=lambda r: (r["mean_match_px"] is None,
                                    r["mean_match_px"] or 1e9))
        sel = passed[:args.max_per_domain]

        reprojs = [r["mean_match_px"] for r in passed
                   if r["mean_match_px"] is not None]
        n_good = sum(1 for r in passed if r["mean_match_px"] is not None
                     and r["mean_match_px"] < args.good_thresh_px)
        report[dom] = {
            "diag_pass": len(passed),
            "good": n_good,
            "median_reproj": (round(float(np.median(reprojs)), 2)
                              if reprojs else None),
            "shown": len(sel),
            "reproj_range": ([round(min(reprojs), 2), round(max(reprojs), 2)]
                             if reprojs else None),
        }

        cells = [c for c in (draw_cell(r, args.good_thresh_px) for r in sel)
                 if c is not None]
        sheet = make_sheet(cells, args.cols)
        if sheet is None:
            continue
        # title strip
        title = (f"{dom.upper()}  diag-PASS pseudo-labels  "
                 f"N={len(passed)}  good={n_good}  "
                 f"med_reproj={report[dom]['median_reproj']}px   "
                 f"[PL=bright cuboid+colored 0-8 | GT=faint green]")
        strip = np.full((38, sheet.shape[1], 3), 20, np.uint8)
        cv2.putText(strip, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 235, 235), 2, cv2.LINE_AA)
        sheet = np.vstack([strip, sheet])
        fn = os.path.join(out_dir, f"{dom}_diag_pass.png")
        cv2.imwrite(fn, sheet)
        saved.append(fn)

    print("[exclude]", exclude)
    print(json.dumps(report, indent=2))
    for s in saved:
        print("[save]", s)


if __name__ == "__main__":
    main()
