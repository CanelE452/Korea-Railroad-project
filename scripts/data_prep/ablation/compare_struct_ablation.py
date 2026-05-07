"""Compare v8_A / v8_B2 / v8_B3 on noapril 188 with B∧C filter metric.

목적: Structural loss ablation의 최종 판정.
  - PnP rate
  - filter B pass
  - filter C pass
  - B ∧ C pass

사용법:
    python scripts/data_prep/compare_struct_ablation.py
"""

import os, sys, glob
import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "self_training"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix
from canonical_filters import filter_B, filter_C


MODELS = [
    ("v8_A",  "weights/v9_ablation_A_coord/final_net_epoch_0065.pth"),
    ("v8_B2", "weights/v9_ablation_B2_coord_edge/final_net_epoch_0065.pth"),
    ("v8_B3", "weights/v9_ablation_B3_coord_edge_small/final_net_epoch_0065.pth"),
]

NOAPRIL_RGB = os.path.join(ROOT, "data/pallet/raw_data/capture0403noapril/rgb")
NOAPRIL_ALL_GLOB = os.path.join(NOAPRIL_RGB, "*.png")

# Camera intrinsics for noapril capture
FX, FY, CX, CY = 614.18, 614.31, 329.28, 234.53


def load_model(weights_path, device):
    net = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device).eval()
    return net


def extract_kps(belief, threshold=0.3):
    OFFSET, WIN = 0.4395, 11
    RAN = WIN // 2
    kps = []
    for i in range(belief.shape[0]):
        bmap = belief[i]
        mx = bmap.max()
        if mx < threshold:
            kps.append((-1.0, -1.0, float(mx)))
            continue
        sm = gaussian_filter(bmap, sigma=2)
        pl = np.zeros_like(sm); pl[1:, :] = sm[:-1, :]
        pr = np.zeros_like(sm); pr[:-1, :] = sm[1:, :]
        pu = np.zeros_like(sm); pu[:, 1:] = sm[:, :-1]
        pd = np.zeros_like(sm); pd[:, :-1] = sm[:, 1:]
        peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            kps.append((-1.0, -1.0, float(mx)))
            continue
        vals = [bmap[yy, xx] for yy, xx in zip(ys, xs)]
        bi = int(np.argmax(vals))
        px, py = int(xs[bi]), int(ys[bi])
        yl = max(0, py - RAN); yh = min(bmap.shape[0], py + RAN + 1)
        xl = max(0, px - RAN); xh = min(bmap.shape[1], px + RAN + 1)
        patch = bmap[yl:yh, xl:xh]
        if patch.sum() > 0:
            yg, xg = np.meshgrid(np.arange(yl, yh), np.arange(xl, xh), indexing="ij")
            wx = float(np.average(xg, weights=patch)) + OFFSET
            wy = float(np.average(yg, weights=patch)) + OFFSET
        else:
            wx, wy = float(px), float(py)
        kps.append((wx, wy, float(mx)))
    return kps


def run_inference(net, img_bgr, device, threshold):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (448, 448))
    normed = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ob, _ = net(tensor)
    belief = ob[-1][0].cpu().numpy()
    kps_bel = extract_kps(belief, threshold)
    h, w = img_bgr.shape[:2]
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w, bh / h
    kps_orig = []
    for kp in kps_bel:
        if kp[0] < 0:
            kps_orig.append(None)
        else:
            kps_orig.append((kp[0] / sx, kp[1] / sy))
    return kps_orig


def eval_model(tag, weights_path, img_paths, device):
    net = load_model(os.path.join(ROOT, weights_path), device)
    cam = make_camera_matrix(FX, FY, CX, CY)
    pnp = PalletPnPSolver(cam)

    n = len(img_paths)
    det_counts = []
    pnp_ok = 0
    b_pass = 0
    c_pass = 0
    bc_pass = 0
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        kps_orig = run_inference(net, img, device, threshold=0.3)
        det_counts.append(sum(1 for k in kps_orig if k is not None))
        success, R, t, _ = pnp.solve(kps_orig)
        if not success:
            continue
        pnp_ok += 1
        fB, _ = filter_B(kps_orig, pnp, R, t,
                         tau_span=0.35, tau_end=0.10, tau_nc=0.02,
                         img_size=(img.shape[1], img.shape[0]))
        fC, _ = filter_C(kps_orig, pnp, R, t, tau_C=0.05)
        if fB: b_pass += 1
        if fC: c_pass += 1
        if fB and fC: bc_pass += 1

    del net
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "n": n,
        "avg_det": float(np.mean(det_counts)) if det_counts else 0.0,
        "pnp_ok": pnp_ok,
        "pnp_rate": pnp_ok / max(n, 1),
        "b_pass": b_pass,
        "c_pass": c_pass,
        "bc_pass": bc_pass,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use ALL 188 noapril images
    img_paths = sorted(glob.glob(NOAPRIL_ALL_GLOB))
    print(f"Found {len(img_paths)} noapril images")

    results = []
    for tag, wp in MODELS:
        print(f"\n--- evaluating {tag} ---")
        r = eval_model(tag, wp, img_paths, device)
        results.append(r)
        print(f"  avg_det={r['avg_det']:.2f}  pnp={r['pnp_ok']}/{r['n']} ({r['pnp_rate']*100:.1f}%)  "
              f"B={r['b_pass']}  C={r['c_pass']}  B∧C={r['bc_pass']}")

    print()
    print("=" * 72)
    print(" Structural Loss Ablation — noapril 188 evaluation")
    print("=" * 72)
    print(f"  {'model':<8}{'avg_kp':>8}{'PnP':>10}{'B':>6}{'C':>6}{'B∧C':>6}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['tag']:<8}{r['avg_det']:>8.2f}"
              f"{r['pnp_ok']:>6}/{r['n']:<3}{r['b_pass']:>6}{r['c_pass']:>6}{r['bc_pass']:>6}")
    print("=" * 72)


if __name__ == "__main__":
    main()
