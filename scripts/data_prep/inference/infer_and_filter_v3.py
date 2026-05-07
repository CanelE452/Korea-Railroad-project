"""DOPE 추론 + Filter 비교 (current vs weighted).

v8_A, v8_E 등 여러 모델을 같은 이미지에 대해
current filter / weighted PnP / weighted C로 비교.

사용법:
    python scripts/data_prep/infer_and_filter_v3.py \
        --weights weights/v8_ablation_E_rel/final_net_epoch_0065.pth \
        --img_dir data/pallet/raw_data/capture0403noapril/rgb \
        --output_dir data/pallet/eval_results/v3_compare \
        --weighted
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # data_prep for shared libs

from visualize_inference import load_model, extract_keypoints, draw_overlay
from pnp_solver import PalletPnPSolver, make_camera_matrix
from canonical_filters import filter_B, filter_C, filter_D


def infer_with_sigma(model, img_bgr, device, temperature=1.0):
    """DOPE 추론 → belief maps + per-keypoint sigma."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out_bel, _ = model(tensor)
        belief_tensor = out_bel[-1][0, :9]  # (9, H, W)

        # Compute sigma from heatmap
        B, C, H, W = 1, 9, belief_tensor.shape[1], belief_tensor.shape[2]
        hm = belief_tensor.unsqueeze(0)  # (1, 9, H, W)

        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

        flat = hm.view(1, C, -1)
        weights = F.softmax(flat / temperature, dim=-1).view(1, C, H, W)

        mu_x = (weights * x_coords).sum(dim=(2, 3))  # (1, 9)
        mu_y = (weights * y_coords).sum(dim=(2, 3))

        var = (weights * ((x_coords - mu_x.unsqueeze(-1).unsqueeze(-1)) ** 2 +
                          (y_coords - mu_y.unsqueeze(-1).unsqueeze(-1)) ** 2)).sum(dim=(2, 3))
        sigma = torch.sqrt(var.clamp(min=1e-6))[0]  # (9,)

    belief_np = out_bel[-1][0].cpu().numpy()
    sigma_np = sigma.cpu().numpy()  # (9,)
    return belief_np, sigma_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    parser.add_argument("--tau_span", type=float, default=0.35)
    parser.add_argument("--tau_end", type=float, default=0.10)
    parser.add_argument("--tau_nc", type=float, default=0.02)
    parser.add_argument("--tau_C", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--weighted", action="store_true",
                        help="Also evaluate weighted PnP + weighted C")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    os.makedirs(args.output_dir, exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(args.img_dir, "*.jpg")) +
        glob.glob(os.path.join(args.img_dir, "*.png"))
    )
    print(f"Model: {args.weights}")
    print(f"Images: {len(imgs)}")
    print(f"Weighted mode: {args.weighted}")

    # Stats for 4 modes
    modes = ["current", "weighted_C", "weighted_PnP_C"]
    stats = {}
    for m in modes:
        stats[m] = {k: 0 for k in ["pnp_ok", "B", "C", "BC"]}
    stats["total"] = 0

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            continue

        belief, sigma = infer_with_sigma(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)
        detected = sum(1 for kp in pred_kps if kp is not None)

        h, w = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = w / bw, h / bh
        pred_orig = []
        sigmas_orig = []
        for ki, kp in enumerate(pred_kps):
            if kp is None:
                pred_orig.append(None)
                sigmas_orig.append(None)
            else:
                pred_orig.append((kp[0] * sx, kp[1] * sy))
                sigmas_orig.append(float(sigma[ki]))

        stats["total"] += 1

        # === Mode 1: Current (unweighted) ===
        success, R, t, _ = pnp.solve(pred_orig)
        if success:
            stats["current"]["pnp_ok"] += 1
            fB, _ = filter_B(pred_orig, pnp, R, t,
                             args.tau_span, args.tau_end, args.tau_nc,
                             img_size=(w, h))
            fC, _ = filter_C(pred_orig, pnp, R, t, args.tau_C)
            if fB:
                stats["current"]["B"] += 1
            if fC:
                stats["current"]["C"] += 1
            if fB and fC:
                stats["current"]["BC"] += 1

        if args.weighted:
            # === Mode 2: Weighted C only (same PnP, weighted filter) ===
            if success:
                stats["weighted_C"]["pnp_ok"] += 1
                # Weighted C: pass sigmas to filter_C for weighted LOO
                fB_wc, _ = filter_B(pred_orig, pnp, R, t,
                                     args.tau_span, args.tau_end, args.tau_nc,
                                     img_size=(w, h))
                fC_wc, _ = filter_C(pred_orig, pnp, R, t, args.tau_C,
                                     sigmas=sigmas_orig)
                if fB_wc:
                    stats["weighted_C"]["B"] += 1
                if fC_wc:
                    stats["weighted_C"]["C"] += 1
                if fB_wc and fC_wc:
                    stats["weighted_C"]["BC"] += 1

            # === Mode 3: Weighted PnP + Weighted C ===
            success_w, R_w, t_w, _ = pnp.solve(pred_orig, sigmas=sigmas_orig)
            if success_w:
                stats["weighted_PnP_C"]["pnp_ok"] += 1
                fB_w, _ = filter_B(pred_orig, pnp, R_w, t_w,
                                    args.tau_span, args.tau_end, args.tau_nc,
                                    img_size=(w, h))
                fC_w, _ = filter_C(pred_orig, pnp, R_w, t_w, args.tau_C,
                                    sigmas=sigmas_orig)
                if fB_w:
                    stats["weighted_PnP_C"]["B"] += 1
                if fC_w:
                    stats["weighted_PnP_C"]["C"] += 1
                if fB_w and fC_w:
                    stats["weighted_PnP_C"]["BC"] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(imgs):
            print(f"  [{i+1}/{len(imgs)}]")

    # Print comparison
    tn = max(stats["total"], 1)
    print(f"\n{'='*70}")
    print(f"Total images: {stats['total']}")
    print(f"{'Mode':<25} {'PnP':>6} {'B':>6} {'C':>6} {'B∧C':>6}")
    print(f"{'-'*70}")
    for m in modes:
        if not args.weighted and m != "current":
            continue
        s = stats[m]
        print(f"{m:<25} {s['pnp_ok']:>6} {s['B']:>6} {s['C']:>6} {s['BC']:>6}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
