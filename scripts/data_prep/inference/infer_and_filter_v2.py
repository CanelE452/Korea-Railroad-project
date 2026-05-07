"""DOPE 추론 + Canonical Geometric Filter (A/B/C) v2.

무차원 비율 기반 필터 — 데이터셋/해상도/카메라 거리 불변.

Core filters:
  A: Normalized flip-equivariant consistency (s_A < tau_A)
  B: Structural coverage + non-collinearity (s_B > tau_cov, r_B > tau_ani)
  C: Normalized LOO PnP stability (s_C < tau_C)

Optional priors:
  D1: Depth range
  D2: Tilt range

사용법:
    python scripts/data_prep/infer_and_filter_v2.py \
        --weights weights/mixed_v8/final_net_epoch_0060.pth \
        --img_dir data/pallet/raw_data/capture0403noapril/rgb \
        --output_dir data/pallet/eval_results/test_canonical
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # data_prep for shared libs

from visualize_inference import load_model, infer, extract_keypoints, draw_overlay
from pnp_solver import PalletPnPSolver, make_camera_matrix
from canonical_filters import filter_A, filter_B, filter_C, filter_D, prior_depth, prior_tilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    parser.add_argument("--tau_A", type=float, default=0.05)
    parser.add_argument("--tau_span", type=float, default=0.35)
    parser.add_argument("--tau_end", type=float, default=0.10)
    parser.add_argument("--tau_nc", type=float, default=0.02)
    parser.add_argument("--tau_C", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    dirs = {}
    for name in ["all", "pnp_ok", "A", "B", "C", "D", "BC", "BC_priors"]:
        dirs[name] = os.path.join(args.output_dir, name)
        os.makedirs(dirs[name], exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(args.img_dir, "*.jpg")) +
        glob.glob(os.path.join(args.img_dir, "*.png"))
    )
    print(f"Model: {args.weights} (device: {device})")
    print(f"Images: {len(imgs)}")
    print(f"Thresholds: tau_span={args.tau_span}, tau_end={args.tau_end}, tau_nc={args.tau_nc}, tau_C={args.tau_C}")

    csv_path = os.path.join(args.output_dir, "filter_details.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", "n_kps", "pnp_ok",
        "fA", "sA_score", "fB", "span", "d_left", "d_right", "nc", "fC", "sC",
        "depth_ok", "depth", "tilt_ok", "tilt",
        "BC", "BC_priors"
    ])

    stats = {k: 0 for k in ["total", "pnp_ok", "A", "B", "C", "D", "BC", "BC_priors"]}

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            continue

        belief = infer(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)
        detected = sum(1 for kp in pred_kps if kp is not None)

        h, w = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = w / bw, h / bh
        pred_orig = []
        for kp in pred_kps:
            if kp is None:
                pred_orig.append(None)
            else:
                pred_orig.append((kp[0] * sx, kp[1] * sy))

        success, R, t, _ = pnp.solve(pred_orig)
        basename = os.path.splitext(os.path.basename(path))[0]

        vis = draw_overlay(img, pred_kps, None, belief, pnp,
                           f"{basename} | {detected}/9")
        cv2.imwrite(os.path.join(dirs["all"], f"{basename}_overlay.jpg"), vis)

        stats["total"] += 1
        pnp_ok = bool(success)
        fA, sA_score = False, float("inf")
        fB, b_details = False, {"span": 0, "d_left": 1, "d_right": 1, "nc": 0}
        fC, sC = False, float("inf")
        depth_ok, depth_val = False, 0
        tilt_ok, tilt_val = False, 90

        if pnp_ok:
            stats["pnp_ok"] += 1
            cv2.imwrite(os.path.join(dirs["pnp_ok"], f"{basename}_overlay.jpg"), vis)

            # A: auxiliary score (not in core gate)
            fA, sA_score = filter_A(model, img, pred_kps, device, pnp, R, t, args.tau_A)
            if fA:
                stats["A"] += 1
                cv2.imwrite(os.path.join(dirs["A"], f"{basename}_overlay.jpg"), vis)

            # B: Visible Structural Support (core)
            fB, b_details = filter_B(pred_orig, pnp, R, t,
                                     args.tau_span, args.tau_end, args.tau_nc,
                                     img_size=(w, h))
            if fB:
                stats["B"] += 1
                cv2.imwrite(os.path.join(dirs["B"], f"{basename}_overlay.jpg"), vis)

            # C: Normalized LOO PnP Stability (core)
            fC, sC = filter_C(pred_orig, pnp, R, t, args.tau_C)
            if fC:
                stats["C"] += 1
                cv2.imwrite(os.path.join(dirs["C"], f"{basename}_overlay.jpg"), vis)

            # D: Conditional Diagonal Incidence
            fD, sD, nD = filter_D(pred_orig, pnp, R, t)
            if fD:
                stats["D"] += 1

            # Optional priors
            depth_ok, depth_val = prior_depth(t)
            tilt_ok, tilt_val = prior_tilt(R)

            # Core gate: B ∧ C
            if fB and fC:
                stats["BC"] += 1
                cv2.imwrite(os.path.join(dirs["BC"], f"{basename}_overlay.jpg"), vis)

            if fB and fC and depth_ok and tilt_ok:
                stats["BC_priors"] += 1
                cv2.imwrite(os.path.join(dirs["BC_priors"], f"{basename}_overlay.jpg"), vis)

        bc_passed = pnp_ok and fB and fC
        bc_priors = bc_passed and depth_ok and tilt_ok
        writer.writerow([
            basename, detected, pnp_ok,
            fA, f"{sA_score:.4f}", fB,
            f"{b_details['span']:.4f}", f"{b_details['d_left']:.4f}", f"{b_details['d_right']:.4f}", f"{b_details['nc']:.4f}",
            fC, f"{sC:.4f}",
            depth_ok, f"{depth_val:.2f}", tilt_ok, f"{tilt_val:.1f}",
            bc_passed, bc_priors,
        ])

        if (i + 1) % 100 == 0 or (i + 1) == len(imgs):
            pn = max(stats["pnp_ok"], 1)
            print(f"  [{i+1}/{len(imgs)}] pnp={stats['pnp_ok']} "
                  f"B={stats['B']} C={stats['C']} "
                  f"BC={stats['BC']} BC+P={stats['BC_priors']}")

    csv_file.close()

    pn = max(stats["pnp_ok"], 1)
    tn = max(stats["total"], 1)
    print(f"\n{'='*60}")
    print(f"Total images:      {stats['total']}")
    print(f"PnP OK:            {stats['pnp_ok']} ({100*stats['pnp_ok']/tn:.1f}%)")
    print(f"Filter A passed:   {stats['A']:>4} ({100*stats['A']/pn:.1f}% of PnP) [Flip Consistency, tau={args.tau_A}]")
    print(f"Filter B passed:   {stats['B']:>4} ({100*stats['B']/pn:.1f}% of PnP) [Span + Endpoint + Non-Collinearity]")
    print(f"Filter C passed:   {stats['C']:>4} ({100*stats['C']/pn:.1f}% of PnP) [Normalized LOO PnP Stability]")
    print(f"Filter D passed:   {stats['D']:>4} ({100*stats['D']/pn:.1f}% of PnP) [Conditional Diagonal Incidence]")
    print(f"---")
    print(f"B∧C passed:        {stats['BC']:>4} ({100*stats['BC']/tn:.1f}% of total) [core]")
    print(f"B∧C+Priors passed: {stats['BC_priors']:>4} ({100*stats['BC_priors']/tn:.1f}% of total)")
    print(f"\nDetails: {csv_path}")
    print(f"Output:  {args.output_dir}/")


if __name__ == "__main__":
    main()
