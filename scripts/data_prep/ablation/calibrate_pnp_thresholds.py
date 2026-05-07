"""Calibrate reprojection-guided PnP refinement thresholds.

PnP 성공한 전체 샘플에서 per-keypoint normalized residual 및 peak confidence 분포를 수집하여
tau_huber, tau_peak, tau_w 추천값을 산출.

사용법:
    python scripts/data_prep/calibrate_pnp_thresholds.py \
        --weights weights/v9_ablation_A_coord/final_net_epoch_0065.pth \
        --img_dir data/pallet/raw_data/capture0403noapril/noapril_calib \
        --output data/pallet/eval_results/calibration_results.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # data_prep for shared libs

from visualize_inference import load_model, extract_keypoints
from pnp_solver import PalletPnPSolver, make_camera_matrix
from canonical_filters import filter_B, filter_C


def collect_images(img_dir):
    """img_dir 또는 filelist.txt 기반으로 이미지 경로 수집."""
    filelist = os.path.join(img_dir, "filelist.txt")
    if os.path.exists(filelist):
        with open(filelist) as f:
            names = [l.strip() for l in f if l.strip()]
        # filelist의 이미지는 상위 rgb/ 디렉토리에 있음
        rgb_dir = os.path.join(os.path.dirname(img_dir), "rgb")
        if os.path.isdir(rgb_dir):
            return sorted([os.path.join(rgb_dir, n) for n in names])
        return sorted([os.path.join(img_dir, n) for n in names])
    return sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate PnP refinement thresholds")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output", default="data/pallet/eval_results/calibration_results.json")
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Belief map peak threshold")
    parser.add_argument("--tau_span", type=float, default=0.35)
    parser.add_argument("--tau_end", type=float, default=0.10)
    parser.add_argument("--tau_nc", type=float, default=0.02)
    parser.add_argument("--tau_C", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    imgs = collect_images(args.img_dir)
    print(f"Model: {args.weights}")
    print(f"Images: {len(imgs)}")

    # Collect per-keypoint data from PnP-success samples
    all_residuals = []       # normalized residual u_i
    all_confidences = []     # peak confidence c_i
    bc_residuals = []        # residuals from B∧C pass samples
    bc_confidences = []

    n_pnp_ok = 0
    n_bc_pass = 0

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            continue

        # Inference
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (448, 448))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()
        pred_kps = extract_keypoints(belief, args.threshold)

        # Scale to original image resolution
        h, w = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = w / bw, h / bh
        pred_orig = []
        peak_confs = []
        for kp in pred_kps:
            if kp is None:
                pred_orig.append(None)
                peak_confs.append(0.0)
            else:
                pred_orig.append((kp[0] * sx, kp[1] * sy))
                peak_confs.append(float(kp[2]))

        # PnP
        success, R, t, _ = pnp.solve(pred_orig)
        if not success:
            continue

        n_pnp_ok += 1

        # Compute normalized residuals
        reproj = pnp.reproject(R, t)
        D = PalletPnPSolver._projected_diagonal(reproj[:8])
        if D < 1e-6:
            continue

        img_residuals = []
        img_confs = []
        for ki in range(9):
            if ki >= len(pred_orig) or pred_orig[ki] is None:
                continue
            u, v = pred_orig[ki][:2]
            r = np.linalg.norm(reproj[ki] - np.array([u, v]))
            u_norm = r / (D + 1e-6)
            img_residuals.append(u_norm)
            img_confs.append(peak_confs[ki])

        all_residuals.extend(img_residuals)
        all_confidences.extend(img_confs)

        # Check B∧C
        fB, _ = filter_B(pred_orig, pnp, R, t,
                         args.tau_span, args.tau_end, args.tau_nc,
                         img_size=(w, h))
        fC, _ = filter_C(pred_orig, pnp, R, t, args.tau_C)
        if fB and fC:
            n_bc_pass += 1
            bc_residuals.extend(img_residuals)
            bc_confidences.extend(img_confs)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(imgs)}] PnP OK: {n_pnp_ok}, B∧C: {n_bc_pass}")

    # Compute percentiles
    res_arr = np.array(all_residuals) if all_residuals else np.array([0.0])
    conf_arr = np.array(all_confidences) if all_confidences else np.array([0.0])
    percentiles = [10, 20, 25, 50, 75, 80, 90, 95]

    res_pct = {f"p{p}": float(np.percentile(res_arr, p)) for p in percentiles}
    conf_pct = {f"p{p}": float(np.percentile(conf_arr, p)) for p in percentiles}

    # Huber weight distribution: w = c * psi(u) where psi(u) = min(1, tau/u)
    # Use p75 residual as preliminary tau_huber to show weight distribution
    prelim_tau = res_pct["p75"]
    weights = []
    for u, c in zip(all_residuals, all_confidences):
        psi = 1.0 if u <= prelim_tau else prelim_tau / (u + 1e-8)
        weights.append(c * psi)
    w_arr = np.array(weights) if weights else np.array([0.0])
    w_pct = {f"p{p}": float(np.percentile(w_arr, p)) for p in percentiles}

    # Recommended thresholds
    recommended = {
        "tau_huber": res_pct["p75"],
        "tau_peak": conf_pct["p10"],
        "tau_w": w_pct["p20"],
    }

    result = {
        "n_images": len(imgs),
        "n_pnp_ok": n_pnp_ok,
        "n_bc_pass": n_bc_pass,
        "n_keypoints_analyzed": len(all_residuals),
        "n_keypoints_bc": len(bc_residuals),
        "residual_percentiles": res_pct,
        "confidence_percentiles": conf_pct,
        "weight_percentiles_at_p75_tau": w_pct,
        "recommended": recommended,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Calibration Results ({len(imgs)} images)")
    print(f"{'='*60}")
    print(f"  PnP success: {n_pnp_ok}/{len(imgs)}")
    print(f"  B∧C pass:    {n_bc_pass}/{len(imgs)}")
    print(f"  Keypoints:   {len(all_residuals)} (PnP-success), {len(bc_residuals)} (B∧C)")
    print(f"\n  Residual (normalized by D):")
    for k, v in res_pct.items():
        print(f"    {k}: {v:.5f}")
    print(f"\n  Peak confidence:")
    for k, v in conf_pct.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Huber weight (tau={prelim_tau:.5f}):")
    for k, v in w_pct.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Recommended thresholds:")
    print(f"    tau_huber = {recommended['tau_huber']:.5f}")
    print(f"    tau_peak  = {recommended['tau_peak']:.4f}")
    print(f"    tau_w     = {recommended['tau_w']:.4f}")
    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.output}")

    # Optional: save histogram data for plotting
    hist_path = args.output.replace(".json", "_histdata.npz")
    np.savez(hist_path,
             residuals=res_arr, confidences=conf_arr, weights=w_arr,
             bc_residuals=np.array(bc_residuals) if bc_residuals else np.array([]),
             bc_confidences=np.array(bc_confidences) if bc_confidences else np.array([]))
    print(f"Histogram data: {hist_path}")


if __name__ == "__main__":
    main()
