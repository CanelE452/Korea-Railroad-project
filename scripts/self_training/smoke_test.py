"""Stage 3 Smoke Test — inference → filter → pseudo-label → finetune 1 epoch.

사용법:
    python scripts/self_training/smoke_test.py \
        --weights weights/mixed_v1/net_epoch_0060.pth \
        --real_dir data/pallet/real_data/real_dev \
        --syn_dir data/pallet/training_data/mixed_v1_train \
        --output_dir weights/_smoke_test
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.dirname(__file__))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d
from geometric_filter import GeometricFilter


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_keypoints_from_belief(belief_maps, threshold=0.3):
    OFFSET = 0.4395
    WIN = 11
    RAN = WIN // 2
    keypoints = []

    for i in range(belief_maps.shape[0]):
        bmap_ori = belief_maps[i]
        max_val = bmap_ori.max()
        if max_val < threshold:
            keypoints.append((-1, -1, float(max_val)))
            continue

        bmap_smooth = gaussian_filter(bmap_ori, sigma=2)
        p = 1
        pad_l = np.zeros_like(bmap_smooth); pad_l[p:, :] = bmap_smooth[:-p, :]
        pad_r = np.zeros_like(bmap_smooth); pad_r[:-p, :] = bmap_smooth[p:, :]
        pad_u = np.zeros_like(bmap_smooth); pad_u[:, p:] = bmap_smooth[:, :-p]
        pad_d = np.zeros_like(bmap_smooth); pad_d[:, :-p] = bmap_smooth[:, p:]

        peaks_binary = (
            (bmap_smooth >= pad_l) & (bmap_smooth >= pad_r) &
            (bmap_smooth >= pad_u) & (bmap_smooth >= pad_d) &
            (bmap_smooth > threshold)
        )

        peak_ys, peak_xs = np.nonzero(peaks_binary)
        if len(peak_xs) == 0:
            keypoints.append((-1, -1, float(max_val)))
            continue

        peak_vals = [bmap_ori[py, px] for py, px in zip(peak_ys, peak_xs)]
        best_idx = np.argmax(peak_vals)
        px, py = int(peak_xs[best_idx]), int(peak_ys[best_idx])

        y_lo = max(0, py - RAN); y_hi = min(bmap_ori.shape[0], py + RAN + 1)
        x_lo = max(0, px - RAN); x_hi = min(bmap_ori.shape[1], px + RAN + 1)
        patch = bmap_ori[y_lo:y_hi, x_lo:x_hi]

        if patch.sum() > 0:
            ys = np.arange(y_lo, y_hi)
            xs = np.arange(x_lo, x_hi)
            xg, yg = np.meshgrid(xs, ys)
            wx = np.average(xg, weights=patch) + OFFSET
            wy = np.average(yg, weights=patch) + OFFSET
        else:
            wx, wy = float(px), float(py)

        keypoints.append((wx, wy, float(max_val)))
    return keypoints


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Smoke Test")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--syn_dir", default="data/pallet/training_data/mixed_v1_train")
    parser.add_argument("--output_dir", default="weights/_smoke_test")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pseudo_dir = os.path.join(args.output_dir, "pseudo_labels")
    os.makedirs(pseudo_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_matrix = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp_solver = PalletPnPSolver(cam_matrix)
    # Relaxed thresholds for real data (legacy filter smoke test)
    filter_config = {
        "tau_reproj": 12.0,
        "tau_ratio_min": 0.4,
        "tau_ratio_max": 2.5,
        "tau_angle_min": 10.0,
        "tau_angle_max": 175.0,
        "tau_size_min": 0.5,
        "tau_size_max": 2.5,
        "min_keypoints": 5,
    }
    geo_filter = GeometricFilter(pnp_solver, filter_config)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # ========== Step 1: Inference ==========
    print("=" * 60)
    print(" Step 1: Inference on real_dev")
    print("=" * 60)

    model = load_model(args.weights, device)
    real_imgs = sorted(glob.glob(os.path.join(args.real_dir, "*.jpg")))
    print(f"  Real images: {len(real_imgs)}")

    inference_results = []

    for img_path in real_imgs:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (448, 448))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out_bel, out_aff = model(tensor)

        belief = out_bel[-1][0].cpu().numpy()
        pred_kps = extract_keypoints_from_belief(belief, args.threshold)

        # Scale to original image coords
        h_orig, w_orig = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = bw / w_orig, bh / h_orig

        pred_kps_orig = []
        n_detected = 0
        for kp in pred_kps:
            if kp[0] < 0:
                pred_kps_orig.append(None)
            else:
                pred_kps_orig.append((float(kp[0]) / sx, float(kp[1]) / sy))
                n_detected += 1

        # PnP
        success, R, t, inliers = pnp_solver.solve(pred_kps_orig)

        # Belief map stats
        peak_vals = [kp[2] for kp in pred_kps if kp[0] >= 0]
        avg_peak = np.mean(peak_vals) if peak_vals else 0

        result = {
            "basename": basename,
            "n_detected": n_detected,
            "pnp_ok": success,
            "avg_peak": avg_peak,
            "keypoints_2d": pred_kps_orig,
            "R": R,
            "t": t,
            "img_path": img_path,
        }
        inference_results.append(result)

        status = f"PnP {'OK' if success else 'FAIL'}"
        print(f"  {basename}: {n_detected}/9 kps, peak={avg_peak:.3f}, {status}")

    pnp_ok_count = sum(1 for r in inference_results if r["pnp_ok"])
    print(f"\n  Summary: PnP OK {pnp_ok_count}/{len(inference_results)} ({pnp_ok_count/max(len(inference_results),1)*100:.0f}%)")

    # ========== Step 2: Geometric Filter ==========
    print(f"\n{'=' * 60}")
    print(" Step 2: Geometric Filter")
    print("=" * 60)

    filter_stats = {"total": 0, "pnp_fail": 0, "c1_fail": 0, "c2_fail": 0, "c3_fail": 0, "passed": 0}
    passed_results = []
    csv_rows = []

    for r in inference_results:
        filter_stats["total"] += 1
        row = {"frame": r["basename"], "n_kps": r["n_detected"], "pnp_ok": r["pnp_ok"],
               "reproj_err": None, "c1_pass": None, "c2_pass": None, "c3_pass": None,
               "est_size": None, "final_pass": False, "fail_reason": ""}

        if not r["pnp_ok"]:
            filter_stats["pnp_fail"] += 1
            row["fail_reason"] = "pnp_fail"
            csv_rows.append(row)
            print(f"  {r['basename']}: SKIP (PnP failed)")
            continue

        is_valid, details = geo_filter.validate(r["keypoints_2d"], r["R"], r["t"])

        row["reproj_err"] = details["reproj_error_mean"]
        row["c1_pass"] = details["condition_1_reproj"]
        row["c2_pass"] = details["condition_2_geometry"]
        row["c3_pass"] = details["condition_3_size"]
        row["est_size"] = details["estimated_size"]
        row["final_pass"] = is_valid

        if is_valid:
            filter_stats["passed"] += 1
            passed_results.append(r)
            print(f"  {r['basename']}: PASS (reproj={details['reproj_error_mean']:.1f}px, size={details['estimated_size']:.2f}m)")
        else:
            reasons = []
            if not details["condition_1_reproj"]:
                filter_stats["c1_fail"] += 1
                reasons.append(f"C1 reproj={details['reproj_error_mean']:.1f}px")
            if not details["condition_2_geometry"]:
                filter_stats["c2_fail"] += 1
                reasons.append("C2 geometry")
            if not details["condition_3_size"]:
                filter_stats["c3_fail"] += 1
                reasons.append(f"C3 size={details['estimated_size']:.2f}m")
            row["fail_reason"] = " + ".join(reasons) if reasons else "unknown"
            print(f"  {r['basename']}: FAIL ({row['fail_reason']})")

        csv_rows.append(row)

    # Save CSV
    import csv
    csv_path = os.path.join(args.output_dir, "filter_details.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "n_kps", "pnp_ok", "reproj_err",
                                                "c1_pass", "c2_pass", "c3_pass", "est_size",
                                                "final_pass", "fail_reason"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Filter details saved: {csv_path}")

    acceptance_rate = filter_stats["passed"] / max(filter_stats["total"], 1) * 100
    print(f"\n  Filter Summary:")
    print(f"    Total:     {filter_stats['total']}")
    print(f"    PnP fail:  {filter_stats['pnp_fail']}")
    print(f"    C1 fail:   {filter_stats['c1_fail']} (reprojection)")
    print(f"    C2 fail:   {filter_stats['c2_fail']} (cuboid geometry)")
    print(f"    C3 fail:   {filter_stats['c3_fail']} (physical size)")
    print(f"    Passed:    {filter_stats['passed']}")
    print(f"    Acceptance rate: {acceptance_rate:.1f}%")

    if acceptance_rate < 10:
        print("    WARNING: Rate < 10% -- filter too strict")
    elif acceptance_rate > 80:
        print("    WARNING: Rate > 80% -- filter may be too loose")
    else:
        print("    OK: Normal range (20-60%)")

    # ========== Step 3: Pseudo-label 생성 ==========
    print(f"\n{'=' * 60}")
    print(" Step 3: Pseudo-label 저장")
    print("=" * 60)

    kp3d = make_pallet_keypoints_3d()

    pseudo_count = 0
    for r in passed_results:
        reproj = pnp_solver.reproject(r["R"], r["t"])

        annotation = {
            "camera_data": {
                "width": 640, "height": 480,
                "intrinsics": {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy},
            },
            "objects": [{
                "class": "pallet",
                "name": "pseudo_label",
                "visibility": 1.0,
                "projected_cuboid": reproj[:8].tolist(),
                "projected_cuboid_centroid": reproj[8].tolist(),
                "gt_source": "pseudo_label",
                "filter_passed": True,
            }],
        }

        # Save image + json pair (NDDS format, PNG for DOPE DataLoader)
        # DOPE DataLoader expects {i:06d}.png + {i:06d}.json pairs
        pseudo_id = f"{pseudo_count:06d}"
        img_dst = os.path.join(pseudo_dir, pseudo_id + ".png")
        json_dst = os.path.join(pseudo_dir, pseudo_id + ".json")

        img_bgr = cv2.imread(r["img_path"])
        cv2.imwrite(img_dst, img_bgr)
        with open(json_dst, "w") as f:
            json.dump(annotation, f, indent=2)
        pseudo_count += 1

    print(f"  Saved {pseudo_count} pseudo-labels to {pseudo_dir}")

    # ========== Step 4: Finetune 1 epoch ==========
    if len(passed_results) == 0:
        print("\n  WARNING: No pseudo-labels passed filter. Skipping finetune.")
        return

    print(f"\n{'=' * 60}")
    print(" Step 4: Finetune 1 epoch (synthetic + pseudo)")
    print("=" * 60)

    # Re-use DOPE's training infrastructure
    from utils import CleanVisiiDopeLoader

    # Create mixed dataset: take first 50 synthetic + all pseudo
    syn_dataset = CleanVisiiDopeLoader(
        [args.syn_dir],
        sigma=4.0, output_size=50, objects=["pallet"],
    )

    # Use a small subset of synthetic for smoke test
    syn_subset_size = min(50, len(syn_dataset))
    syn_subset = torch.utils.data.Subset(syn_dataset, range(syn_subset_size))

    # Pseudo-label dataset (if any)
    pseudo_dataset = CleanVisiiDopeLoader(
        [pseudo_dir],
        sigma=4.0, output_size=50, objects=["pallet"],
    )

    combined = torch.utils.data.ConcatDataset([syn_subset, pseudo_dataset])
    loader = torch.utils.data.DataLoader(combined, batch_size=4, shuffle=True, num_workers=0)

    print(f"  Synthetic subset: {syn_subset_size}")
    print(f"  Pseudo-labels:    {len(pseudo_dataset)}")
    print(f"  Combined:         {len(combined)}")
    print(f"  Batches:          {len(loader)}")

    # Load model for finetuning
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    losses = []
    for batch_idx, targets in enumerate(loader):
        optimizer.zero_grad()
        data = targets["img"].cuda()
        target_belief = targets["beliefs"].cuda()
        target_aff = targets["affinities"].cuda()

        output_belief, output_aff = model(data)

        loss_bel = torch.tensor(0.0).cuda()
        loss_aff = torch.tensor(0.0).cuda()
        for stage in range(len(output_aff)):
            loss_aff += ((output_aff[stage] - target_aff) ** 2).mean()
            loss_bel += ((output_belief[stage] - target_belief) ** 2).mean()

        loss = loss_bel + loss_aff

        # NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠ BATCH {batch_idx}: NaN/Inf detected! loss={loss.item()}")
            return

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 5 == 0:
            print(f"  batch {batch_idx}/{len(loader)}: loss={loss.item():.6f} (bel={loss_bel.item():.6f}, aff={loss_aff.item():.6f})")

    print(f"\n  1 epoch done. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Mean loss: {np.mean(losses):.4f}")
    print(f"  NaN detected: No")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "smoke_test_epoch1.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # ========== Summary ==========
    print(f"\n{'=' * 60}")
    print(" SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"  Inference:    {pnp_ok_count}/{len(inference_results)} PnP OK")
    print(f"  Filter:       {filter_stats['passed']}/{filter_stats['total']} passed ({acceptance_rate:.0f}%)")
    print(f"  Pseudo-label: {len(passed_results)} saved")
    print(f"  Finetune:     loss {losses[0]:.4f} → {losses[-1]:.4f}, no NaN")
    print(f"  Status:       Pipeline OK")


if __name__ == "__main__":
    main()
