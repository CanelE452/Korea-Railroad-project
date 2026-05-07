"""3-Way PnP 비교: current vs sigma-weighted vs reproj-guided.

v8_A 모델을 지정된 이미지에 대해 3가지 PnP 모드로 비교하고,
B∧C filter 통과율 및 mode 간 transition을 분석.

사용법:
    python scripts/data_prep/infer_and_filter_v4.py \
        --weights weights/v9_ablation_A_coord/final_net_epoch_0065.pth \
        --img_dir data/pallet/raw_data/capture0403noapril/noapril_eval \
        --output_dir data/pallet/eval_results/pnp_reproj_compare \
        --calibration_json data/pallet/eval_results/calibration_results.json
"""

import argparse
import glob
import json
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

from visualize_inference import load_model, extract_keypoints
from pnp_solver import PalletPnPSolver, make_camera_matrix
from canonical_filters import filter_B, filter_C


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # front face
    (4, 5), (5, 6), (6, 7), (7, 4),  # rear face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]


def collect_images(img_dir):
    """img_dir 또는 filelist.txt 기반으로 이미지 경로 수집."""
    filelist = os.path.join(img_dir, "filelist.txt")
    if os.path.exists(filelist):
        with open(filelist) as f:
            names = [l.strip() for l in f if l.strip()]
        rgb_dir = os.path.join(os.path.dirname(img_dir), "rgb")
        if os.path.isdir(rgb_dir):
            return sorted([os.path.join(rgb_dir, n) for n in names])
        return sorted([os.path.join(img_dir, n) for n in names])
    return sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )


def infer_with_sigma(model, img_bgr, device, temperature=1.0):
    """DOPE inference -> belief maps + per-keypoint sigma."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out_bel, _ = model(tensor)
        belief_tensor = out_bel[-1][0, :9]
        C, H, W = 9, belief_tensor.shape[1], belief_tensor.shape[2]
        hm = belief_tensor.unsqueeze(0)
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
        flat = hm.view(1, C, -1)
        weights = F.softmax(flat / temperature, dim=-1).view(1, C, H, W)
        mu_x = (weights * x_coords).sum(dim=(2, 3))
        mu_y = (weights * y_coords).sum(dim=(2, 3))
        var = (weights * ((x_coords - mu_x.unsqueeze(-1).unsqueeze(-1)) ** 2 +
                          (y_coords - mu_y.unsqueeze(-1).unsqueeze(-1)) ** 2)).sum(dim=(2, 3))
        sigma = torch.sqrt(var.clamp(min=1e-6))[0]

    belief_np = out_bel[-1][0].cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    return belief_np, sigma_np


def draw_cuboid(img, pnp_solver, R, t, color=(0, 255, 255), thickness=2):
    """Draw wireframe cuboid on image."""
    vis = img.copy()
    reproj = pnp_solver.reproject(R, t)
    for i0, i1 in CUBOID_EDGES:
        p0 = tuple(reproj[i0].astype(int))
        p1 = tuple(reproj[i1].astype(int))
        cv2.line(vis, p0, p1, color, thickness)
    return vis


def evaluate_mode(pnp, pred_orig, R, t, success, sigmas_orig,
                  peak_confs, tau_span, tau_end, tau_nc, tau_C,
                  img_size, mode, reproj_params=None):
    """Run PnP + filters for a given mode. Returns result dict."""
    result = {"pnp_ok": False, "B": False, "C": False, "BC": False,
              "R": None, "t": None, "meta": None}

    if mode == "current":
        if not success:
            return result
        R_m, t_m = R, t
        result["pnp_ok"] = True

    elif mode == "sigma_weighted":
        s_ok, R_m, t_m, _ = pnp.solve(pred_orig, sigmas=sigmas_orig)
        if not s_ok:
            return result
        result["pnp_ok"] = True

    elif mode == "reproj_guided":
        p = reproj_params or {}
        s_ok, R_m, t_m, _, meta = pnp.solve_reproj_guided(
            pred_orig, peak_confidences=peak_confs,
            tau_huber=p.get("tau_huber", 0.05),
            tau_peak=p.get("tau_peak", 0.3),
            tau_w=p.get("tau_w", 0.1),
        )
        if not s_ok:
            return result
        result["pnp_ok"] = True
        result["meta"] = meta

    else:
        return result

    w, h = img_size
    result["R"] = R_m.tolist()
    result["t"] = t_m.tolist()

    fB, _ = filter_B(pred_orig, pnp, R_m, t_m,
                     tau_span, tau_end, tau_nc, img_size=(w, h))
    fC, _ = filter_C(pred_orig, pnp, R_m, t_m, tau_C)
    result["B"] = bool(fB)
    result["C"] = bool(fC)
    result["BC"] = bool(fB and fC)

    return result


def main():
    parser = argparse.ArgumentParser(description="3-Way PnP comparison")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_json", default=None,
                        help="Load tau_huber/tau_peak/tau_w from calibration output")
    parser.add_argument("--tau_huber", type=float, default=None)
    parser.add_argument("--tau_peak", type=float, default=None)
    parser.add_argument("--tau_w", type=float, default=None)
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    parser.add_argument("--tau_span", type=float, default=0.35)
    parser.add_argument("--tau_end", type=float, default=0.10)
    parser.add_argument("--tau_nc", type=float, default=0.02)
    parser.add_argument("--tau_C", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--save_per_image", action="store_true",
                        help="Save per-image JSON for detailed analysis")
    parser.add_argument("--visualize", action="store_true",
                        help="Save transition case overlay images")
    parser.add_argument("--max_vis", type=int, default=20,
                        help="Max visualizations per transition type")
    args = parser.parse_args()

    # Load reproj-guided thresholds
    reproj_params = {"tau_huber": 0.05, "tau_peak": 0.3, "tau_w": 0.1}
    if args.calibration_json and os.path.exists(args.calibration_json):
        with open(args.calibration_json) as f:
            calib = json.load(f)
        rec = calib.get("recommended", {})
        reproj_params["tau_huber"] = rec.get("tau_huber", 0.05)
        reproj_params["tau_peak"] = rec.get("tau_peak", 0.3)
        reproj_params["tau_w"] = rec.get("tau_w", 0.1)
        print(f"Loaded calibration: {reproj_params}")
    # CLI overrides
    if args.tau_huber is not None:
        reproj_params["tau_huber"] = args.tau_huber
    if args.tau_peak is not None:
        reproj_params["tau_peak"] = args.tau_peak
    if args.tau_w is not None:
        reproj_params["tau_w"] = args.tau_w

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    os.makedirs(args.output_dir, exist_ok=True)

    imgs = collect_images(args.img_dir)
    print(f"Model: {args.weights}")
    print(f"Images: {len(imgs)}")
    print(f"Reproj params: {reproj_params}")

    MODES = ["current", "sigma_weighted", "reproj_guided"]
    stats = {m: {"pnp_ok": 0, "B": 0, "C": 0, "BC": 0} for m in MODES}
    stats["total"] = 0

    # Per-image results for transition analysis
    per_image = []

    # Transition counters (current vs reproj_guided)
    transitions = {
        "cur_fail_rg_pass": 0,
        "cur_pass_rg_fail": 0,
        "sig_fail_rg_pass": 0,
        "sig_pass_rg_fail": 0,
    }

    # Reproj-guided diagnostics
    rg_fallback_count = 0
    rg_sanity_skip_count = 0
    rg_total_pnp = 0

    # Visualization lists
    vis_cases = {
        "cur_fail_rg_pass": [],
        "cur_pass_rg_fail": [],
    }

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            continue

        belief, sigma = infer_with_sigma(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)

        h, w = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = w / bw, h / bh
        pred_orig = []
        sigmas_orig = []
        peak_confs = []
        detected_count = 0
        for ki, kp in enumerate(pred_kps):
            if kp is None:
                pred_orig.append(None)
                sigmas_orig.append(None)
                peak_confs.append(0.0)
            else:
                pred_orig.append((kp[0] * sx, kp[1] * sy))
                sigmas_orig.append(float(sigma[ki]))
                peak_confs.append(float(kp[2]))
                detected_count += 1

        stats["total"] += 1

        # Current PnP (shared across modes that need it)
        success_cur, R_cur, t_cur, _ = pnp.solve(pred_orig)

        img_results = {"file": os.path.basename(path), "detected": detected_count}

        for mode in MODES:
            res = evaluate_mode(
                pnp, pred_orig, R_cur, t_cur, success_cur,
                sigmas_orig, peak_confs,
                args.tau_span, args.tau_end, args.tau_nc, args.tau_C,
                (w, h), mode, reproj_params)

            img_results[mode] = {
                "pnp_ok": res["pnp_ok"],
                "B": res["B"],
                "C": res["C"],
                "BC": res["BC"],
            }

            if res["pnp_ok"]:
                stats[mode]["pnp_ok"] += 1
            if res["B"]:
                stats[mode]["B"] += 1
            if res["C"]:
                stats[mode]["C"] += 1
            if res["BC"]:
                stats[mode]["BC"] += 1

            # Reproj-guided diagnostics
            if mode == "reproj_guided" and res["meta"] is not None:
                meta = res["meta"]
                rg_total_pnp += 1
                if meta.get("fallback_used"):
                    rg_fallback_count += 1
                if meta.get("sanity_skip"):
                    rg_sanity_skip_count += 1
                img_results["rg_meta"] = {
                    "fallback": meta.get("fallback_used", False),
                    "sanity_skip": meta.get("sanity_skip", False),
                    "n_selected": meta.get("n_selected", 0),
                    "init_res": meta.get("initial_residual_mean"),
                    "refined_res": meta.get("refined_residual_mean"),
                    "D": meta.get("D"),
                }

        per_image.append(img_results)

        # Transition analysis
        cur_bc = img_results.get("current", {}).get("BC", False)
        rg_bc = img_results.get("reproj_guided", {}).get("BC", False)
        sig_bc = img_results.get("sigma_weighted", {}).get("BC", False)

        if not cur_bc and rg_bc:
            transitions["cur_fail_rg_pass"] += 1
            vis_cases["cur_fail_rg_pass"].append((path, img_results))
        if cur_bc and not rg_bc:
            transitions["cur_pass_rg_fail"] += 1
            vis_cases["cur_pass_rg_fail"].append((path, img_results))
        if not sig_bc and rg_bc:
            transitions["sig_fail_rg_pass"] += 1
        if sig_bc and not rg_bc:
            transitions["sig_pass_rg_fail"] += 1

        if (i + 1) % 30 == 0 or (i + 1) == len(imgs):
            print(f"  [{i+1}/{len(imgs)}] current BC={stats['current']['BC']}, "
                  f"sigma BC={stats['sigma_weighted']['BC']}, "
                  f"reproj BC={stats['reproj_guided']['BC']}")

    # ========== Print Results ==========
    tn = max(stats["total"], 1)
    print(f"\n{'='*75}")
    print(f"3-Way PnP Comparison ({stats['total']} images)")
    print(f"Reproj params: tau_huber={reproj_params['tau_huber']:.5f}, "
          f"tau_peak={reproj_params['tau_peak']:.4f}, tau_w={reproj_params['tau_w']:.4f}")
    print(f"{'='*75}")
    print(f"{'Mode':<20} {'PnP':>6} {'B':>6} {'C':>6} {'B∧C':>6} "
          f"{'PnP%':>7} {'BC%':>7}")
    print(f"{'-'*75}")
    for m in MODES:
        s = stats[m]
        pnp_pct = s['pnp_ok'] / tn * 100
        bc_pct = s['BC'] / tn * 100
        print(f"{m:<20} {s['pnp_ok']:>6} {s['B']:>6} {s['C']:>6} {s['BC']:>6} "
              f"{pnp_pct:>6.1f}% {bc_pct:>6.1f}%")
    print(f"{'='*75}")

    # Transition analysis
    print(f"\nTransition Analysis (B∧C):")
    print(f"  current→reproj_guided:  fail→pass={transitions['cur_fail_rg_pass']}, "
          f"pass→fail={transitions['cur_pass_rg_fail']}, "
          f"net={transitions['cur_fail_rg_pass'] - transitions['cur_pass_rg_fail']:+d}")
    print(f"  sigma→reproj_guided:    fail→pass={transitions['sig_fail_rg_pass']}, "
          f"pass→fail={transitions['sig_pass_rg_fail']}, "
          f"net={transitions['sig_fail_rg_pass'] - transitions['sig_pass_rg_fail']:+d}")

    # Reproj-guided diagnostics
    if rg_total_pnp > 0:
        print(f"\nReproj-Guided Diagnostics:")
        print(f"  Total PnP attempts: {rg_total_pnp}")
        print(f"  Sanity skip: {rg_sanity_skip_count} ({rg_sanity_skip_count/rg_total_pnp*100:.1f}%)")
        print(f"  Fallback used: {rg_fallback_count} ({rg_fallback_count/rg_total_pnp*100:.1f}%)")
        print(f"  Refinement applied: {rg_total_pnp - rg_fallback_count - rg_sanity_skip_count}")

    # Save summary
    summary = {
        "n_images": stats["total"],
        "reproj_params": reproj_params,
        "stats": {m: dict(stats[m]) for m in MODES},
        "transitions": transitions,
        "rg_diagnostics": {
            "total_pnp": rg_total_pnp,
            "sanity_skip": rg_sanity_skip_count,
            "fallback": rg_fallback_count,
        },
    }
    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Save per-image results
    if args.save_per_image:
        per_image_path = os.path.join(args.output_dir, "per_image_results.json")
        with open(per_image_path, "w") as f:
            json.dump(per_image, f, indent=2)
        print(f"Per-image: {per_image_path}")

    # Visualize transition cases
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "transitions")
        os.makedirs(vis_dir, exist_ok=True)
        for case_type, cases in vis_cases.items():
            for ci, (img_path, img_res) in enumerate(cases[:args.max_vis]):
                img = cv2.imread(img_path)
                if img is None:
                    continue

                panels = []
                for mode in ["current", "reproj_guided"]:
                    panel = img.copy()
                    mr = img_res.get(mode, {})
                    if mr.get("pnp_ok") and mode in img_res:
                        # Find R, t from per_image data
                        # Re-run PnP to get R, t for overlay
                        belief, sigma = infer_with_sigma(model, img, device)
                        pred_kps = extract_keypoints(belief, args.threshold)
                        h, w = img.shape[:2]
                        bh, bw = belief.shape[1], belief.shape[2]
                        sx, sy = w / bw, h / bh
                        pred_orig = []
                        peak_confs_v = []
                        for ki, kp in enumerate(pred_kps):
                            if kp is None:
                                pred_orig.append(None)
                                peak_confs_v.append(0.0)
                            else:
                                pred_orig.append((kp[0]*sx, kp[1]*sy))
                                peak_confs_v.append(float(kp[2]))

                        if mode == "current":
                            s, R_v, t_v, _ = pnp.solve(pred_orig)
                        else:
                            s, R_v, t_v, _, _ = pnp.solve_reproj_guided(
                                pred_orig, peak_confidences=peak_confs_v,
                                **reproj_params)
                        if s:
                            panel = draw_cuboid(panel, pnp, R_v, t_v)

                    bc_str = "BC" if mr.get("BC") else ("B" if mr.get("B") else "x")
                    label = f"{mode}: {bc_str}"
                    cv2.putText(panel, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(panel, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    panels.append(panel)

                combined = np.hstack(panels)
                fname = f"{case_type}_{ci:03d}_{os.path.basename(img_path)}"
                out_path = os.path.join(vis_dir, fname)
                cv2.imwrite(out_path, combined)

            if cases:
                print(f"  {case_type}: {min(len(cases), args.max_vis)} images saved")


if __name__ == "__main__":
    main()
