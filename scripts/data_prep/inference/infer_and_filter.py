"""DOPE 추론 + 3단계 Geometric Filter (A/B/C) 적용.

플랫 디렉토리(*.jpg/*.png)에 대해:
1) DOPE 추론 → keypoint 추출 → PnP
2) Filter A: Flip Consistency
3) Filter B: Diagonal Concurrency
4) Filter C: Leave-One-Out PnP
5) 결과를 all/, pnp_ok/, filter_passed/ 로 분류 저장

사용법:
    python scripts/data_prep/infer_and_filter.py \
        --weights weights/mixed_v1/final_net_epoch_0060.pth \
        --img_dir data/pallet/real_data \
        --output_dir data/pallet/real_data_results_mixed_v1
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


# ── Filter A: Flip Consistency ──────────────────────────────────────────

# 좌우 flip 시 대칭 keypoint 매핑 (0↔1, 2↔3, 4↔5, 6↔7)
FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]


def filter_a_flip_consistency(model, img_bgr, pred_kps_belief, device, tau_a=5.0):
    """원본 예측 vs flip 예측의 대칭 일관성."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Flip 이미지 추론
    img_flip = cv2.flip(img_bgr, 1)
    img_rgb = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief_flip = out_bel[-1][0].cpu().numpy()
    kps_flip = extract_keypoints(belief_flip, 0.1)

    bh, bw = belief_flip.shape[1], belief_flip.shape[2]

    # 원본 keypoint (belief 좌표계)
    kp_orig = []
    for kp in pred_kps_belief[:8]:
        if kp is None:
            kp_orig.append(None)
        else:
            kp_orig.append(np.array([kp[0], kp[1]]))

    # Flip keypoint → 원래 좌표로 변환 + 대칭 매핑
    kp_flip_back = [None] * 8
    for kp_idx, kp in enumerate(kps_flip[:8]):
        if kp is None:
            continue
        kp_flip_back[kp_idx] = np.array([bw - kp[0], kp[1]])

    kp_flip_remapped = [None] * 8
    for a, b in FLIP_PAIRS:
        kp_flip_remapped[a] = kp_flip_back[b]
        kp_flip_remapped[b] = kp_flip_back[a]

    # 일관성 계산
    errors = []
    for i in range(8):
        if kp_orig[i] is None or kp_flip_remapped[i] is None:
            continue
        errors.append(np.linalg.norm(kp_orig[i] - kp_flip_remapped[i]))

    if len(errors) < 4:
        return False, float("inf")

    mean_err = float(np.mean(errors))
    return mean_err < tau_a, mean_err


# ── Filter A2: Belief Confidence ────────────────────────────────────────

def filter_a2_belief_confidence(belief_maps, pred_kps, tau_peak=0.3, min_confident=4):
    """감지된 keypoint의 belief peak 값 기반 신뢰도 검증.

    Flip consistency 대신 — 어두운/어려운 이미지에서 flip이 불안정하므로
    belief peak 값이 충분히 높은 keypoint가 최소 개수 이상인지 확인.
    """
    peaks = []
    for i, kp in enumerate(pred_kps[:8]):
        if kp is None:
            continue
        x, y = int(round(kp[0])), int(round(kp[1]))
        h, w = belief_maps.shape[1], belief_maps.shape[2]
        if 0 <= y < h and 0 <= x < w:
            peaks.append(float(belief_maps[i, y, x]))

    if not peaks:
        return False, 0.0

    confident = sum(1 for p in peaks if p > tau_peak)
    mean_peak = float(np.mean(peaks))

    return confident >= min_confident, mean_peak


# ── Filter B: Diagonal Concurrency ──────────────────────────────────────

FACES = [
    (0, 1, 2, 3),  # front
    (4, 5, 6, 7),  # rear
    (0, 1, 5, 4),  # top
    (3, 2, 6, 7),  # bottom
    (1, 2, 6, 5),  # left
    (0, 3, 7, 4),  # right
]


def filter_b_diagonal_concurrency(kps_orig, tau_b=8.0):
    """각 면의 대각선 중점 일치 검증 (2D)."""
    kp = []
    for p in kps_orig[:8]:
        if p is None:
            return False, float("inf")
        kp.append(np.array([p[0], p[1]], dtype=np.float64))

    errors = []
    for a, b, c, d in FACES:
        mid_ac = (kp[a] + kp[c]) / 2
        mid_bd = (kp[b] + kp[d]) / 2
        errors.append(np.linalg.norm(mid_ac - mid_bd))

    if not errors:
        return False, float("inf")

    mean_err = float(np.mean(errors))
    return mean_err < tau_b, mean_err


# ── Filter B2: 3D Geometry Check ────────────────────────────────────────

def filter_b2_3d_geometry(kps_orig, pnp_solver, R, t,
                          tau_reproj=30.0, tau_vol_lo=0.3, tau_vol_hi=3.0,
                          tau_spread=10000.0, min_kps=4):
    """PnP 결과의 3D 기하학적 타당성 검증.

    감지된 keypoint만으로 검증 (8개 미만도 가능).
    1) Reprojection error: 감지된 keypoint만 PnP 재투영과 비교
    2) Volume ratio: PnP로 복원된 3D cuboid 부피 vs GT 부피
    3) Keypoint spread: 재투영된 8개 점의 convex hull 면적 (납작 cuboid 탈락)
    """
    # 감지된 keypoint 수집
    detected_idx = []
    detected_2d = []
    for i, p in enumerate(kps_orig[:8]):
        if p is not None:
            detected_idx.append(i)
            detected_2d.append([float(p[0]), float(p[1])])

    if len(detected_idx) < min_kps:
        return False, {"reproj": float("inf"), "vol_ratio": 0, "spread": 0, "spread_det": 0}

    detected_2d = np.array(detected_2d, dtype=np.float64)

    # 1) Reprojection error — 감지된 keypoint만 비교
    reproj_all = pnp_solver.reproject(R, t)[:8]
    reproj_detected = reproj_all[detected_idx]
    reproj_err = float(np.mean(np.linalg.norm(detected_2d - reproj_detected, axis=1)))

    # 2) Volume ratio — PnP 3D cuboid
    kp3d = pnp_solver.keypoints_3d[:8]
    pts_cam = (R @ kp3d.T).T + t  # (8, 3)
    e01 = np.linalg.norm(pts_cam[1] - pts_cam[0])
    e03 = np.linalg.norm(pts_cam[3] - pts_cam[0])
    e04 = np.linalg.norm(pts_cam[4] - pts_cam[0])
    vol_pred = e01 * e03 * e04
    vol_gt = 1.1 * 1.1 * 0.15  # 0.1815 m³
    vol_ratio = vol_pred / vol_gt if vol_gt > 0 else 0

    # 3) Spread 검증
    from scipy.spatial import ConvexHull

    # 재투영 spread (PnP cuboid가 납작하지 않은지)
    try:
        spread_reproj = ConvexHull(reproj_all).volume
    except Exception:
        spread_reproj = 0

    # 감지된 keypoint spread (점들이 몰려있지 않은지)
    if len(detected_2d) >= 3:
        try:
            spread_det = ConvexHull(detected_2d).volume
        except Exception:
            spread_det = 0
    else:
        spread_det = 0

    # reproj median (outlier 영향 감소)
    reproj_per_kp = np.linalg.norm(detected_2d - reproj_detected, axis=1)
    reproj_med = float(np.median(reproj_per_kp))

    details = {
        "reproj": reproj_err,
        "reproj_med": reproj_med,
        "vol_ratio": vol_ratio,
        "spread": spread_reproj,
        "spread_det": spread_det,
    }

    passed = (reproj_med < tau_reproj and
              reproj_err < tau_reproj * 2 and  # mean도 과도하지 않아야
              spread_reproj > tau_spread and
              spread_det > 200)  # 감지 점이 충분히 퍼져있어야 (200px² 이상)
    return passed, details


# ── Filter C: Leave-One-Out PnP ─────────────────────────────────────────

def filter_c_loo_pnp(kps_orig, pnp_solver, tau_c=10.0):
    """8개 keypoint 중 1개씩 빼고 PnP → reprojection 안정성."""
    kp2d = []
    for p in kps_orig[:8]:
        if p is None:
            return False, float("inf")
        kp2d.append([float(p[0]), float(p[1])])
    kp2d = np.array(kp2d, dtype=np.float64)
    kp3d = pnp_solver.keypoints_3d[:8].astype(np.float64)

    # Full PnP
    success_full, rvec_full, tvec_full = cv2.solvePnP(
        kp3d, kp2d, pnp_solver.camera_matrix, None, flags=cv2.SOLVEPNP_EPNP
    )
    if not success_full:
        return False, float("inf")

    reproj_full, _ = cv2.projectPoints(
        kp3d, rvec_full, tvec_full, pnp_solver.camera_matrix, None
    )
    reproj_full = reproj_full.reshape(-1, 2)

    # LOO
    max_err = 0.0
    for i in range(8):
        mask = [j for j in range(8) if j != i]
        success_loo, rvec_loo, tvec_loo = cv2.solvePnP(
            kp3d[mask], kp2d[mask], pnp_solver.camera_matrix, None,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success_loo:
            continue
        reproj_loo, _ = cv2.projectPoints(
            kp3d, rvec_loo, tvec_loo, pnp_solver.camera_matrix, None
        )
        reproj_loo = reproj_loo.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(reproj_full - reproj_loo, axis=1)))
        max_err = max(max_err, err)

    return max_err < tau_c, max_err


# ── Filter C2: Pose Plausibility ───────────────────────────────────────

def filter_c2_pose_plausibility(R, t, tau_depth_min=0.5, tau_depth_max=15.0,
                                 tau_tilt=45.0):
    """PnP pose의 물리적 타당성 검증.

    LOO PnP 대신 — keypoint 적을 때 LOO가 불안정하므로
    pose 자체가 물리적으로 말이 되는지 확인:
    1) 깊이(거리)가 합리적 범위인지
    2) 팔레트가 대략 수평인지 (과도한 기울기 아닌지)
    """
    # 1) Depth check
    depth = float(t[2])
    if depth < tau_depth_min or depth > tau_depth_max:
        return False, {"depth": depth, "tilt": 90.0}

    # 2) Tilt check — 팔레트 Y축(up)이 카메라 프레임에서 얼마나 기울었는지
    # 팔레트 object frame Y축 = [0, 1, 0]
    # 카메라 프레임에서: R @ [0, 1, 0]
    y_obj = np.array([0, 1, 0], dtype=np.float64)
    y_cam = R @ y_obj
    # 카메라 프레임 Y축(아래 방향)과의 각도
    # 팔레트가 바닥에 있으면 y_cam ≈ [0, -1, 0] 또는 [0, 1, 0]
    cos_angle = abs(y_cam[1])  # Y 성분의 절대값
    tilt_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))

    details = {"depth": depth, "tilt": tilt_deg}
    return tilt_deg < tau_tilt, details


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    parser.add_argument("--tau_a", type=float, default=5.0, help="Filter A threshold")
    parser.add_argument("--tau_b", type=float, default=8.0, help="Filter B threshold")
    parser.add_argument("--tau_c", type=float, default=10.0, help="Filter C threshold")
    parser.add_argument("--threshold", type=float, default=0.3, help="Belief threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    dirs = {}
    for name in ["all", "pnp_ok",
                  "filter_a", "filter_a2",
                  "filter_b", "filter_b2",
                  "filter_c", "filter_c2",
                  "passed_abc", "passed_a2b2c2"]:
        dirs[name] = os.path.join(args.output_dir, name)
        os.makedirs(dirs[name], exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(args.img_dir, "*.jpg")) +
        glob.glob(os.path.join(args.img_dir, "*.png"))
    )
    print(f"Model: {args.weights} (device: {device})")
    print(f"Images: {len(imgs)}")
    print(f"Thresholds: tau_a={args.tau_a}, tau_b={args.tau_b}, tau_c={args.tau_c}")

    csv_path = os.path.join(args.output_dir, "filter_details.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", "n_kps", "pnp_ok",
        "filter_a", "err_a", "filter_a2", "a2_peak",
        "filter_b", "err_b", "filter_b2", "b2_reproj", "b2_spread", "b2_spread_det",
        "filter_c", "err_c", "filter_c2", "c2_depth", "c2_tilt",
        "passed_abc", "passed_a2b2c2"
    ])

    stats = {k: 0 for k in [
        "total", "pnp_ok",
        "a", "a2", "b", "b2", "c", "c2",
        "abc", "a2b2c2"
    ]}

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            continue

        belief = infer(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)
        detected = sum(1 for kp in pred_kps if kp is not None)

        # 원본 해상도로 변환
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

        # 오버레이 → all/
        vis = draw_overlay(img, pred_kps, None, belief, pnp,
                           f"{basename} | {detected}/9")
        cv2.imwrite(os.path.join(dirs["all"], f"{basename}_overlay.jpg"), vis)

        stats["total"] += 1
        pnp_ok = bool(success)
        fa, err_a = False, float("inf")
        fa2, a2_peak = False, 0.0
        fb, err_b = False, float("inf")
        fb2, b2_details = False, {"reproj": float("inf"), "spread": 0, "spread_det": 0}
        fc, err_c = False, float("inf")
        fc2, c2_details = False, {"depth": 0, "tilt": 90}

        if pnp_ok:
            stats["pnp_ok"] += 1
            cv2.imwrite(os.path.join(dirs["pnp_ok"], f"{basename}_overlay.jpg"), vis)

            # Filter A (Flip Consistency)
            fa, err_a = filter_a_flip_consistency(model, img, pred_kps, device, args.tau_a)
            if fa:
                stats["a"] += 1
                cv2.imwrite(os.path.join(dirs["filter_a"], f"{basename}_overlay.jpg"), vis)

            # Filter A2 (Belief Confidence)
            fa2, a2_peak = filter_a2_belief_confidence(belief, pred_kps)
            if fa2:
                stats["a2"] += 1
                cv2.imwrite(os.path.join(dirs["filter_a2"], f"{basename}_overlay.jpg"), vis)

            # Filter B (2D Diagonal)
            fb, err_b = filter_b_diagonal_concurrency(pred_orig, args.tau_b)
            if fb:
                stats["b"] += 1
                cv2.imwrite(os.path.join(dirs["filter_b"], f"{basename}_overlay.jpg"), vis)

            # Filter B2 (3D Spread + Reproj)
            fb2, b2_details = filter_b2_3d_geometry(pred_orig, pnp, R, t)
            if fb2:
                stats["b2"] += 1
                cv2.imwrite(os.path.join(dirs["filter_b2"], f"{basename}_overlay.jpg"), vis)

            # Filter C (LOO PnP)
            fc, err_c = filter_c_loo_pnp(pred_orig, pnp, args.tau_c)
            if fc:
                stats["c"] += 1
                cv2.imwrite(os.path.join(dirs["filter_c"], f"{basename}_overlay.jpg"), vis)

            # Filter C2 (Pose Plausibility)
            fc2, c2_details = filter_c2_pose_plausibility(R, t)
            if fc2:
                stats["c2"] += 1
                cv2.imwrite(os.path.join(dirs["filter_c2"], f"{basename}_overlay.jpg"), vis)

            # 조합
            if fa and fb and fc:
                stats["abc"] += 1
                cv2.imwrite(os.path.join(dirs["passed_abc"], f"{basename}_overlay.jpg"), vis)
            if fa2 and fb2 and fc2:
                stats["a2b2c2"] += 1
                cv2.imwrite(os.path.join(dirs["passed_a2b2c2"], f"{basename}_overlay.jpg"), vis)

        passed_abc = pnp_ok and fa and fb and fc
        passed_v2 = pnp_ok and fa2 and fb2 and fc2
        writer.writerow([
            basename, detected, pnp_ok,
            fa, f"{err_a:.2f}", fa2, f"{a2_peak:.3f}",
            fb, f"{err_b:.2f}", fb2, f"{b2_details['reproj']:.2f}", f"{b2_details['spread']:.1f}", f"{b2_details['spread_det']:.1f}",
            fc, f"{err_c:.2f}", fc2, f"{c2_details['depth']:.2f}", f"{c2_details['tilt']:.1f}",
            passed_abc, passed_v2
        ])

        if (i + 1) % 100 == 0 or (i + 1) == len(imgs):
            print(f"  [{i+1}/{len(imgs)}] pnp={stats['pnp_ok']} "
                  f"A={stats['a']} A2={stats['a2']} B={stats['b']} B2={stats['b2']} "
                  f"C={stats['c']} C2={stats['c2']} "
                  f"ABC={stats['abc']} A2B2C2={stats['a2b2c2']}")

    csv_file.close()

    print(f"\n{'='*60}")
    print(f"Total images:     {stats['total']}")
    print(f"PnP OK:           {stats['pnp_ok']} ({100*stats['pnp_ok']/max(stats['total'],1):.1f}%)")
    pnp_n = max(stats['pnp_ok'], 1)
    tot_n = max(stats['total'], 1)
    print(f"Filter A passed:  {stats['a']:>4} ({100*stats['a']/pnp_n:.1f}% of PnP) [Flip Consistency]")
    print(f"Filter A2 passed: {stats['a2']:>4} ({100*stats['a2']/pnp_n:.1f}% of PnP) [Belief Confidence]")
    print(f"Filter B passed:  {stats['b']:>4} ({100*stats['b']/pnp_n:.1f}% of PnP) [2D Diagonal]")
    print(f"Filter B2 passed: {stats['b2']:>4} ({100*stats['b2']/pnp_n:.1f}% of PnP) [3D Spread+Reproj]")
    print(f"Filter C passed:  {stats['c']:>4} ({100*stats['c']/pnp_n:.1f}% of PnP) [LOO PnP]")
    print(f"Filter C2 passed: {stats['c2']:>4} ({100*stats['c2']/pnp_n:.1f}% of PnP) [Pose Plausibility]")
    print(f"---")
    print(f"A+B+C passed:     {stats['abc']:>4} ({100*stats['abc']/tot_n:.1f}% of total) [기존]")
    print(f"A2+B2+C2 passed:  {stats['a2b2c2']:>4} ({100*stats['a2b2c2']/tot_n:.1f}% of total) [신규]")
    print(f"\nDetails: {csv_path}")
    print(f"Output:  {args.output_dir}/")


if __name__ == "__main__":
    main()
