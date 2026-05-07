"""NN matching metric 자체 검증 스크립트.

목적: 발표 방어 로직 — "metric이 올바르게 작동하는가"를 합성 val 데이터로 증명.

3가지 test:
  T1 (Identity):          GT projected_cuboid를 prediction으로 넣음 → 0 error 나와야 함
  T2 (Permutation):       GT를 랜덤 shuffle해서 pred로 → Hungarian이 복원하여 0 error
  T3 (Known perturbation): GT + Gaussian(sigma=10px) noise → mean ≈ 10 * sqrt(pi/2) ≈ 12.5px

통과 기준:
  T1: max_err < 1e-6
  T2: max_err < 1e-6
  T3: mean_err in [10, 15] px (이론값 주변)

Usage:
    python scripts/data_prep/eval/validate_nn_metric.py \
        --val_dir data/pallet/training_data/val \
        --n_frames 500
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_nn_matching import nn_matching_error  # noqa


def load_gt_cuboids(val_dir, n_frames):
    json_files = sorted(glob.glob(os.path.join(val_dir, "*.json")))[:n_frames]
    cuboids = []
    for p in json_files:
        with open(p) as f:
            data = json.load(f)
        if not data.get("objects"):
            continue
        cub = data["objects"][0].get("projected_cuboid")
        if cub is None or len(cub) < 8:
            continue
        cuboids.append(np.array(cub[:8]))
    return cuboids


def test_identity(cuboids):
    """T1: GT를 pred로 넣기 → 0 error 이어야 함."""
    all_dists = []
    for gt in cuboids:
        pred = [tuple(p) for p in gt]
        dists, _ = nn_matching_error(pred, gt.tolist())
        all_dists.extend(dists.tolist())
    arr = np.array(all_dists)
    return {
        "n": len(arr),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "pass": float(arr.max()) < 1e-6,
    }


def test_permutation(cuboids, seed=0):
    """T2: GT를 random permutation해서 pred로 → Hungarian이 복원해서 0 error."""
    rng = np.random.default_rng(seed)
    all_dists = []
    for gt in cuboids:
        perm = rng.permutation(8)
        pred = [tuple(gt[i]) for i in perm]
        dists, _ = nn_matching_error(pred, gt.tolist())
        all_dists.extend(dists.tolist())
    arr = np.array(all_dists)
    return {
        "n": len(arr),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "pass": float(arr.max()) < 1e-6,
    }


def test_known_noise(cuboids, sigma=10.0, seed=0):
    """T3: GT + Gaussian noise(sigma) → mean_err ≈ sigma * sqrt(pi/2) (Rayleigh mean)."""
    rng = np.random.default_rng(seed)
    expected_mean = sigma * np.sqrt(np.pi / 2)  # 2D distance의 Rayleigh mean
    all_dists = []
    for gt in cuboids:
        noise = rng.normal(0, sigma, size=(8, 2))
        perturbed = gt + noise
        pred = [tuple(p) for p in perturbed]
        dists, _ = nn_matching_error(pred, gt.tolist())
        all_dists.extend(dists.tolist())
    arr = np.array(all_dists)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "expected_mean": float(expected_mean),
        "pass": abs(arr.mean() - expected_mean) < 1.0,
    }


def test_swap_detection(cuboids, seed=0):
    """T4 (추가): 2개 corner만 swap → Hungarian이 swap 복원 (0 error) vs direct index (2 swap dists)."""
    rng = np.random.default_rng(seed)
    nn_dists = []
    direct_dists = []
    for gt in cuboids:
        i, j = rng.choice(8, 2, replace=False)
        pred_arr = gt.copy()
        pred_arr[[i, j]] = pred_arr[[j, i]]
        pred = [tuple(p) for p in pred_arr]

        # NN matching
        dists, _ = nn_matching_error(pred, gt.tolist())
        nn_dists.extend(dists.tolist())

        # Direct index
        d_direct = np.linalg.norm(pred_arr - gt, axis=1)
        direct_dists.extend(d_direct.tolist())

    nn_arr = np.array(nn_dists)
    direct_arr = np.array(direct_dists)
    return {
        "nn_max": float(nn_arr.max()),
        "nn_mean": float(nn_arr.mean()),
        "direct_max": float(direct_arr.max()),
        "direct_mean": float(direct_arr.mean()),
        "nn_pass": float(nn_arr.max()) < 1e-6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="data/pallet/training_data/val")
    parser.add_argument("--n_frames", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading GT from: {args.val_dir} (first {args.n_frames} frames)")
    cuboids = load_gt_cuboids(args.val_dir, args.n_frames)
    print(f"Loaded {len(cuboids)} frames with projected_cuboid\n")

    print("=" * 60)
    print("T1: Identity test (GT as pred)")
    print("=" * 60)
    r1 = test_identity(cuboids)
    print(f"  n={r1['n']}, max_err={r1['max']:.2e}, mean_err={r1['mean']:.2e}")
    print(f"  PASS: {r1['pass']}  (expect max < 1e-6)")

    print("\n" + "=" * 60)
    print("T2: Random permutation test (shuffled GT as pred)")
    print("=" * 60)
    r2 = test_permutation(cuboids, args.seed)
    print(f"  n={r2['n']}, max_err={r2['max']:.2e}, mean_err={r2['mean']:.2e}")
    print(f"  PASS: {r2['pass']}  (Hungarian recovers full permutation)")

    print("\n" + "=" * 60)
    print(f"T3: Known Gaussian noise (sigma={args.sigma}px)")
    print("=" * 60)
    r3 = test_known_noise(cuboids, args.sigma, args.seed)
    print(f"  n={r3['n']}, mean_err={r3['mean']:.2f}px, median={r3['median']:.2f}px")
    print(f"  Expected Rayleigh mean = sigma*sqrt(pi/2) = {r3['expected_mean']:.2f}px")
    print(f"  PASS: {r3['pass']}  (within +/-1px of theoretical)")

    print("\n" + "=" * 60)
    print("T4: Swap detection (NN vs direct index, 2 corners swapped)")
    print("=" * 60)
    r4 = test_swap_detection(cuboids, args.seed)
    print(f"  NN matching:   max={r4['nn_max']:.2e}, mean={r4['nn_mean']:.2e}")
    print(f"  Direct index:  max={r4['direct_max']:.1f}px, mean={r4['direct_mean']:.2f}px")
    print(f"  NN PASS: {r4['nn_pass']}  (swap invariant)")
    print(f"  -> Direct index would penalize swap ({r4['direct_mean']:.1f}px artificial error)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = r1["pass"] and r2["pass"] and r3["pass"] and r4["nn_pass"]
    print(f"  T1 Identity:           {'PASS' if r1['pass'] else 'FAIL'}")
    print(f"  T2 Permutation:        {'PASS' if r2['pass'] else 'FAIL'}")
    print(f"  T3 Known noise:        {'PASS' if r3['pass'] else 'FAIL'}")
    print(f"  T4 Swap invariance:    {'PASS' if r4['nn_pass'] else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS - metric validated' if all_pass else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
