"""Geometric Filter for pseudo-label validation in self-training.

## Current design (2026-04-11 after GT-based P/R analysis)

Primary gate: **RANSAC subset consensus** (filter F11 in filter_pr_eval.py).

Selection rationale (see _docs/filter/2026-04-11_selection.md):
- GT-based precision/recall on capture0403middle (440 frames) compared 23
  filter candidates across 2 models (v8_A ep68, selftrain_r1).
- Default canonical B∧C (F7) had very low recall (F1 ≤ 0.235 on ep68,
  ≤ 0.069 on r1). Threshold sweeps (2x/3x/very-loose) could not bring
  F7's F1 above ~0.53, never matching RANSAC subset consensus.
- RANSAC subset consensus with min_consensus=6 was the top-ranked filter
  on BOTH models (F1 = 0.833 on ep68, 0.722 on r1).
- consensus parameter sweep (4/5/6/7/8) confirmed 6 as the sweet spot.

## What this filter does

Given 2D keypoints predicted by DOPE, the filter runs its own RANSAC
with random subsets of 5 points. Each subset yields a candidate pose
via EPnP; the pose is scored by how many of the detected 2D points it
projects to within `ransac_reproj_px`. The pose with the highest inlier
count (consensus) wins. If that consensus is at least `ransac_min_consensus`,
the pose is accepted.

A size sanity check (0.5m < recovered pallet width < 2.5m) is kept as
an additional physical-plausibility guard.

## API

Two entry points:

1. `solve_and_validate(keypoints_2d)` — primary self-training use. Runs
   RANSAC subset consensus internally and returns both the recovered
   pose and the accept/reject decision.

2. `validate(keypoints_2d, R, t)` — legacy API that checks whether an
   externally-provided pose has enough inlier support. Kept so that
   self_train.py can call this filter after an external solver and for
   ablation runs that score poses recovered by different solvers.

Config (from `config/stage3_selftrain.yaml` geometric_filter section):
    ransac_n_iter: 50            # RANSAC iterations
    ransac_subset: 5             # keypoints per subset
    ransac_reproj_px: 5.0        # inlier distance threshold
    ransac_min_consensus: 6      # minimum inliers for accept
    tau_size_min: 0.5            # minimum physical width (m)
    tau_size_max: 2.5            # maximum physical width (m)
    min_keypoints: 5             # minimum detected keypoints
"""

import cv2
import numpy as np


class GeometricFilter:
    """Validate pseudo-labels via RANSAC subset consensus + size sanity."""

    def __init__(self, pnp_solver, config=None):
        """
        Args:
            pnp_solver: PalletPnPSolver instance (for camera matrix / 3D model).
            config: dict with filter parameters. Uses defaults if None.
        """
        self.pnp_solver = pnp_solver

        cfg = config or {}
        self.ransac_n_iter = int(cfg.get("ransac_n_iter", 50))
        self.ransac_subset = int(cfg.get("ransac_subset", 5))
        self.ransac_reproj_px = float(cfg.get("ransac_reproj_px", 5.0))
        self.ransac_min_consensus = int(cfg.get("ransac_min_consensus", 6))
        self.tau_size_min = float(cfg.get("tau_size_min", 0.5))
        self.tau_size_max = float(cfg.get("tau_size_max", 2.5))
        self.min_keypoints = int(cfg.get("min_keypoints", 5))

        self._rng = np.random.default_rng(cfg.get("seed", 0))

    # ── Primary entry point ──────────────────────────────────────────
    def solve_and_validate(self, keypoints_2d):
        """Run RANSAC subset consensus + size sanity.

        Args:
            keypoints_2d: list of 9 elements, each (u, v), (u, v, conf),
                or None for missing.

        Returns:
            is_valid: bool, accept/reject.
            R: (3, 3) rotation matrix, or None.
            t: (3,) translation vector, or None.
            details: dict with consensus count, size, and flags.
        """
        details = {
            "consensus": 0,
            "n_detected": 0,
            "estimated_size": 0.0,
            "consensus_pass": False,
            "size_pass": False,
            "reproj_error_mean": float("inf"),  # kept for self_train.py compatibility
        }

        detected_idx = []
        detected_2d = []
        for i, pt in enumerate(keypoints_2d[:8]):
            if pt is None:
                continue
            if hasattr(pt, "__len__") and len(pt) >= 2:
                u, v = float(pt[0]), float(pt[1])
                if u >= 0 and v >= 0:
                    detected_idx.append(i)
                    detected_2d.append([u, v])

        details["n_detected"] = len(detected_idx)
        if len(detected_idx) < max(self.min_keypoints, self.ransac_subset):
            return False, None, None, details

        detected_2d = np.array(detected_2d, dtype=np.float64)
        kp3d_all = self.pnp_solver.keypoints_3d[:8].astype(np.float64)
        detected_3d = kp3d_all[detected_idx]

        best_consensus, best_rvec, best_tvec = self._ransac_subset(
            detected_2d, detected_3d)

        details["consensus"] = int(best_consensus)
        if best_rvec is None:
            return False, None, None, details

        R, _ = cv2.Rodrigues(best_rvec)
        t = best_tvec.flatten()

        # Compute mean reprojection error vs detected (for logging compatibility)
        reproj, _ = cv2.projectPoints(
            detected_3d, best_rvec, best_tvec,
            self.pnp_solver.camera_matrix, self.pnp_solver.dist_coeffs)
        errors = np.linalg.norm(reproj.reshape(-1, 2) - detected_2d, axis=1)
        details["reproj_error_mean"] = float(np.mean(errors))

        # Consensus gate
        consensus_pass = best_consensus >= self.ransac_min_consensus
        details["consensus_pass"] = consensus_pass

        # Size sanity
        size_pass, est_size = self._check_size(R, t)
        details["size_pass"] = size_pass
        details["estimated_size"] = est_size

        is_valid = bool(consensus_pass and size_pass)
        return is_valid, R, t, details

    # ── Legacy / ablation API ────────────────────────────────────────
    def validate(self, keypoints_2d, R, t):
        """Validate an externally-provided pose via inlier count + size.

        Kept so self_train.py's current flow (external solver → filter)
        continues to work and so ablation runs can pass in poses from
        alternative solvers.

        Args:
            keypoints_2d: list of 9 elements, each (u, v) or None.
            R: (3, 3) rotation matrix.
            t: (3,) translation vector.

        Returns:
            is_valid: bool.
            details: dict with consensus / size / reproj_error_mean.
        """
        details = {
            "consensus": 0,
            "n_detected": 0,
            "estimated_size": 0.0,
            "consensus_pass": False,
            "size_pass": False,
            "reproj_error_mean": float("inf"),
        }

        detected_idx = []
        detected_2d = []
        for i, pt in enumerate(keypoints_2d[:8]):
            if pt is None:
                continue
            if hasattr(pt, "__len__") and len(pt) >= 2:
                u, v = float(pt[0]), float(pt[1])
                if u >= 0 and v >= 0:
                    detected_idx.append(i)
                    detected_2d.append([u, v])

        details["n_detected"] = len(detected_idx)
        if len(detected_idx) < self.min_keypoints:
            return False, details

        detected_2d = np.array(detected_2d, dtype=np.float64)

        # Reproject all 8 corners with the given pose
        reproj_all = self.pnp_solver.reproject(R, t)[:8]
        reproj_detected = reproj_all[detected_idx]
        errors = np.linalg.norm(reproj_detected - detected_2d, axis=1)
        details["reproj_error_mean"] = float(np.mean(errors))

        consensus = int(np.sum(errors < self.ransac_reproj_px))
        details["consensus"] = consensus
        consensus_pass = consensus >= self.ransac_min_consensus
        details["consensus_pass"] = consensus_pass

        size_pass, est_size = self._check_size(R, t)
        details["size_pass"] = size_pass
        details["estimated_size"] = est_size

        return bool(consensus_pass and size_pass), details

    # ── Internals ────────────────────────────────────────────────────
    def _ransac_subset(self, detected_2d, detected_3d):
        """Random subset EPnP with full-inlier consensus voting.

        Returns (best_consensus, best_rvec, best_tvec). If no iteration
        succeeds, best_rvec / best_tvec are None and best_consensus is 0.
        """
        n = len(detected_2d)
        best_consensus = 0
        best_rvec, best_tvec = None, None

        for _ in range(self.ransac_n_iter):
            if n == self.ransac_subset:
                sel = np.arange(n)
            else:
                sel = self._rng.choice(n, size=self.ransac_subset, replace=False)
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    detected_3d[sel], detected_2d[sel],
                    self.pnp_solver.camera_matrix,
                    self.pnp_solver.dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP,
                )
            except cv2.error:
                continue
            if not ok or float(tvec[2, 0]) < 0:
                continue

            reproj, _ = cv2.projectPoints(
                detected_3d, rvec, tvec,
                self.pnp_solver.camera_matrix,
                self.pnp_solver.dist_coeffs,
            )
            errors = np.linalg.norm(reproj.reshape(-1, 2) - detected_2d, axis=1)
            consensus = int(np.sum(errors < self.ransac_reproj_px))
            if consensus > best_consensus:
                best_consensus = consensus
                best_rvec, best_tvec = rvec, tvec

        return best_consensus, best_rvec, best_tvec

    def _check_size(self, R, t):
        """Sanity check: recovered pallet width must be physically plausible."""
        kp3d = self.pnp_solver.keypoints_3d[:8]
        pts_cam = (R @ kp3d.T).T + t
        # Distance between the two corners that span the width edge (0–1).
        # With canonical ordering this is the width edge; with other
        # conventions it's still a cuboid edge of order ~width magnitude.
        width_3d = float(np.linalg.norm(pts_cam[0] - pts_cam[1]))
        return (self.tau_size_min < width_3d < self.tau_size_max), width_3d
