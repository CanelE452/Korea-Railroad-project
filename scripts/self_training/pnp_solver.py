"""EPnP + RANSAC wrapper for 6D pose recovery from 2D keypoints.

Defines the standard pallet 3D keypoints (KS T 1002: 1100x1100x150mm)
and provides a clean interface for pose estimation via OpenCV solvePnP.

Usage:
    solver = PalletPnPSolver(camera_matrix, pallet_dims=(1.1, 1.1, 0.15))
    success, R, t, inliers = solver.solve(keypoints_2d)
"""

import cv2
import numpy as np


# Cuboid vertex ordering follows NDDS / DOPE convention (see cuboid.py):
#   0: FrontTopRight     4: RearTopRight
#   1: FrontTopLeft      5: RearTopLeft
#   2: FrontBottomLeft   6: RearBottomLeft
#   3: FrontBottomRight  7: RearBottomRight
#   8: Centroid

def make_pallet_keypoints_3d(width=1.1, depth=1.1, height=0.15):
    """Generate 9 keypoints (8 cuboid corners + centroid) in object frame.

    Uses OpenCV camera convention where the object coordinate system has:
      - X axis: right
      - Y axis: down
      - Z axis: forward

    This matches how DOPE's Cuboid3d.generate_vertexes() defines vertices
    when coord_system is None (default OpenCV convention).

    Args:
        width:  pallet width along X axis (meters).
        depth:  pallet depth along Z axis (meters).
        height: pallet height along Y axis (meters).

    Returns:
        np.ndarray of shape (9, 3).
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0

    # right/left along X, top/bottom along Y, front/rear along Z
    right, left = w, -w
    top, bottom = -h, h
    front, rear = d, -d

    corners = np.array([
        [right, top, front],      # 0: FrontTopRight
        [left,  top, front],      # 1: FrontTopLeft
        [left,  bottom, front],   # 2: FrontBottomLeft
        [right, bottom, front],   # 3: FrontBottomRight
        [right, top, rear],       # 4: RearTopRight
        [left,  top, rear],       # 5: RearTopLeft
        [left,  bottom, rear],    # 6: RearBottomLeft
        [right, bottom, rear],    # 7: RearBottomRight
    ], dtype=np.float64)

    centroid = corners.mean(axis=0, keepdims=True)  # (1, 3)
    return np.vstack([corners, centroid])            # (9, 3)


def make_pallet_keypoints_3d_isaac(width=1.1, depth=1.3, height=0.11):
    """Generate 9 keypoints in Isaac canonical ordering (matches synthetic data).

    Isaac _canonical_corners ordering (see scripts/data_prep/isaac_sim/sdg_math.py),
    expressed in pnp_solver / OpenCV frame (X=right, Y=down, Z=forward):

        0 = left-top-front    (mn_x, mx_y_isaac, mx_z) → (-w, -h, +d)
        1 = right-top-front   (0→1 = +X)
        2 = right-bottom-front
        3 = left-bottom-front (0→3 = +Y = down = Isaac -Y)
        4 = left-top-back     (0→4 = -Z)
        5 = right-top-back
        6 = right-bottom-back
        7 = left-bottom-back
        Front face = {0,1,2,3} at Z_max
        Top face   = {0,1,4,5} at Y_min

    Use this for GT annotation consistent with Isaac synthetic training data.
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, +d],  # 0: left-top-front
        [+w, -h, +d],  # 1: right-top-front
        [+w, +h, +d],  # 2: right-bottom-front
        [-w, +h, +d],  # 3: left-bottom-front
        [-w, -h, -d],  # 4: left-top-back
        [+w, -h, -d],  # 5: right-top-back
        [+w, +h, -d],  # 6: right-bottom-back
        [-w, +h, -d],  # 7: left-bottom-back
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])


def make_camera_matrix(fx, fy, cx, cy):
    """Build 3x3 camera intrinsic matrix."""
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


class PalletPnPSolver:
    """Solve pallet 6D pose from 2D keypoint detections via EPnP + RANSAC."""

    def __init__(self, camera_matrix, dist_coeffs=None,
                 pallet_dims=(1.1, 1.1, 0.15),
                 use_ransac=True, ransac_reproj_threshold=8.0,
                 ransac_iterations=100,
                 keypoints_3d=None):
        """
        Args:
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs:   distortion coefficients (default: zero).
            pallet_dims:   (width, depth, height) in meters.
            use_ransac:    whether to use RANSAC variant.
            ransac_reproj_threshold: inlier threshold in pixels.
            ransac_iterations: max RANSAC iterations.
            keypoints_3d:  optional (9,3) override for 3D model points.
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (np.array(dist_coeffs, dtype=np.float64)
                            if dist_coeffs is not None
                            else np.zeros((4, 1), dtype=np.float64))
        self.keypoints_3d = (np.array(keypoints_3d, dtype=np.float64)
                             if keypoints_3d is not None
                             else make_pallet_keypoints_3d(*pallet_dims))
        self.use_ransac = use_ransac
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_iterations = ransac_iterations

    def solve(self, keypoints_2d, sigmas=None, w_min=0.3, w_max=1.8):
        """Estimate 6D pose from 2D keypoint detections.

        Args:
            keypoints_2d: list of 9 elements. Each element is either
                (u, v) tuple, (u, v, confidence) tuple, or None.
            sigmas: optional list of 9 per-keypoint uncertainties.
                If provided, used for confidence-weighted refinement.
                Weight = clip(1/(sigma+eps), w_min, w_max), mean=1 normalized.
            w_min: minimum weight (hard points still get this much).
            w_max: maximum weight cap.

        Returns:
            success: bool, whether PnP succeeded.
            R: (3, 3) rotation matrix (world-to-camera).
            t: (3,) translation vector.
            inliers: array of inlier indices, or None.
        """
        obj_2d = []
        obj_3d = []
        raw_weights = []
        for i in range(9):
            if i >= len(keypoints_2d):
                continue
            pt = keypoints_2d[i]
            if pt is None:
                continue
            if hasattr(pt, '__len__') and len(pt) >= 2:
                u, v = float(pt[0]), float(pt[1])
                if u < 0 or v < 0:
                    continue
                obj_2d.append([u, v])
                obj_3d.append(self.keypoints_3d[i])
                if sigmas is not None and i < len(sigmas) and sigmas[i] is not None:
                    raw_weights.append(1.0 / (float(sigmas[i]) + 1e-4))
                else:
                    raw_weights.append(1.0)

        # Clip + normalize weights
        weights = []
        if sigmas is not None and raw_weights:
            rw = np.array(raw_weights, dtype=np.float64)
            rw = np.clip(rw, w_min, w_max)
            rw = len(rw) * rw / (rw.sum() + 1e-8)  # mean=1 normalize
            weights = rw.tolist()
        else:
            weights = [1.0] * len(obj_2d)

        if len(obj_2d) < 4:
            return False, None, None, None

        obj_2d = np.array(obj_2d, dtype=np.float64)
        obj_3d = np.array(obj_3d, dtype=np.float64)

        # If sigmas provided, sort by confidence and use top-k reliable points first
        # Then refine with all points using weighted iterative PnP
        inliers = None
        if self.use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_3d, obj_2d,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=self.ransac_reproj_threshold,
                iterationsCount=self.ransac_iterations,
            )
            # Weighted refinement: use initial PnP as seed, refine with weights
            if success and sigmas is not None and len(weights) == len(obj_2d):
                w = np.array(weights, dtype=np.float64)
                w = w / w.max()  # normalize to [0, 1]
                # Weighted reprojection: keep only high-confidence points for refinement
                high_conf = w > 0.2  # at least 20% of max confidence
                if high_conf.sum() >= 4:
                    try:
                        success_r, rvec_r, tvec_r = cv2.solvePnP(
                            obj_3d[high_conf], obj_2d[high_conf],
                            self.camera_matrix, self.dist_coeffs,
                            rvec=rvec, tvec=tvec,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        if success_r:
                            rvec, tvec = rvec_r, tvec_r
                    except cv2.error:
                        pass  # keep original PnP result
        else:
            success, rvec, tvec = cv2.solvePnP(
                obj_3d, obj_2d,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )

        if not success:
            return False, None, None, None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Flip if object is behind camera (z < 0)
        if t[2] < 0:
            t = -t
            R = -R

        return True, R, t, inliers

    # ── Reprojection-Guided PnP Refinement ──────────────────────────────

    @staticmethod
    def _projected_diagonal(reproj_8pts):
        """Max pairwise distance of 8 reprojected 2D points."""
        max_dist = 0.0
        for i in range(len(reproj_8pts)):
            for j in range(i + 1, len(reproj_8pts)):
                d = np.linalg.norm(reproj_8pts[i] - reproj_8pts[j])
                if d > max_dist:
                    max_dist = d
        return max_dist

    @staticmethod
    def _huber_weight(u, tau):
        """Standard Huber-style weight: 1.0 for u<=tau, tau/u for u>tau."""
        if u <= tau:
            return 1.0
        return tau / (u + 1e-8)

    def solve_reproj_guided(self, keypoints_2d, peak_confidences=None,
                            tau_huber=0.05, tau_peak=0.3, tau_w=0.1,
                            min_inliers=4, fallback_on_worse=True):
        """Reprojection-guided PnP refinement.

        1. Initial EPnP+RANSAC → (R0, t0)
        2. Compute per-keypoint reprojection residual, normalized by diagonal
        3. Huber-weighted inlier selection with coverage constraint
        4. Re-estimate pose with selected inliers (ITERATIVE, seeded)
        5. Fallback to initial if refinement is worse

        Args:
            keypoints_2d: list of 9 elements, each (u,v), (u,v,conf), or None.
            peak_confidences: list of 9 floats (belief map peak values).
                If None, all detected points get confidence 1.0.
            tau_huber: Huber kernel threshold (normalized by projected diagonal).
            tau_peak: Minimum peak confidence to be a candidate.
            tau_w: Minimum combined weight to be selected.
            min_inliers: Minimum points for refinement.
            fallback_on_worse: Keep original if refined mean residual is higher.

        Returns:
            success: bool
            R: (3,3) rotation matrix
            t: (3,) translation vector
            inliers: array of inlier indices (in 0-8 keypoint space), or None
            meta: dict with diagnostics
        """
        meta = {
            "initial_residual_mean": None,
            "refined_residual_mean": None,
            "fallback_used": False,
            "n_candidates": 0,
            "n_selected": 0,
            "weights": [],
            "residuals": [],
            "D": 0.0,
            "sanity_skip": False,
        }

        # Step 1: Initial pose (unweighted)
        success0, R0, t0, inliers0 = self.solve(keypoints_2d)
        if not success0:
            return False, None, None, None, meta

        # Collect detected keypoints with indices
        detected = []  # list of (idx, u, v, confidence)
        for i in range(min(9, len(keypoints_2d))):
            pt = keypoints_2d[i]
            if pt is None:
                continue
            if hasattr(pt, '__len__') and len(pt) >= 2:
                u, v = float(pt[0]), float(pt[1])
                if u < 0 or v < 0:
                    continue
                c = 1.0
                if peak_confidences is not None and i < len(peak_confidences):
                    c = float(peak_confidences[i]) if peak_confidences[i] is not None else 1.0
                detected.append((i, u, v, c))

        # Step 2: Initial sanity check
        reproj_all = self.reproject(R0, t0)
        D = self._projected_diagonal(reproj_all[:8])
        meta["D"] = float(D)

        if len(detected) < 4 or D < 10.0 or D > 2000.0:
            meta["sanity_skip"] = True
            return success0, R0, t0, inliers0, meta

        # Compute initial residuals (normalized)
        residuals = []
        for idx, u, v, c in detected:
            r = np.linalg.norm(reproj_all[idx] - np.array([u, v]))
            residuals.append(r / (D + 1e-6))

        init_mean = float(np.mean(residuals))
        meta["initial_residual_mean"] = init_mean
        meta["residuals"] = [float(r) for r in residuals]

        if init_mean > 0.3:
            meta["sanity_skip"] = True
            return success0, R0, t0, inliers0, meta

        # Step 3-4: Huber weight + candidate selection
        weights = []
        candidates = []  # (idx_in_detected, keypoint_idx, u, v, weight)
        for di, (idx, u, v, c) in enumerate(detected):
            psi = self._huber_weight(residuals[di], tau_huber)
            w = c * psi
            weights.append(w)
            if c >= tau_peak and w >= tau_w:
                candidates.append((di, idx, u, v, w))

        meta["weights"] = [float(w) for w in weights]
        meta["n_candidates"] = len(candidates)

        if len(candidates) < min_inliers:
            meta["fallback_used"] = True
            return success0, R0, t0, inliers0, meta

        # Step 5: Inline coverage check
        cand_2d = np.array([[u, v] for _, _, u, v, _ in candidates])
        x_range = cand_2d[:, 0].max() - cand_2d[:, 0].min()
        y_range = cand_2d[:, 1].max() - cand_2d[:, 1].min()
        if x_range < D * 0.30 or y_range < D * 0.10:
            meta["fallback_used"] = True
            return success0, R0, t0, inliers0, meta

        # Step 6: Pose re-estimation with selected subset
        cand_idx = [idx for _, idx, _, _, _ in candidates]
        obj_3d = self.keypoints_3d[cand_idx].astype(np.float64)
        obj_2d = cand_2d.astype(np.float64)
        meta["n_selected"] = len(cand_idx)

        rvec0, _ = cv2.Rodrigues(R0)
        try:
            success_r, rvec_r, tvec_r = cv2.solvePnP(
                obj_3d, obj_2d,
                self.camera_matrix, self.dist_coeffs,
                rvec=rvec0, tvec=t0.reshape(3, 1),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            meta["fallback_used"] = True
            return success0, R0, t0, inliers0, meta

        if not success_r:
            meta["fallback_used"] = True
            return success0, R0, t0, inliers0, meta

        R_r, _ = cv2.Rodrigues(rvec_r)
        t_r = tvec_r.flatten()

        # Flip if behind camera
        if t_r[2] < 0:
            meta["fallback_used"] = True
            return success0, R0, t0, inliers0, meta

        # Step 7: Fallback comparison
        if fallback_on_worse:
            reproj_r = self.reproject(R_r, t_r)
            refined_residuals = []
            for idx, u, v, c in detected:
                r = np.linalg.norm(reproj_r[idx] - np.array([u, v]))
                refined_residuals.append(r / (D + 1e-6))
            refined_mean = float(np.mean(refined_residuals))
            meta["refined_residual_mean"] = refined_mean

            if refined_mean > init_mean:
                meta["fallback_used"] = True
                return success0, R0, t0, inliers0, meta
        else:
            meta["refined_residual_mean"] = None

        return True, R_r, t_r, np.array(cand_idx), meta

    # Cuboid edge groups by dimension (Isaac ordering)
    WIDTH_EDGES = [(0, 1), (3, 2), (4, 5), (7, 6)]   # X axis
    HEIGHT_EDGES = [(0, 3), (1, 2), (4, 7), (5, 6)]   # Y axis
    DEPTH_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]    # Z axis

    def solve_adaptive(self, keypoints_2d, max_iter=3, min_edges=2,
                       step_clamp=(0.85, 1.15), total_clamp=(0.5, 2.0)):
        """2D edge ratio로 치수 역추정 후 동적 K_3D로 PnP.

        1. 현재 K_3D로 초기 PnP
        2. detected edge의 2D 길이 vs reprojected edge 길이 비율 계산
        3. 차원별(W/D/H) median ratio로 K_3D 스케일링 (clamped)
        4. 새 K_3D로 PnP 재실행, 수렴까지 반복

        Args:
            keypoints_2d: list of 9 elements, (u,v), (u,v,conf), or None.
            max_iter: 최대 반복 횟수.
            min_edges: 차원별 최소 관측 edge 수 (미달 시 해당 차원 고정).
            step_clamp: per-iteration scale 범위 (발산 방지).
            total_clamp: 누적 scale 허용 범위.

        Returns:
            success, R, t, inliers, meta dict
        """
        detected_map = {}
        for i in range(min(9, len(keypoints_2d))):
            pt = keypoints_2d[i]
            if pt is not None and hasattr(pt, '__len__') and len(pt) >= 2:
                detected_map[i] = np.array([float(pt[0]), float(pt[1])])

        success0, R0, t0, inliers0 = self.solve(keypoints_2d)
        if not success0:
            return False, None, None, None, {"reason": "initial_pnp_fail"}

        original_kp3d = self.keypoints_3d.copy()
        scale = np.array([1.0, 1.0, 1.0])
        R, t, inliers = R0, t0, inliers0
        converged_iter = 0

        for it in range(max_iter):
            reproj = self.reproject(R, t)[:8]

            ratios_per_dim = [[], [], []]
            edge_groups = [self.WIDTH_EDGES, self.HEIGHT_EDGES, self.DEPTH_EDGES]

            for dim_idx, edges in enumerate(edge_groups):
                for i, j in edges:
                    if i in detected_map and j in detected_map:
                        obs_len = np.linalg.norm(detected_map[i] - detected_map[j])
                        exp_len = np.linalg.norm(reproj[i] - reproj[j])
                        if exp_len > 3.0:
                            ratios_per_dim[dim_idx].append(obs_len / exp_len)

            step = np.array([1.0, 1.0, 1.0])
            for dim_idx in range(3):
                if len(ratios_per_dim[dim_idx]) >= min_edges:
                    r = float(np.median(ratios_per_dim[dim_idx]))
                    step[dim_idx] = np.clip(r, step_clamp[0], step_clamp[1])

            if np.allclose(step, 1.0, atol=0.01):
                converged_iter = it
                break

            scale *= step
            scale = np.clip(scale, total_clamp[0], total_clamp[1])

            scaled_kp3d = original_kp3d.copy()
            scaled_kp3d[:, 0] *= scale[0]
            scaled_kp3d[:, 1] *= scale[1]
            scaled_kp3d[:, 2] *= scale[2]

            self.keypoints_3d = scaled_kp3d
            success, R, t, inliers = self.solve(keypoints_2d)
            if not success:
                self.keypoints_3d = original_kp3d
                return success0, R0, t0, inliers0, {
                    "reason": "refined_pnp_fail", "iter": it,
                    "scale": scale.tolist(), "fallback": True,
                }
            converged_iter = it + 1

        est_w = float(np.linalg.norm(self.keypoints_3d[0] - self.keypoints_3d[1]))
        est_h = float(np.linalg.norm(self.keypoints_3d[0] - self.keypoints_3d[3]))
        est_d = float(np.linalg.norm(self.keypoints_3d[0] - self.keypoints_3d[4]))

        self.keypoints_3d = original_kp3d

        # Reprojection 비교: adaptive가 fixed보다 나은지
        reproj_fixed = self._mean_reproj_error(R0, t0, detected_map, original_kp3d)
        scaled_kp3d = original_kp3d.copy()
        scaled_kp3d[:, 0] *= scale[0]
        scaled_kp3d[:, 1] *= scale[1]
        scaled_kp3d[:, 2] *= scale[2]
        reproj_adapt = self._mean_reproj_error(R, t, detected_map, scaled_kp3d)

        fallback = reproj_adapt > reproj_fixed
        if fallback:
            R, t, inliers = R0, t0, inliers0

        return True, R, t, inliers, {
            "width": est_w, "height": est_h, "depth": est_d,
            "scale": scale.tolist(), "iters": converged_iter,
            "reproj_fixed": reproj_fixed, "reproj_adapt": reproj_adapt,
            "fallback": fallback,
        }

    def _mean_reproj_error(self, R, t, detected_map, kp3d):
        """detected points의 평균 reprojection error."""
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(
            kp3d, rvec, t.reshape(3, 1),
            self.camera_matrix, self.dist_coeffs,
        )
        proj = proj.reshape(-1, 2)
        errors = []
        for i, pt in detected_map.items():
            if i < len(proj):
                errors.append(np.linalg.norm(proj[i] - pt))
        return float(np.mean(errors)) if errors else float("inf")

    def reproject(self, R, t):
        """Reproject all 9 3D keypoints onto image plane.

        Args:
            R: (3, 3) rotation matrix.
            t: (3,) translation vector.

        Returns:
            np.ndarray of shape (9, 2) with pixel coordinates.
        """
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(
            self.keypoints_3d, rvec, t.reshape(3, 1),
            self.camera_matrix, self.dist_coeffs,
        )
        return projected.reshape(-1, 2)
