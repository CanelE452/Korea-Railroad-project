"""Evaluation metrics for 6D pose estimation.

Implements:
  - ADD (Average Distance of Model Points)
  - 5cm 5 degree metric
  - 2D reprojection error

Usage:
    from metrics import compute_ADD, compute_5cm_5deg, compute_reproj_error
"""

import numpy as np


def compute_ADD(R_gt, t_gt, R_pred, t_pred, model_points):
    """Compute ADD (Average Distance of Model Points).

    Measures the mean L2 distance between corresponding model points
    transformed by the ground-truth and predicted poses.

    Args:
        R_gt: (3, 3) ground-truth rotation matrix.
        t_gt: (3,) ground-truth translation vector.
        R_pred: (3, 3) predicted rotation matrix.
        t_pred: (3,) predicted translation vector.
        model_points: (N, 3) 3D model points in object frame.

    Returns:
        add: float, mean distance in meters.
        is_correct: bool, True if ADD < 10% of model diameter.
        diameter: float, model diameter in meters.
    """
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()
    model_points = np.asarray(model_points)

    transformed_gt = (R_gt @ model_points.T).T + t_gt
    transformed_pred = (R_pred @ model_points.T).T + t_pred

    distances = np.linalg.norm(transformed_gt - transformed_pred, axis=1)
    add = float(np.mean(distances))

    diameter = compute_diameter(model_points)
    is_correct = add < 0.1 * diameter

    return add, is_correct, diameter


def compute_ADD_S(R_gt, t_gt, R_pred, t_pred, model_points):
    """Compute ADD-S (symmetric variant) using closest point matching.

    For symmetric objects, each predicted point is matched to the
    closest ground-truth point instead of its corresponding point.

    Args:
        R_gt: (3, 3) ground-truth rotation matrix.
        t_gt: (3,) ground-truth translation vector.
        R_pred: (3, 3) predicted rotation matrix.
        t_pred: (3,) predicted translation vector.
        model_points: (N, 3) 3D model points in object frame.

    Returns:
        add_s: float, mean closest-point distance in meters.
        is_correct: bool, True if ADD-S < 10% of model diameter.
        diameter: float, model diameter in meters.
    """
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()
    model_points = np.asarray(model_points)

    transformed_gt = (R_gt @ model_points.T).T + t_gt
    transformed_pred = (R_pred @ model_points.T).T + t_pred

    # For each predicted point, find closest ground-truth point
    distances = []
    for pt_pred in transformed_pred:
        dists = np.linalg.norm(transformed_gt - pt_pred, axis=1)
        distances.append(np.min(dists))

    add_s = float(np.mean(distances))
    diameter = compute_diameter(model_points)
    is_correct = add_s < 0.1 * diameter

    return add_s, is_correct, diameter


def compute_diameter(model_points):
    """Compute the diameter of a 3D model (max pairwise distance).

    Args:
        model_points: (N, 3) array of 3D points.

    Returns:
        float: diameter in the same units as the input points.
    """
    model_points = np.asarray(model_points)
    # For efficiency with large point clouds, use a sampling approach
    if len(model_points) > 500:
        idx = np.random.choice(len(model_points), 500, replace=False)
        pts = model_points[idx]
    else:
        pts = model_points

    max_dist = 0.0
    for i in range(len(pts)):
        dists = np.linalg.norm(pts[i:] - pts[i], axis=1)
        d = np.max(dists)
        if d > max_dist:
            max_dist = d
    return float(max_dist)


def compute_5cm_5deg(R_gt, t_gt, R_pred, t_pred):
    """Compute the 5cm-5-degree metric.

    A prediction is correct if both:
      - Translation error < 5 cm
      - Rotation error < 5 degrees

    Args:
        R_gt: (3, 3) ground-truth rotation matrix.
        t_gt: (3,) ground-truth translation (meters).
        R_pred: (3, 3) predicted rotation matrix.
        t_pred: (3,) predicted translation (meters).

    Returns:
        is_correct: bool.
        trans_error_cm: float, translation error in cm.
        rot_error_deg: float, rotation error in degrees.
    """
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()

    # Translation error in cm
    trans_error = np.linalg.norm(t_gt - t_pred)
    trans_error_cm = float(trans_error * 100)

    # Rotation error in degrees
    R_diff = R_gt @ R_pred.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    rot_error_deg = float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))

    is_correct = (trans_error_cm < 5.0) and (rot_error_deg < 5.0)
    return is_correct, trans_error_cm, rot_error_deg


def compute_reproj_error(kp_gt_2d, kp_pred_2d):
    """Compute mean 2D reprojection error.

    Args:
        kp_gt_2d: (N, 2) ground-truth 2D keypoints.
        kp_pred_2d: (N, 2) predicted 2D keypoints.

    Returns:
        mean_error: float, mean pixel distance.
        per_point_errors: (N,) array of per-point errors.
    """
    kp_gt_2d = np.asarray(kp_gt_2d, dtype=np.float64)
    kp_pred_2d = np.asarray(kp_pred_2d, dtype=np.float64)

    per_point_errors = np.linalg.norm(kp_gt_2d - kp_pred_2d, axis=1)
    mean_error = float(np.mean(per_point_errors))
    return mean_error, per_point_errors


def compute_auc(errors, max_threshold=0.1):
    """Compute Area Under the ADD Curve.

    Args:
        errors: list of ADD error values.
        max_threshold: maximum threshold (fraction of diameter).

    Returns:
        auc: float in [0, 1].
    """
    errors = np.array(errors)
    thresholds = np.linspace(0, max_threshold, 100)
    accuracies = []
    for th in thresholds:
        acc = np.mean(errors < th)
        accuracies.append(acc)
    return float(np.trapz(accuracies, thresholds) / max_threshold)


class PoseEvaluator:
    """Aggregator for evaluating a batch of pose predictions."""

    def __init__(self, model_points):
        """
        Args:
            model_points: (N, 3) 3D model points for ADD computation.
        """
        self.model_points = np.asarray(model_points)
        self.diameter = compute_diameter(self.model_points)
        self.results = []

    def add_prediction(self, R_gt, t_gt, R_pred, t_pred,
                       kp_gt_2d=None, kp_pred_2d=None):
        """Record one prediction for later summary.

        Args:
            R_gt, t_gt: ground-truth pose.
            R_pred, t_pred: predicted pose.
            kp_gt_2d, kp_pred_2d: optional 2D keypoints for reproj error.
        """
        add, add_correct, _ = compute_ADD(
            R_gt, t_gt, R_pred, t_pred, self.model_points)
        add_s, add_s_correct, _ = compute_ADD_S(
            R_gt, t_gt, R_pred, t_pred, self.model_points)
        correct_5cm5deg, trans_err, rot_err = compute_5cm_5deg(
            R_gt, t_gt, R_pred, t_pred)

        result = {
            "add": add,
            "add_correct": add_correct,
            "add_s": add_s,
            "add_s_correct": add_s_correct,
            "5cm5deg": correct_5cm5deg,
            "trans_error_cm": trans_err,
            "rot_error_deg": rot_err,
        }

        if kp_gt_2d is not None and kp_pred_2d is not None:
            reproj_err, _ = compute_reproj_error(kp_gt_2d, kp_pred_2d)
            result["reproj_error_px"] = reproj_err

        self.results.append(result)

    def summarize(self):
        """Compute aggregate metrics over all recorded predictions.

        Returns:
            dict with summary statistics.
        """
        if not self.results:
            return {"num_predictions": 0}

        adds = [r["add"] for r in self.results]
        adds_s = [r["add_s"] for r in self.results]
        summary = {
            "num_predictions": len(self.results),
            "model_diameter": self.diameter,
            "ADD_mean": float(np.mean(adds)),
            "ADD_median": float(np.median(adds)),
            "ADD_correct_rate": float(np.mean([r["add_correct"] for r in self.results])),
            "ADD-S_mean": float(np.mean(adds_s)),
            "ADD-S_median": float(np.median(adds_s)),
            "ADD-S_correct_rate": float(np.mean([r["add_s_correct"] for r in self.results])),
            "ADD-S_auc": compute_auc(np.array(adds_s) / self.diameter),
            "5cm5deg_rate": float(np.mean([r["5cm5deg"] for r in self.results])),
            "trans_error_mean_cm": float(np.mean([r["trans_error_cm"] for r in self.results])),
            "rot_error_mean_deg": float(np.mean([r["rot_error_deg"] for r in self.results])),
            "ADD_auc": compute_auc(np.array(adds) / self.diameter),
        }

        if "reproj_error_px" in self.results[0]:
            summary["reproj_error_mean_px"] = float(
                np.mean([r["reproj_error_px"] for r in self.results]))

        return summary

    def reset(self):
        """Clear all recorded results."""
        self.results = []
