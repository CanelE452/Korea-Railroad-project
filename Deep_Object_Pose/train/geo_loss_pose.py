"""geo_loss.py — pose-level loss 헬퍼 (keypoint / reprojection / volume / ADD).

keypoint_l2_loss        : pred ↔ gt L2 MSE (2D keypoint)
CUBOID_FACES            : cuboid 6 face vertex 인덱스 (front/rear/top/bot/left/right)
diagonal_consistency_loss : 각 face 의 두 diagonal 의 midpoint 일치 (cuboid prior)
reprojection_loss       : (rvec, tvec, kp3d, K, gt_kp2d) → reprojection MSE
volume_loss             : 예측 cuboid 부피 / GT 부피 비 (1 에 가까워야)
add_loss                : ADD (Average Distance of model points)
huber                   : element-wise Huber (robust regression)
object_diagonal         : 2D cuboid bbox 대각선 (loss normalization 용)
"""
import torch
import torch.nn.functional as F

from geo_loss_bpnp import rodrigues_batch, project_points


def keypoint_l2_loss(pred_kp, gt_kp):
    """pred_kp, gt_kp: (B, C, 2) MSE."""
    return F.mse_loss(pred_kp, gt_kp)


CUBOID_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),  # front, rear
    (0, 1, 5, 4), (3, 2, 6, 7),  # top, bottom
    (1, 2, 6, 5), (0, 3, 7, 4),  # left, right
]


def diagonal_consistency_loss(kp):
    """6 face 각각 두 diagonal 의 midpoint 가 일치해야 (planar cuboid prior).
    kp: (B, 8+, 2) — 최소 8 corner."""
    loss = torch.tensor(0.0, device=kp.device, dtype=kp.dtype)
    for a, b, c, d in CUBOID_FACES:
        mid_ac = (kp[:, a] + kp[:, c]) / 2
        mid_bd = (kp[:, b] + kp[:, d]) / 2
        loss = loss + F.mse_loss(mid_ac, mid_bd)
    return loss / len(CUBOID_FACES)


def reprojection_loss(rvec, tvec, kp3d, K, gt_kp2d, valid):
    """reprojection MSE per-image, valid mask 평균."""
    R = rodrigues_batch(rvec)
    p2d = project_points(R, tvec, kp3d, K)
    err = ((p2d - gt_kp2d) ** 2).sum(dim=-1).mean(dim=-1)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec.device, dtype=rvec.dtype)
    return (err * valid).sum() / valid.sum()


def volume_loss(rvec_pred, tvec_pred, rvec_gt, tvec_gt, kp3d, valid):
    """예측 cuboid 부피 / GT cuboid 부피 ratio. 1 에 가까울수록 좋음."""
    R_pred = rodrigues_batch(rvec_pred)
    R_gt = rodrigues_batch(rvec_gt)
    P = kp3d[:8].unsqueeze(0)
    pred_3d = (torch.bmm(P.expand(R_pred.shape[0], -1, -1),
                         R_pred.transpose(1, 2)) + tvec_pred.unsqueeze(1))
    gt_3d = (torch.bmm(P.expand(R_gt.shape[0], -1, -1),
                       R_gt.transpose(1, 2)) + tvec_gt.unsqueeze(1))

    def _vol(pts):
        e01 = torch.norm(pts[:, 1] - pts[:, 0], dim=-1)
        e03 = torch.norm(pts[:, 3] - pts[:, 0], dim=-1)
        e04 = torch.norm(pts[:, 4] - pts[:, 0], dim=-1)
        return e01 * e03 * e04

    vol_pred = _vol(pred_3d)
    vol_gt = _vol(gt_3d).clamp(min=1e-6)
    ratio = vol_pred / vol_gt
    err = (ratio - 1.0) ** 2
    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec_pred.device, dtype=rvec_pred.dtype)
    return (err * valid).sum() / valid.sum()


def add_loss(rvec_pred, tvec_pred, rvec_gt, tvec_gt, kp3d, valid):
    """ADD (Average Distance of model points) — 모든 corner 의 3D L2 평균."""
    R_pred = rodrigues_batch(rvec_pred)
    R_gt = rodrigues_batch(rvec_gt)
    P = kp3d.unsqueeze(0)
    pred_3d = (torch.bmm(P.expand(R_pred.shape[0], -1, -1),
                         R_pred.transpose(1, 2)) + tvec_pred.unsqueeze(1))
    gt_3d = (torch.bmm(P.expand(R_gt.shape[0], -1, -1),
                       R_gt.transpose(1, 2)) + tvec_gt.unsqueeze(1))
    dist = torch.norm(pred_3d - gt_3d, dim=-1).mean(dim=-1)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec_pred.device, dtype=rvec_pred.dtype)
    return (dist * valid).sum() / valid.sum()


def huber(x, delta=0.03):
    """element-wise Huber loss (robust)."""
    abs_x = x.abs()
    return torch.where(abs_x < delta, 0.5 * x * x / delta, abs_x - 0.5 * delta)


def object_diagonal(kp2d):
    """2D cuboid bbox 의 대각선 길이 (B,) — loss normalization 용."""
    corners = kp2d[:, :8]
    mn = corners.min(dim=1).values
    mx = corners.max(dim=1).values
    return torch.norm(mx - mn, dim=-1).clamp(min=1.0)
