"""Geometric loss for DOPE training.

Soft-argmax + BPnP(Backpropagatable PnP)로 belief map MSE 위에
3D 기하학적 제약을 추가. DOPE 모델 구조는 변경하지 않음 —
loss 계산용으로만 사용되며 inference 시에는 제거.

v2: Structural losses 추가 (2026-04-07)
  - FlipEquivarianceLoss: 좌우반전 일관성 (shortcut 학습 방지)
  - SparseEdgeLoss: cuboid edge log-ratio (구조 일관성)
  - object diagonal normalization + Huber loss

Reference:
    BPnP: Chen et al., "End-to-End Learnable Geometric Vision by
    Backpropagating PnP Optimization", CVPR 2020
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rodrigues (batch) ──────────────────────────────────────────────────

def rodrigues_batch(rvec):
    """Rotation vector → rotation matrix (batch).

    Args:
        rvec: (B, 3) rotation vectors.
    Returns:
        R: (B, 3, 3) rotation matrices.
    """
    B = rvec.shape[0]
    theta = torch.norm(rvec, dim=-1, keepdim=True).unsqueeze(-1)  # (B,1,1)
    eps = 1e-8
    k = rvec / (theta.squeeze(-1) + eps)  # (B, 3) unit axis

    K = torch.zeros(B, 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


# ── Projection ─────────────────────────────────────────────────────────

def project_points(R, t, kp3d, K):
    """3D keypoints → 2D pixel coordinates.

    Args:
        R: (B, 3, 3), t: (B, 3), kp3d: (N, 3), K: (3, 3)
    Returns:
        p2d: (B, N, 2)
    """
    B = R.shape[0]
    N = kp3d.shape[0]
    P = kp3d.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
    P_cam = torch.bmm(P, R.transpose(1, 2)) + t.unsqueeze(1)  # (B, N, 3)
    P_proj = torch.bmm(P_cam, K.T.unsqueeze(0).expand(B, -1, -1))  # (B, N, 3)
    z = P_proj[:, :, 2:3].clamp(min=1e-6)
    return P_proj[:, :, :2] / z


def projection_jacobian(rvec, tvec, kp3d, K):
    """Projection Jacobian w.r.t. pose (rvec, tvec).

    Args:
        rvec: (B, 3), tvec: (B, 3), kp3d: (N, 3), K: (3, 3)
    Returns:
        J: (B, 2N, 6)
    """
    B = rvec.shape[0]
    N = kp3d.shape[0]
    device = rvec.device
    dtype = rvec.dtype

    R = rodrigues_batch(rvec)
    P = kp3d.unsqueeze(0).expand(B, -1, -1)
    P_cam = torch.bmm(P, R.transpose(1, 2)) + tvec.unsqueeze(1)  # (B, N, 3)

    fx, fy = K[0, 0], K[1, 1]
    X = P_cam[:, :, 0]
    Y = P_cam[:, :, 1]
    Z = P_cam[:, :, 2].clamp(min=1e-6)
    Z2 = Z * Z

    # dp/dP_cam: (B, N, 2, 3)
    dpdP = torch.zeros(B, N, 2, 3, device=device, dtype=dtype)
    dpdP[:, :, 0, 0] = fx / Z
    dpdP[:, :, 0, 2] = -fx * X / Z2
    dpdP[:, :, 1, 1] = fy / Z
    dpdP[:, :, 1, 2] = -fy * Y / Z2

    # dP_cam/drvec = -[R @ P_w]× (skew-symmetric)
    RP = P_cam - tvec.unsqueeze(1)  # R @ P_w
    dPdr = torch.zeros(B, N, 3, 3, device=device, dtype=dtype)
    dPdr[:, :, 0, 1] = RP[:, :, 2]
    dPdr[:, :, 0, 2] = -RP[:, :, 1]
    dPdr[:, :, 1, 0] = -RP[:, :, 2]
    dPdr[:, :, 1, 2] = RP[:, :, 0]
    dPdr[:, :, 2, 0] = RP[:, :, 1]
    dPdr[:, :, 2, 1] = -RP[:, :, 0]

    # dP_cam/dt = I
    dPdt = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(B, N, -1, -1)

    dpdr = torch.matmul(dpdP, dPdr)  # (B, N, 2, 3)
    dpdt = torch.matmul(dpdP, dPdt)  # (B, N, 2, 3)

    J_per_point = torch.cat([dpdr, dpdt], dim=-1)  # (B, N, 2, 6)
    return J_per_point.reshape(B, 2 * N, 6)


# ── Soft-Argmax ────────────────────────────────────────────────────────

class SpatialSoftArgmax2D(nn.Module):
    """Differentiable soft-argmax for 2D heatmaps.

    Also computes per-keypoint spatial variance (sigma^2) as uncertainty.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmap, return_sigma=False):
        """
        Args:
            heatmap: (B, C, H, W) belief maps.
            return_sigma: if True, also return per-keypoint sigma.
        Returns:
            coords: (B, C, 2) — (x, y) in heatmap pixel coordinates.
            sigma: (B, C) — spatial std dev per keypoint (only if return_sigma).
        """
        B, C, H, W = heatmap.shape
        device = heatmap.device

        y_coords = torch.arange(H, device=device, dtype=heatmap.dtype).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=heatmap.dtype).view(1, 1, 1, W)

        flat = heatmap.view(B, C, -1)
        weights = F.softmax(flat / self.temperature, dim=-1)
        weights = weights.view(B, C, H, W)

        x = (weights * x_coords).sum(dim=(2, 3))
        y = (weights * y_coords).sum(dim=(2, 3))

        coords = torch.stack([x, y], dim=-1).float()

        if not return_sigma:
            return coords

        # Spatial variance: sigma^2 = E[(u - mu)^2] for each keypoint
        mu_x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        mu_y = y.unsqueeze(-1).unsqueeze(-1)
        var = (weights * ((x_coords - mu_x) ** 2 + (y_coords - mu_y) ** 2)).sum(dim=(2, 3))
        sigma = torch.sqrt(var.clamp(min=1e-6))  # (B, C)

        return coords, sigma


# ── BPnP (Backpropagatable PnP) ───────────────────────────────────────

class BPnP(torch.autograd.Function):
    """Forward: cv2.solvePnP, Backward: implicit differentiation."""

    @staticmethod
    def forward(ctx, kp2d, kp3d, K, damping=1e-4):
        """
        Args:
            kp2d: (B, N, 2) predicted 2D keypoints (pixel coords).
            kp3d: (N, 3) 3D model keypoints (fixed).
            K: (3, 3) camera intrinsics.
            damping: LM damping for (J^T J + λI)^{-1}.
        Returns:
            pose: (B, 6) — [rvec(3), tvec(3)].
            valid: (B,) — 1.0 if PnP succeeded, 0.0 otherwise.
        """
        B, N, _ = kp2d.shape
        device = kp2d.device
        dtype = kp2d.dtype

        kp2d_np = kp2d.detach().cpu().numpy()
        kp3d_np = kp3d.detach().cpu().numpy().astype(np.float64)
        K_np = K.detach().cpu().numpy().astype(np.float64)

        poses = np.zeros((B, 6), dtype=np.float32)
        valids = np.zeros(B, dtype=np.float32)

        for b in range(B):
            pts2d = kp2d_np[b].astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                kp3d_np, pts2d, K_np, None, flags=cv2.SOLVEPNP_EPNP
            )
            if success:
                t = tvec.flatten()
                if t[2] < 0:
                    rvec = -rvec
                    t = -t
                poses[b, :3] = rvec.flatten()
                poses[b, 3:] = t
                valids[b] = 1.0

        pose = torch.from_numpy(poses).float().to(device=device)
        valid = torch.from_numpy(valids).float().to(device=device)

        # Clamp to prevent NaN/Inf in backward
        pose = torch.clamp(pose, -100.0, 100.0)

        ctx.save_for_backward(kp2d, kp3d, K, pose, valid)
        ctx.damping = damping
        return pose, valid

    @staticmethod
    def backward(ctx, grad_pose, grad_valid):
        kp2d, kp3d, K, pose, valid = ctx.saved_tensors
        damping = ctx.damping
        B, N, _ = kp2d.shape
        device = kp2d.device

        # Force everything to float32 on correct device
        rvec = pose[:, :3].detach().float().to(device)
        tvec = pose[:, 3:].detach().float().to(device)
        kp3d_f = kp3d.detach().float().to(device)
        K_f = K.detach().float().to(device)
        grad_pose_f = grad_pose.detach().float().to(device)
        valid_f = valid.detach().float().to(device)

        # Projection Jacobian: (B, 2N, 6) — all float32
        J = projection_jacobian(rvec, tvec, kp3d_f, K_f).float()

        # grad_kp2d = J @ (J^T J + λI)^{-1} @ grad_pose
        JtJ = torch.bmm(J.transpose(1, 2), J)
        eye = torch.eye(6, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        JtJ_reg = JtJ + max(damping, 1e-3) * eye

        # Use pinv for maximum stability
        JtJ_inv = torch.linalg.pinv(JtJ_reg).float()
        x = torch.bmm(JtJ_inv, grad_pose_f.unsqueeze(-1))  # (B, 6, 1)
        grad_kp2d_flat = torch.bmm(J, x).squeeze(-1).float()  # (B, 2N)

        # Mask + clamp
        grad_kp2d_flat = grad_kp2d_flat * valid_f.unsqueeze(-1)
        grad_kp2d_flat = torch.nan_to_num(grad_kp2d_flat, nan=0.0, posinf=0.0, neginf=0.0)
        grad_kp2d_flat = torch.clamp(grad_kp2d_flat, -10.0, 10.0)
        grad_kp2d = grad_kp2d_flat.view(B, N, 2)

        return grad_kp2d, None, None, None


# ── Loss Functions ─────────────────────────────────────────────────────

def keypoint_l2_loss(pred_kp, gt_kp):
    """L2 distance between predicted and GT keypoints.

    Args:
        pred_kp, gt_kp: (B, C, 2)
    """
    return F.mse_loss(pred_kp, gt_kp)


CUBOID_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),  # front, rear
    (0, 1, 5, 4), (3, 2, 6, 7),  # top, bottom
    (1, 2, 6, 5), (0, 3, 7, 4),  # left, right
]


def diagonal_consistency_loss(kp):
    """Cuboid face diagonal midpoint consistency (2D).

    Args:
        kp: (B, 8+, 2) — at least 8 corner keypoints.
    """
    loss = torch.tensor(0.0, device=kp.device, dtype=kp.dtype)
    for a, b, c, d in CUBOID_FACES:
        mid_ac = (kp[:, a] + kp[:, c]) / 2
        mid_bd = (kp[:, b] + kp[:, d]) / 2
        loss = loss + F.mse_loss(mid_ac, mid_bd)
    return loss / len(CUBOID_FACES)


def reprojection_loss(rvec, tvec, kp3d, K, gt_kp2d, valid):
    """Reprojection error as loss.

    Args:
        rvec: (B, 3), tvec: (B, 3), kp3d: (N, 3), K: (3, 3)
        gt_kp2d: (B, N, 2), valid: (B,)
    """
    R = rodrigues_batch(rvec)
    p2d = project_points(R, tvec, kp3d, K)  # (B, N, 2)
    err = ((p2d - gt_kp2d) ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec.device, dtype=rvec.dtype)
    return (err * valid).sum() / valid.sum()


def volume_loss(rvec_pred, tvec_pred, rvec_gt, tvec_gt, kp3d, valid):
    """3D cuboid volume ratio loss.

    Args:
        rvec_pred, tvec_pred: (B, 3) predicted pose.
        rvec_gt, tvec_gt: (B, 3) GT pose.
        kp3d: (N, 3), valid: (B,)
    """
    R_pred = rodrigues_batch(rvec_pred)
    R_gt = rodrigues_batch(rvec_gt)
    P = kp3d[:8].unsqueeze(0)  # (1, 8, 3) corners only

    pred_3d = torch.bmm(P.expand(R_pred.shape[0], -1, -1), R_pred.transpose(1, 2)) + tvec_pred.unsqueeze(1)
    gt_3d = torch.bmm(P.expand(R_gt.shape[0], -1, -1), R_gt.transpose(1, 2)) + tvec_gt.unsqueeze(1)

    def _vol(pts):
        e01 = torch.norm(pts[:, 1] - pts[:, 0], dim=-1)
        e03 = torch.norm(pts[:, 3] - pts[:, 0], dim=-1)
        e04 = torch.norm(pts[:, 4] - pts[:, 0], dim=-1)
        return e01 * e03 * e04

    vol_pred = _vol(pred_3d)
    vol_gt = _vol(gt_3d).clamp(min=1e-6)
    ratio = vol_pred / vol_gt
    err = (ratio - 1.0) ** 2  # (B,)

    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec_pred.device, dtype=rvec_pred.dtype)
    return (err * valid).sum() / valid.sum()


def add_loss(rvec_pred, tvec_pred, rvec_gt, tvec_gt, kp3d, valid):
    """ADD (Average Distance of model points) as loss.

    Args:
        rvec_pred, tvec_pred: (B, 3), rvec_gt, tvec_gt: (B, 3)
        kp3d: (N, 3), valid: (B,)
    """
    R_pred = rodrigues_batch(rvec_pred)
    R_gt = rodrigues_batch(rvec_gt)
    P = kp3d.unsqueeze(0)

    pred_3d = torch.bmm(P.expand(R_pred.shape[0], -1, -1), R_pred.transpose(1, 2)) + tvec_pred.unsqueeze(1)
    gt_3d = torch.bmm(P.expand(R_gt.shape[0], -1, -1), R_gt.transpose(1, 2)) + tvec_gt.unsqueeze(1)

    dist = torch.norm(pred_3d - gt_3d, dim=-1).mean(dim=-1)  # (B,)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=rvec_pred.device, dtype=rvec_pred.dtype)
    return (dist * valid).sum() / valid.sum()


# ── Huber Loss (robust) ────────────────────────────────────────────────

def huber(x, delta=0.03):
    """Element-wise Huber loss."""
    abs_x = x.abs()
    return torch.where(abs_x < delta, 0.5 * x * x / delta, abs_x - 0.5 * delta)


# ── Object Diagonal ───────────────────────────────────────────────────

def object_diagonal(kp2d):
    """Compute projected cuboid diagonal for normalization.

    Args:
        kp2d: (B, 9, 2) or (B, 8+, 2) — GT 2D keypoints.
    Returns:
        diag: (B,) projected diagonal length.
    """
    corners = kp2d[:, :8]  # (B, 8, 2)
    mn = corners.min(dim=1).values  # (B, 2)
    mx = corners.max(dim=1).values  # (B, 2)
    diag = torch.norm(mx - mn, dim=-1).clamp(min=1.0)  # (B,)
    return diag


# ── Flip Equivariance Loss ────────────────────────────────────────────

# Pallet horizontal flip: left↔right keypoint swap
# Y=UP convention, corners: 0↔1, 2↔3, 4↔5, 6↔7, centroid(8) unchanged
FLIP_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 8]


class FlipEquivarianceLoss(nn.Module):
    """Flip consistency loss on soft-argmax coordinates.

    원본/flip 이미지에서 뽑은 좌표가 일관되도록 regularize.
    flip branch는 stop-gradient (한쪽만 밈).
    """

    def __init__(self, soft_argmax, delta=0.03):
        super().__init__()
        self.soft_argmax = soft_argmax
        self.delta = delta

    def forward(self, pred_belief, net, data_flip):
        """
        Args:
            pred_belief: (B, 9, H, W) — 원본 이미지 belief map (final stage).
            net: DOPE network (for flip forward pass).
            data_flip: (B, C, 448, 448) — horizontally flipped input.
        Returns:
            loss: scalar.
            info: dict.
        """
        B, C, H, W = pred_belief.shape

        # 원본 좌표
        kp_orig = self.soft_argmax(pred_belief)  # (B, 9, 2)

        # flip 이미지 forward (stop gradient)
        with torch.no_grad():
            flip_belief, _ = net(data_flip)
            flip_belief = flip_belief[-1][:, :9]  # (B, 9, H, W)

        kp_flip = self.soft_argmax(flip_belief)  # (B, 9, 2)

        # flip 좌표를 원래 좌표계로 되돌림: x → (W-1) - x
        kp_flip_unflip = kp_flip.clone()
        kp_flip_unflip[:, :, 0] = (W - 1) - kp_flip[:, :, 0]

        # channel permutation (left↔right swap)
        kp_flip_unflip = kp_flip_unflip[:, FLIP_PERM]

        # object diagonal for normalization
        diag = object_diagonal(kp_orig).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # normalized distance
        diff = (kp_orig - kp_flip_unflip) / (diag + 1e-6)  # (B, 9, 2)
        loss = huber(diff, self.delta).mean()

        return loss, {'struct/flip': loss.item()}


# ── Sparse Edge Loss ──────────────────────────────────────────────────

# Cuboid edges: 12 physical edges + 4 face diagonals
CUBOID_EDGES = [
    # Front face (Z_max): 0-1, 1-2, 2-3, 3-0
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Rear face (Z_min): 4-5, 5-6, 6-7, 7-4
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Depth edges: 0-4, 1-5, 2-6, 3-7
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CUBOID_DIAGONALS = [
    # Front face diagonals
    (0, 2), (1, 3),
    # Rear face diagonals
    (4, 6), (5, 7),
]

# Edge weights: face=1.0, depth=0.5, diagonal=0.3
EDGE_WEIGHTS = (
    [1.0] * 4 + [1.0] * 4 +  # front + rear face edges
    [0.5] * 4 +               # depth edges
    [0.3] * 4                 # diagonals
)


class SparseEdgeLoss(nn.Module):
    """Log-ratio edge length loss on sparse cuboid graph.

    예측/GT 간 edge 길이 비율이 1에 가깝도록. Object diagonal normalize.
    GT projected edge가 너무 짧으면 skip (perspective collapse 방지).
    """

    def __init__(self, min_edge_ratio=0.03, delta=0.03):
        super().__init__()
        self.edges = CUBOID_EDGES + CUBOID_DIAGONALS
        self.weights = torch.tensor(EDGE_WEIGHTS, dtype=torch.float32)
        self.min_edge_ratio = min_edge_ratio
        self.delta = delta

    def forward(self, pred_kp, gt_kp):
        """
        Args:
            pred_kp: (B, 9, 2) predicted soft-argmax coords.
            gt_kp: (B, 9, 2) GT soft-argmax coords.
        Returns:
            loss: scalar.
            info: dict.
        """
        device = pred_kp.device
        B = pred_kp.shape[0]
        eps = 1e-6

        diag = object_diagonal(gt_kp)  # (B,)

        weights = self.weights.to(device)
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)

        for idx, (i, j) in enumerate(self.edges):
            pred_len = torch.norm(pred_kp[:, i] - pred_kp[:, j], dim=-1)  # (B,)
            gt_len = torch.norm(gt_kp[:, i] - gt_kp[:, j], dim=-1)        # (B,)

            # skip edges that are too short in GT (perspective collapse)
            valid = (gt_len / (diag + eps)) > self.min_edge_ratio  # (B,)

            if valid.sum() < 1:
                continue

            log_ratio = torch.log((pred_len + eps) / (gt_len + eps))  # (B,)
            edge_loss = huber(log_ratio, self.delta)  # (B,)

            w = weights[idx]
            total_loss = total_loss + (edge_loss * valid.float() * w).sum()
            total_weight = total_weight + valid.float().sum() * w

        if total_weight < 1:
            return torch.tensor(0.0, device=device), {'struct/edge': 0.0}

        loss = total_loss / total_weight
        return loss, {'struct/edge': loss.item()}


# ── Coordinate Huber Loss ─────────────────────────────────────────────

def coord_vis_loss(pred_kp, gt_kp, vis_weight, delta=0.03, eps=1e-6):
    """Visibility-weighted normalized coordinate Huber loss.

    Weighted average: visible points contribute more, self-occluded less.
    Loss scale is invariant to number of visible points.

    Args:
        pred_kp: (B, 9, 2), gt_kp: (B, 9, 2)
        vis_weight: (B, 9) per-keypoint visibility (1.0/0.5/0.0)
        delta: Huber delta
    Returns:
        loss: scalar, info: dict
    """
    diag = object_diagonal(gt_kp).unsqueeze(-1)  # (B, 1)
    diff = torch.norm(pred_kp - gt_kp, dim=-1) / (diag + eps)  # (B, 9)
    robust_err = huber(diff, delta)  # (B, 9)
    weighted_err = vis_weight * robust_err  # (B, 9)
    loss = weighted_err.sum(dim=-1) / (vis_weight.sum(dim=-1) + eps)  # (B,)
    loss = loss.mean()

    info = {
        'vis/coord_vis': loss.item(),
        'vis/visible_pts': (vis_weight > 0.9).float().sum().item() / vis_weight.shape[0],
        'vis/occluded_pts': ((vis_weight > 0.1) & (vis_weight < 0.9)).float().sum().item() / vis_weight.shape[0],
        'vis/invisible_pts': (vis_weight < 0.1).float().sum().item() / vis_weight.shape[0],
    }
    return loss, info


def coord_huber_loss(pred_kp, gt_kp, delta=0.03):
    """Normalized coordinate Huber loss.

    Args:
        pred_kp: (B, 9, 2), gt_kp: (B, 9, 2)
    Returns:
        loss: scalar, info: dict
    """
    diag = object_diagonal(gt_kp).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    diff = (pred_kp - gt_kp) / (diag + 1e-6)
    loss = huber(diff, delta).mean()
    return loss, {'struct/coord': loss.item()}


# ── Reliability-Aware Coordinate Loss ─────────────────────────────────

def coord_rel_loss(pred_kp, gt_kp, sigma, delta=0.03, lambda_log=0.5, eps=1e-4,
                    w_min=0.4, w_max=1.8):
    """Uncertainty-weighted coordinate loss (v2: detach + clip + normalize).

    Uses sigma as reweighting signal, NOT as learnable uncertainty.
    - detach: model can't game sigma to reduce loss
    - clip: no point is ignored (w_min) or dominates (w_max)
    - normalize: per-image loss scale stays constant

    Args:
        pred_kp: (B, 9, 2) predicted soft-argmax coordinates.
        gt_kp: (B, 9, 2) GT soft-argmax coordinates.
        sigma: (B, 9) per-keypoint spatial std dev.
        delta: Huber delta.
        lambda_log: IGNORED (kept for API compat, log term removed in v2).
        eps: numerical stability.
        w_min: minimum weight (hard points still get this much).
        w_max: maximum weight cap.
    Returns:
        loss: scalar, info: dict
    """
    B, K = sigma.shape
    diag = object_diagonal(gt_kp).unsqueeze(-1)  # (B, 1)

    # Normalized coordinate error per keypoint: (B, 9)
    diff = torch.norm(pred_kp - gt_kp, dim=-1) / (diag + eps)
    robust_err = huber(diff, delta)

    # Detached, clipped, normalized weights
    u = sigma.detach()  # stop gradient — model can't game sigma
    w = (1.0 / (u + eps)).clamp(min=w_min, max=w_max)  # (B, 9)
    w_norm = K * w / (w.sum(dim=-1, keepdim=True) + eps)  # normalize to mean=1 per image

    # Weighted loss
    loss = (w_norm * robust_err).mean()

    info = {
        'rel/coord_rel': loss.item(),
        'rel/sigma_mean': sigma.mean().item(),
        'rel/sigma_min': sigma.min().item(),
        'rel/sigma_max': sigma.max().item(),
        'rel/w_mean': w_norm.mean().item(),
        'rel/w_min': w_norm.min().item(),
        'rel/w_max': w_norm.max().item(),
    }
    return loss, info


# ── Reliability Loss Module ───────────────────────────────────────────

class VisibilityCoordLoss(nn.Module):
    """Visibility-aware coordinate loss.

    GT visibility로 가중된 coord Huber loss.
    visible 점은 세게, self-occluded 점은 약하게, out-of-frame 점은 안 배움.
    """

    def __init__(self, temperature=1.0, delta=0.03):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.delta = delta

    def forward(self, pred_belief, gt_belief, vis_weight, epoch=0, warmup=0):
        """
        Args:
            pred_belief: (B, 9, H, W)
            gt_belief: (B, 9, H, W)
            vis_weight: (B, 9) GT visibility weights
            epoch, warmup: for compatibility
        Returns:
            loss, info dict
        """
        if epoch < warmup:
            return torch.tensor(0.0, device=pred_belief.device), {'vis/total': 0.0}

        pred_kp = self.soft_argmax(pred_belief)
        gt_kp = self.soft_argmax(gt_belief)

        loss, info = coord_vis_loss(pred_kp, gt_kp, vis_weight, self.delta)

        peak_vals = pred_belief.view(pred_belief.shape[0], pred_belief.shape[1], -1).max(dim=-1).values
        info['vis/belief_peak_mean'] = peak_vals.mean().item()
        info['vis/total'] = loss.item()
        return loss, info


class ReliabilityLoss(nn.Module):
    """Reliability-controlled coordinate loss.

    Extracts per-keypoint uncertainty from heatmap and uses it
    for heteroscedastic coordinate supervision.
    """

    def __init__(self, temperature=1.0, delta=0.03, lambda_log=0.5,
                 w_min=0.4, w_max=1.8):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.delta = delta
        self.lambda_log = lambda_log
        self.w_min = w_min
        self.w_max = w_max

    def forward(self, pred_belief, gt_belief, epoch=0, warmup=0):
        """
        Args:
            pred_belief: (B, 9, H, W) predicted belief maps.
            gt_belief: (B, 9, H, W) GT belief maps.
            epoch: current epoch.
            warmup: epochs before enabling.
        Returns:
            total_loss: scalar.
            loss_dict: dict with detailed metrics.
        """
        loss_dict = {}
        device = pred_belief.device
        total = torch.tensor(0.0, device=device)

        if epoch < warmup:
            loss_dict['rel/total'] = 0.0
            return total, loss_dict

        # Extract coordinates + uncertainty from predicted heatmap
        pred_kp, sigma = self.soft_argmax(pred_belief, return_sigma=True)
        gt_kp = self.soft_argmax(gt_belief)  # (B, 9, 2)

        # Reliability-aware coordinate loss (v2: detach+clip+normalize)
        l_rel, info = coord_rel_loss(
            pred_kp, gt_kp, sigma,
            delta=self.delta, lambda_log=self.lambda_log,
            w_min=self.w_min, w_max=self.w_max
        )
        loss_dict.update(info)

        # Belief peak health monitoring
        peak_vals = pred_belief.view(pred_belief.shape[0], pred_belief.shape[1], -1).max(dim=-1).values
        loss_dict['rel/belief_peak_mean'] = peak_vals.mean().item()
        loss_dict['rel/belief_peak_min'] = peak_vals.min().item()

        total = l_rel
        loss_dict['rel/total'] = total.item()
        return total, loss_dict


# ── Structural Loss Module ────────────────────────────────────────────

# ── Vanishing Point Consistency Loss ─────────────────────────────────────
#
# Cuboid has 12 edges in 3 parallel groups. Under projective projection,
# each group of 4 parallel edges must meet at a single vanishing point.
# Unlike edge-length ratios, vanishing-point concurrency is a *projective
# invariant* — it holds even under strong perspective distortion.
#
# Keypoint convention (Y=UP):
#   0,1 = top-front-left/right ; 2,3 = bot-front-right/left
#   4,5 = top-back-left/right  ; 6,7 = bot-back-right/left
VP_EDGE_GROUPS = {
    'x': [(0, 1), (3, 2), (4, 5), (7, 6)],  # width-parallel
    'y': [(0, 3), (1, 2), (4, 7), (5, 6)],  # height-parallel (Y=UP)
    'z': [(0, 4), (1, 5), (2, 6), (3, 7)],  # depth-parallel
}


def _homogeneous_line(p1, p2, scale):
    """Homogeneous line through two 2D points, with coordinate normalization.

    To make the loss scale-invariant and numerically balanced, we normalize
    2D points by `scale` (object diagonal) before forming the line. This
    makes the last coordinate O(1) instead of O(pixels²).

    Args:
        p1, p2: (B, 2) points in pixels.
        scale: (B,) normalization factor (e.g. object diagonal).
    Returns:
        line: (B, 3) homogeneous line coefficients (a, b, c).
    """
    s = scale.unsqueeze(-1).clamp_min(1e-6)
    q1 = p1 / s
    q2 = p2 / s
    x1, y1 = q1[..., 0], q1[..., 1]
    x2, y2 = q2[..., 0], q2[..., 1]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return torch.stack([a, b, c], dim=-1)  # (B, 3)


def _vp_loss_one_group(kp, group_edges, eps=1e-6):
    """Vanishing-point concurrency loss for one parallel-edge group.

    Given 4 image-projected edges that should share a single vanishing
    point (possibly at infinity), we compute the best VP as the point
    p ∈ P² minimizing Σ_i (l_i · p)² subject to ||p|| = 1, where l_i are
    the 4 homogeneous lines. This is the smallest-eigenvector problem
    for M = Σ l_i l_i^T ∈ R^{3×3}, and the loss is the corresponding
    smallest eigenvalue λ_min(M).

    Crucially, this formulation handles vanishing points at infinity
    (parallel image lines) without any special-casing, because the unit
    sphere includes both finite and infinite points equally.

    The 2D coordinates are normalized by the object diagonal first so
    the line coefficients are dimensionless and the loss is scale-
    invariant.

    Args:
        kp: (B, N, 2) keypoints in pixels.
        group_edges: list of 4 (i, j) tuples.
    Returns:
        loss: (B,) scalar per batch.
    """
    diag = object_diagonal(kp).clamp_min(eps)  # (B,)

    # Build 4 homogeneous lines (coordinate-normalized).
    lines = []
    for (i, j) in group_edges:
        lines.append(_homogeneous_line(kp[:, i], kp[:, j], diag))
    L = torch.stack(lines, dim=1)  # (B, 4, 3)

    # M = Σ l_i l_i^T   (B, 3, 3)
    M = L.transpose(1, 2) @ L

    # Smallest eigenvalue of symmetric M via closed-form 3x3 or eigh.
    # torch.linalg.eigvalsh is stable and differentiable.
    eigvals = torch.linalg.eigvalsh(M)  # (B, 3), ascending order
    return eigvals[..., 0]  # smallest eigenvalue = squared residual


def vanishing_point_loss(kp_2d):
    """Vanishing-point concurrency loss for a cuboid cuboid.

    Args:
        kp_2d: (B, 9, 2) soft-argmax keypoints. Only indices 0–7 used.
    Returns:
        loss: scalar.
        info: dict.
    """
    total_per_group = []
    for name, edges in VP_EDGE_GROUPS.items():
        total_per_group.append(_vp_loss_one_group(kp_2d, edges))
    group_loss = torch.stack(total_per_group, dim=-1)  # (B, 3)
    loss = group_loss.mean()
    info = {
        'struct/vp': loss.item(),
        'struct/vp_x': group_loss[..., 0].mean().item(),
        'struct/vp_y': group_loss[..., 1].mean().item(),
        'struct/vp_z': group_loss[..., 2].mean().item(),
    }
    return loss, info


class StructuralLoss(nn.Module):
    """Combined structural losses: flip + edge + coord + vp.

    학습 시에만 사용. DOPE 모델 구조 변경 없음.
    """

    def __init__(self, soft_argmax, lambdas=None, delta=0.03):
        """
        Args:
            soft_argmax: SpatialSoftArgmax2D instance.
            lambdas: dict of loss weights.
            delta: Huber delta for all losses.
        """
        super().__init__()
        self.soft_argmax = soft_argmax
        self.flip_loss = FlipEquivarianceLoss(soft_argmax, delta=delta)
        self.edge_loss = SparseEdgeLoss(delta=delta)
        self.delta = delta

        default_lambdas = {
            'flip': 0.02,
            'edge': 0.05,
            'coord': 0.10,
            'vp': 0.0,
        }
        self.lambdas = {**default_lambdas, **(lambdas or {})}

    def forward(self, pred_belief, gt_belief, net=None, data_flip=None,
                epoch=0, warmup=10):
        """
        Args:
            pred_belief: (B, 9, H, W)
            gt_belief: (B, 9, H, W)
            net: DOPE network (for flip loss). None = skip flip.
            data_flip: (B, C, 448, 448) flipped input. None = skip flip.
            epoch: current epoch.
            warmup: epochs before enabling structural losses.
        Returns:
            total_loss: scalar.
            loss_dict: dict.
        """
        loss_dict = {}
        device = pred_belief.device
        total = torch.tensor(0.0, device=device)

        if epoch < warmup:
            loss_dict['struct/total'] = 0.0
            return total, loss_dict

        # ramp-up: linear from warmup to warmup+10
        ramp = min(1.0, (epoch - warmup) / 10.0)

        # soft-argmax coords
        pred_kp = self.soft_argmax(pred_belief)  # (B, 9, 2)
        gt_kp = self.soft_argmax(gt_belief)      # (B, 9, 2)

        # Coordinate Huber loss
        if self.lambdas['coord'] > 0:
            l_coord, info_coord = coord_huber_loss(pred_kp, gt_kp, self.delta)
            loss_dict.update(info_coord)
            total = total + self.lambdas['coord'] * ramp * l_coord

        # Sparse edge loss
        if self.lambdas['edge'] > 0:
            l_edge, info_edge = self.edge_loss(pred_kp, gt_kp)
            loss_dict.update(info_edge)
            total = total + self.lambdas['edge'] * ramp * l_edge

        # Flip equivariance loss
        if self.lambdas['flip'] > 0 and net is not None and data_flip is not None:
            l_flip, info_flip = self.flip_loss(pred_belief, net, data_flip)
            loss_dict.update(info_flip)
            total = total + self.lambdas['flip'] * ramp * l_flip

        # Vanishing-point concurrency loss (projective-invariant cuboid prior)
        if self.lambdas.get('vp', 0.0) > 0:
            l_vp, info_vp = vanishing_point_loss(pred_kp)
            loss_dict.update(info_vp)
            total = total + self.lambdas['vp'] * ramp * l_vp

        loss_dict['struct/total'] = total.item()
        loss_dict['struct/ramp'] = ramp
        return total, loss_dict


# ── Main Module ────────────────────────────────────────────────────────

class GeometricLoss(nn.Module):
    """Geometric loss combining soft-argmax + BPnP.

    학습 시에만 사용. Inference 시에는 완전히 제거 가능.
    DOPE 모델의 파라미터만 업데이트되며, 이 모듈은 학습 파라미터 없음.
    """

    def __init__(self, kp3d_np, K_np,
                 belief_size=50, input_size=448, orig_size=(640, 480),
                 temperature=1.0, damping=1e-4,
                 lambdas=None):
        """
        Args:
            kp3d_np: (9, 3) numpy — 3D model keypoints.
            K_np: (3, 3) numpy — camera intrinsics for original image.
            belief_size: belief map resolution (default 50).
            input_size: network input resolution (default 448).
            orig_size: (W, H) original image resolution.
            temperature: soft-argmax temperature.
            damping: BPnP LM damping.
            lambdas: dict of loss weights.
        """
        super().__init__()

        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.damping = damping

        # 3D keypoints
        self.register_buffer('kp3d', torch.from_numpy(kp3d_np.astype(np.float64)).float())

        # Camera matrix scaled to belief map space
        W_orig, H_orig = orig_size
        sx = input_size / W_orig * belief_size / input_size  # = belief_size / W_orig
        sy = input_size / H_orig * belief_size / input_size  # = belief_size / H_orig
        K_bel = np.array([
            [K_np[0, 0] * sx, 0, K_np[0, 2] * sx],
            [0, K_np[1, 1] * sy, K_np[1, 2] * sy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.register_buffer('K', torch.from_numpy(K_bel).float())

        # Loss weights
        default_lambdas = {
            'kp_l2': 1.0,
            'diagonal': 0.5,
            'reprojection': 0.1,
            'volume': 0.1,
            'add': 0.1,
        }
        self.lambdas = {**default_lambdas, **(lambdas or {})}

    def forward(self, pred_belief, gt_belief, epoch=0, warmup=5):
        """
        Args:
            pred_belief: (B, 9, H, W) predicted belief maps (final stage).
            gt_belief: (B, 9, H, W) ground truth belief maps.
            epoch: current epoch (for warmup).
            warmup: epochs before enabling PnP-based losses.

        Returns:
            total_loss: scalar geometric loss.
            loss_dict: dict of individual loss values (for logging).
        """
        loss_dict = {}
        device = pred_belief.device

        # ── Soft-argmax ──
        pred_kp = self.soft_argmax(pred_belief)  # (B, 9, 2)
        gt_kp = self.soft_argmax(gt_belief)      # (B, 9, 2)

        # ── Keypoint L2 ──
        l_kp = keypoint_l2_loss(pred_kp, gt_kp)
        loss_dict['geo/kp_l2'] = l_kp.item()

        # ── Diagonal consistency ──
        l_diag = diagonal_consistency_loss(pred_kp[:, :8])
        loss_dict['geo/diagonal'] = l_diag.item()

        total = self.lambdas['kp_l2'] * l_kp + self.lambdas['diagonal'] * l_diag

        # ── PnP-based losses (after warmup) ──
        if epoch >= warmup:
            kp3d_8 = self.kp3d[:8]
            pred_kp_8 = pred_kp[:, :8]
            gt_kp_8 = gt_kp[:, :8].detach()

            try:
                # BPnP on predicted and GT keypoints
                pose_pred, valid_pred = BPnP.apply(
                    pred_kp_8, kp3d_8, self.K, self.damping
                )
                pose_gt, valid_gt = BPnP.apply(
                    gt_kp_8, kp3d_8, self.K, self.damping
                )

                valid = valid_pred * valid_gt
                rvec_p, tvec_p = pose_pred[:, :3].float(), pose_pred[:, 3:].float()
                rvec_g, tvec_g = pose_gt[:, :3].float(), pose_gt[:, 3:].float()

                if valid.sum() > 0:
                    # Reprojection
                    l_reproj = reprojection_loss(
                        rvec_p, tvec_p, kp3d_8, self.K, gt_kp_8, valid
                    )
                    if not (torch.isnan(l_reproj) or torch.isinf(l_reproj)):
                        loss_dict['geo/reproj'] = l_reproj.item()
                        total = total + self.lambdas['reprojection'] * l_reproj

                    # Volume
                    l_vol = volume_loss(rvec_p, tvec_p, rvec_g, tvec_g, self.kp3d, valid)
                    if not (torch.isnan(l_vol) or torch.isinf(l_vol)):
                        loss_dict['geo/volume'] = l_vol.item()
                        total = total + self.lambdas['volume'] * l_vol

                    # ADD
                    l_add = add_loss(rvec_p, tvec_p, rvec_g, tvec_g, self.kp3d, valid)
                    if not (torch.isnan(l_add) or torch.isinf(l_add)):
                        loss_dict['geo/add'] = l_add.item()
                        total = total + self.lambdas['add'] * l_add

                loss_dict['geo/pnp_valid_rate'] = valid.mean().item()
            except Exception:
                pass  # PnP failed for this batch, skip geometric losses

        # NaN guard
        if torch.isnan(total) or torch.isinf(total):
            total = l_kp + l_diag  # fallback to soft-argmax only losses
        loss_dict['geo/total'] = total.item()
        return total, loss_dict
