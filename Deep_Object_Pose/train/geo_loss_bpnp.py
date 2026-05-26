"""geo_loss.py — BPnP (Backpropagatable PnP) + Soft-Argmax + projection 헬퍼.

rodrigues_batch        : batch rvec → R matrix (Rodrigues, autograd-friendly)
project_points         : (R, t, kp3d, K) → 2D pixel
projection_jacobian    : projection Jacobian w.r.t. (rvec, tvec) (BPnP backward 용)
SpatialSoftArgmax2D    : differentiable soft-argmax + sigma (uncertainty)
BPnP                   : forward = cv2.solvePnP, backward = implicit diff
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rodrigues_batch(rvec):
    """rvec (B, 3) → R (B, 3, 3). autograd-friendly Rodrigues."""
    B = rvec.shape[0]
    theta = torch.norm(rvec, dim=-1, keepdim=True).unsqueeze(-1)
    eps = 1e-8
    k = rvec / (theta.squeeze(-1) + eps)

    K = torch.zeros(B, 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]

    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def project_points(R, t, kp3d, K):
    """3D → 2D pixel projection. R (B,3,3), t (B,3), kp3d (N,3), K (3,3) → (B,N,2)."""
    B = R.shape[0]
    P = kp3d.unsqueeze(0).expand(B, -1, -1)
    P_cam = torch.bmm(P, R.transpose(1, 2)) + t.unsqueeze(1)
    P_proj = torch.bmm(P_cam, K.T.unsqueeze(0).expand(B, -1, -1))
    z = P_proj[:, :, 2:3].clamp(min=1e-6)
    return P_proj[:, :, :2] / z


def projection_jacobian(rvec, tvec, kp3d, K):
    """Projection Jacobian w.r.t. (rvec, tvec). (B, 2N, 6) — BPnP backward 용."""
    B = rvec.shape[0]
    N = kp3d.shape[0]
    device = rvec.device
    dtype = rvec.dtype

    R = rodrigues_batch(rvec)
    P = kp3d.unsqueeze(0).expand(B, -1, -1)
    P_cam = torch.bmm(P, R.transpose(1, 2)) + tvec.unsqueeze(1)

    fx, fy = K[0, 0], K[1, 1]
    X = P_cam[:, :, 0]; Y = P_cam[:, :, 1]
    Z = P_cam[:, :, 2].clamp(min=1e-6); Z2 = Z * Z

    dpdP = torch.zeros(B, N, 2, 3, device=device, dtype=dtype)
    dpdP[:, :, 0, 0] = fx / Z;   dpdP[:, :, 0, 2] = -fx * X / Z2
    dpdP[:, :, 1, 1] = fy / Z;   dpdP[:, :, 1, 2] = -fy * Y / Z2

    RP = P_cam - tvec.unsqueeze(1)
    dPdr = torch.zeros(B, N, 3, 3, device=device, dtype=dtype)
    dPdr[:, :, 0, 1] = RP[:, :, 2];  dPdr[:, :, 0, 2] = -RP[:, :, 1]
    dPdr[:, :, 1, 0] = -RP[:, :, 2]; dPdr[:, :, 1, 2] = RP[:, :, 0]
    dPdr[:, :, 2, 0] = RP[:, :, 1];  dPdr[:, :, 2, 1] = -RP[:, :, 0]

    dPdt = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(B, N, -1, -1)

    dpdr = torch.matmul(dpdP, dPdr)
    dpdt = torch.matmul(dpdP, dPdt)
    J_per_point = torch.cat([dpdr, dpdt], dim=-1)
    return J_per_point.reshape(B, 2 * N, 6)


class SpatialSoftArgmax2D(nn.Module):
    """Differentiable soft-argmax for 2D heatmaps + per-keypoint sigma (uncertainty)."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmap, return_sigma=False):
        """heatmap (B, C, H, W) → coords (B, C, 2). return_sigma 면 sigma (B, C) 도."""
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

        mu_x = x.unsqueeze(-1).unsqueeze(-1)
        mu_y = y.unsqueeze(-1).unsqueeze(-1)
        var = (weights * ((x_coords - mu_x) ** 2 + (y_coords - mu_y) ** 2)).sum(dim=(2, 3))
        sigma = torch.sqrt(var.clamp(min=1e-6))
        return coords, sigma


class BPnP(torch.autograd.Function):
    """Backpropagatable PnP. forward = cv2.solvePnP, backward = implicit diff.

    Reference: Chen et al., "End-to-End Learnable Geometric Vision by
    Backpropagating PnP Optimization", CVPR 2020.
    """

    @staticmethod
    def forward(ctx, kp2d, kp3d, K, damping=1e-4):
        """kp2d (B, N, 2), kp3d (N, 3), K (3, 3) → pose (B, 6), valid (B,)."""
        B, N, _ = kp2d.shape
        device = kp2d.device

        kp2d_np = kp2d.detach().cpu().numpy()
        kp3d_np = kp3d.detach().cpu().numpy().astype(np.float64)
        K_np = K.detach().cpu().numpy().astype(np.float64)

        poses = np.zeros((B, 6), dtype=np.float32)
        valids = np.zeros(B, dtype=np.float32)

        for b in range(B):
            pts2d = kp2d_np[b].astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                kp3d_np, pts2d, K_np, None, flags=cv2.SOLVEPNP_EPNP)
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
        pose = torch.clamp(pose, -100.0, 100.0)   # NaN/Inf 방지

        ctx.save_for_backward(kp2d, kp3d, K, pose, valid)
        ctx.damping = damping
        return pose, valid

    @staticmethod
    def backward(ctx, grad_pose, grad_valid):
        kp2d, kp3d, K, pose, valid = ctx.saved_tensors
        damping = ctx.damping
        B, N, _ = kp2d.shape
        device = kp2d.device

        rvec = pose[:, :3].detach().float().to(device)
        tvec = pose[:, 3:].detach().float().to(device)
        kp3d_f = kp3d.detach().float().to(device)
        K_f = K.detach().float().to(device)
        grad_pose_f = grad_pose.detach().float().to(device)
        valid_f = valid.detach().float().to(device)

        J = projection_jacobian(rvec, tvec, kp3d_f, K_f).float()
        JtJ = torch.bmm(J.transpose(1, 2), J)
        eye = torch.eye(6, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        JtJ_reg = JtJ + max(damping, 1e-3) * eye

        JtJ_inv = torch.linalg.pinv(JtJ_reg).float()
        x = torch.bmm(JtJ_inv, grad_pose_f.unsqueeze(-1))
        grad_kp2d_flat = torch.bmm(J, x).squeeze(-1).float()

        grad_kp2d_flat = grad_kp2d_flat * valid_f.unsqueeze(-1)
        grad_kp2d_flat = torch.nan_to_num(grad_kp2d_flat, nan=0.0, posinf=0.0, neginf=0.0)
        grad_kp2d_flat = torch.clamp(grad_kp2d_flat, -10.0, 10.0)
        grad_kp2d = grad_kp2d_flat.view(B, N, 2)

        return grad_kp2d, None, None, None
