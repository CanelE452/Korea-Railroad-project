"""geo_loss.py — Structural loss (flip / edge / VP).

FLIP_PERM             : 좌우 flip 시 keypoint index permutation (0↔1, 2↔3, ...)
FlipEquivarianceLoss  : 원본 ↔ flip image 의 좌표 일관성 (shortcut 방지)
CUBOID_EDGES          : 12 physical edge + 4 face diagonal
EDGE_WEIGHTS          : face=1.0, depth=0.5, diagonal=0.3
SparseEdgeLoss        : edge log-ratio Huber (구조 일관성)
VP_EDGE_GROUPS        : 3 parallel-edge group (x, y, z 방향)
vanishing_point_loss  : 4 parallel edge 가 단일 vanishing point 에 모임 (projective invariant)
StructuralLoss        : 위 loss 들 + coord_huber 통합 wrapper
"""
import torch
import torch.nn as nn

from geo_loss_pose import object_diagonal, huber
from geo_loss_coord import coord_huber_loss


# ── Flip Equivariance ────────────────────────────────────────────────────

# Pallet horizontal flip: 0↔1, 2↔3, 4↔5, 6↔7, centroid 유지
FLIP_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 8]


class FlipEquivarianceLoss(nn.Module):
    """Flip consistency loss on soft-argmax coords.
    원본/flip image 에서 뽑은 좌표가 일관되도록 regularize. flip branch 는 stop-gradient."""

    def __init__(self, soft_argmax, delta=0.03):
        super().__init__()
        self.soft_argmax = soft_argmax
        self.delta = delta

    def forward(self, pred_belief, net, data_flip):
        B, C, H, W = pred_belief.shape
        kp_orig = self.soft_argmax(pred_belief)

        with torch.no_grad():
            flip_belief, _ = net(data_flip)
            flip_belief = flip_belief[-1][:, :9]

        kp_flip = self.soft_argmax(flip_belief)
        # flip 좌표를 원래 좌표계로: x → (W-1) - x + channel permutation
        kp_flip_unflip = kp_flip.clone()
        kp_flip_unflip[:, :, 0] = (W - 1) - kp_flip[:, :, 0]
        kp_flip_unflip = kp_flip_unflip[:, FLIP_PERM]

        diag = object_diagonal(kp_orig).unsqueeze(-1).unsqueeze(-1)
        diff = (kp_orig - kp_flip_unflip) / (diag + 1e-6)
        loss = huber(diff, self.delta).mean()
        return loss, {'struct/flip': loss.item()}


# ── Sparse Edge ──────────────────────────────────────────────────────────

CUBOID_EDGES = [
    # Front face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Rear face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Depth (front↔rear connecting)
    (0, 4), (1, 5), (2, 6), (3, 7),
]
CUBOID_DIAGONALS = [
    (0, 2), (1, 3),   # front
    (4, 6), (5, 7),   # rear
]
EDGE_WEIGHTS = (
    [1.0] * 4 + [1.0] * 4 +
    [0.5] * 4 +
    [0.3] * 4
)


class SparseEdgeLoss(nn.Module):
    """log-ratio edge length loss — 예측/GT edge 길이 비율이 1 에 가깝도록.
    Object diagonal normalize. 너무 짧은 GT edge 는 skip (perspective collapse 방지)."""

    def __init__(self, min_edge_ratio=0.03, delta=0.03):
        super().__init__()
        self.edges = CUBOID_EDGES + CUBOID_DIAGONALS
        self.weights = torch.tensor(EDGE_WEIGHTS, dtype=torch.float32)
        self.min_edge_ratio = min_edge_ratio
        self.delta = delta

    def forward(self, pred_kp, gt_kp):
        device = pred_kp.device
        eps = 1e-6
        diag = object_diagonal(gt_kp)
        weights = self.weights.to(device)
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)

        for idx, (i, j) in enumerate(self.edges):
            pred_len = torch.norm(pred_kp[:, i] - pred_kp[:, j], dim=-1)
            gt_len = torch.norm(gt_kp[:, i] - gt_kp[:, j], dim=-1)
            valid = (gt_len / (diag + eps)) > self.min_edge_ratio
            if valid.sum() < 1:
                continue
            log_ratio = torch.log((pred_len + eps) / (gt_len + eps))
            edge_loss = huber(log_ratio, self.delta)
            w = weights[idx]
            total_loss = total_loss + (edge_loss * valid.float() * w).sum()
            total_weight = total_weight + valid.float().sum() * w

        if total_weight < 1:
            return torch.tensor(0.0, device=device), {'struct/edge': 0.0}
        loss = total_loss / total_weight
        return loss, {'struct/edge': loss.item()}


# ── Vanishing Point ──────────────────────────────────────────────────────
# 12 cuboid edge 가 3 parallel group (x, y, z). 각 group 4 edge 는 단일 vanishing
# point 에 모여야 함 — edge-length ratio 와 달리 *projective invariant*.
VP_EDGE_GROUPS = {
    'x': [(0, 1), (3, 2), (4, 5), (7, 6)],
    'y': [(0, 3), (1, 2), (4, 7), (5, 6)],
    'z': [(0, 4), (1, 5), (2, 6), (3, 7)],
}


def _homogeneous_line(p1, p2, scale):
    """homogeneous line 계수 (a, b, c). 좌표는 scale(diagonal) 로 normalize 해서
    last coord 가 O(1) 가 되도록 (numerical balance + scale invariance)."""
    s = scale.unsqueeze(-1).clamp_min(1e-6)
    q1 = p1 / s
    q2 = p2 / s
    x1, y1 = q1[..., 0], q1[..., 1]
    x2, y2 = q2[..., 0], q2[..., 1]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return torch.stack([a, b, c], dim=-1)


def _vp_loss_one_group(kp, group_edges, eps=1e-6):
    """4 line 의 concurrency = M = Σ l_i l_i^T 의 smallest eigenvalue.
    vanishing point at infinity 도 자연스럽게 처리 (unit sphere)."""
    diag = object_diagonal(kp).clamp_min(eps)
    lines = []
    for (i, j) in group_edges:
        lines.append(_homogeneous_line(kp[:, i], kp[:, j], diag))
    L = torch.stack(lines, dim=1)
    M = L.transpose(1, 2) @ L
    eigvals = torch.linalg.eigvalsh(M)
    return eigvals[..., 0]


def vanishing_point_loss(kp_2d):
    """3 group VP concurrency loss. kp_2d: (B, 9, 2), 0~7 만 사용."""
    total_per_group = []
    for name, edges in VP_EDGE_GROUPS.items():
        total_per_group.append(_vp_loss_one_group(kp_2d, edges))
    group_loss = torch.stack(total_per_group, dim=-1)
    loss = group_loss.mean()
    info = {
        'struct/vp': loss.item(),
        'struct/vp_x': group_loss[..., 0].mean().item(),
        'struct/vp_y': group_loss[..., 1].mean().item(),
        'struct/vp_z': group_loss[..., 2].mean().item(),
    }
    return loss, info


# ── Combined Structural Loss ─────────────────────────────────────────────

class StructuralLoss(nn.Module):
    """flip + edge + coord + vp 통합. warmup 후 linear ramp-up."""

    def __init__(self, soft_argmax, lambdas=None, delta=0.03):
        super().__init__()
        self.soft_argmax = soft_argmax
        self.flip_loss = FlipEquivarianceLoss(soft_argmax, delta=delta)
        self.edge_loss = SparseEdgeLoss(delta=delta)
        self.delta = delta
        default = {'flip': 0.02, 'edge': 0.05, 'coord': 0.10, 'vp': 0.0}
        self.lambdas = {**default, **(lambdas or {})}

    def forward(self, pred_belief, gt_belief, net=None, data_flip=None,
                epoch=0, warmup=10):
        loss_dict = {}
        device = pred_belief.device
        total = torch.tensor(0.0, device=device)

        if epoch < warmup:
            loss_dict['struct/total'] = 0.0
            return total, loss_dict

        ramp = min(1.0, (epoch - warmup) / 10.0)
        pred_kp = self.soft_argmax(pred_belief)
        gt_kp = self.soft_argmax(gt_belief)

        if self.lambdas['coord'] > 0:
            l_coord, info_coord = coord_huber_loss(pred_kp, gt_kp, self.delta)
            loss_dict.update(info_coord)
            total = total + self.lambdas['coord'] * ramp * l_coord

        if self.lambdas['edge'] > 0:
            l_edge, info_edge = self.edge_loss(pred_kp, gt_kp)
            loss_dict.update(info_edge)
            total = total + self.lambdas['edge'] * ramp * l_edge

        if self.lambdas['flip'] > 0 and net is not None and data_flip is not None:
            l_flip, info_flip = self.flip_loss(pred_belief, net, data_flip)
            loss_dict.update(info_flip)
            total = total + self.lambdas['flip'] * ramp * l_flip

        if self.lambdas.get('vp', 0.0) > 0:
            l_vp, info_vp = vanishing_point_loss(pred_kp)
            loss_dict.update(info_vp)
            total = total + self.lambdas['vp'] * ramp * l_vp

        loss_dict['struct/total'] = total.item()
        loss_dict['struct/ramp'] = ramp
        return total, loss_dict
