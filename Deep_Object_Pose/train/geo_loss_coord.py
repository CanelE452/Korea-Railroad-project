"""geo_loss.py — Coordinate level loss (Huber / visibility-weighted / reliability).

coord_vis_loss        : GT visibility 로 가중된 normalized coord Huber
coord_huber_loss      : normalized coord Huber (no visibility)
coord_rel_loss        : sigma 기반 reliability-weighted coord Huber (v2)
VisibilityCoordLoss   : nn.Module wrapper for coord_vis_loss
ReliabilityLoss       : nn.Module wrapper for coord_rel_loss (sigma 자동 추출)
"""
import torch
import torch.nn as nn

from geo_loss_bpnp import SpatialSoftArgmax2D
from geo_loss_pose import object_diagonal, huber


def coord_vis_loss(pred_kp, gt_kp, vis_weight, delta=0.03, eps=1e-6):
    """Visibility-weighted normalized coord Huber.

    visible(1.0) > self-occluded(0.5) > out-of-frame(0.0) per-keypoint.
    Loss scale 은 visible 갯수에 invariant.
    """
    diag = object_diagonal(gt_kp).unsqueeze(-1)
    diff = torch.norm(pred_kp - gt_kp, dim=-1) / (diag + eps)
    robust_err = huber(diff, delta)
    weighted_err = vis_weight * robust_err
    loss = weighted_err.sum(dim=-1) / (vis_weight.sum(dim=-1) + eps)
    loss = loss.mean()

    info = {
        'vis/coord_vis': loss.item(),
        'vis/visible_pts': (vis_weight > 0.9).float().sum().item() / vis_weight.shape[0],
        'vis/occluded_pts': ((vis_weight > 0.1) & (vis_weight < 0.9)).float().sum().item() / vis_weight.shape[0],
        'vis/invisible_pts': (vis_weight < 0.1).float().sum().item() / vis_weight.shape[0],
    }
    return loss, info


def coord_huber_loss(pred_kp, gt_kp, delta=0.03):
    """normalized coord Huber, visibility 없이."""
    diag = object_diagonal(gt_kp).unsqueeze(-1).unsqueeze(-1)
    diff = (pred_kp - gt_kp) / (diag + 1e-6)
    loss = huber(diff, delta).mean()
    return loss, {'struct/coord': loss.item()}


def coord_rel_loss(pred_kp, gt_kp, sigma, delta=0.03, lambda_log=0.5, eps=1e-4,
                   w_min=0.4, w_max=1.8):
    """Reliability-weighted coord Huber (v2: detach + clip + normalize).

    sigma 를 reweighting signal 로 사용 (학습 가능 X — detach):
      - hard point 도 최소 w_min 가중치
      - 너무 confident 한 점은 w_max 로 cap
      - per-image 평균 1 로 normalize
    """
    B, K = sigma.shape
    diag = object_diagonal(gt_kp).unsqueeze(-1)

    diff = torch.norm(pred_kp - gt_kp, dim=-1) / (diag + eps)
    robust_err = huber(diff, delta)

    u = sigma.detach()
    w = (1.0 / (u + eps)).clamp(min=w_min, max=w_max)
    w_norm = K * w / (w.sum(dim=-1, keepdim=True) + eps)

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


class VisibilityCoordLoss(nn.Module):
    """Visibility-aware coordinate loss — GT visibility 로 가중된 coord Huber."""

    def __init__(self, temperature=1.0, delta=0.03):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.delta = delta

    def forward(self, pred_belief, gt_belief, vis_weight, epoch=0, warmup=0):
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
    """Reliability-controlled coord loss — heatmap 에서 sigma 추출 후 heteroscedastic supervision."""

    def __init__(self, temperature=1.0, delta=0.03, lambda_log=0.5,
                 w_min=0.4, w_max=1.8):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.delta = delta
        self.lambda_log = lambda_log
        self.w_min = w_min
        self.w_max = w_max

    def forward(self, pred_belief, gt_belief, epoch=0, warmup=0):
        loss_dict = {}
        device = pred_belief.device
        total = torch.tensor(0.0, device=device)

        if epoch < warmup:
            loss_dict['rel/total'] = 0.0
            return total, loss_dict

        pred_kp, sigma = self.soft_argmax(pred_belief, return_sigma=True)
        gt_kp = self.soft_argmax(gt_belief)

        l_rel, info = coord_rel_loss(
            pred_kp, gt_kp, sigma,
            delta=self.delta, lambda_log=self.lambda_log,
            w_min=self.w_min, w_max=self.w_max,
        )
        loss_dict.update(info)

        peak_vals = pred_belief.view(pred_belief.shape[0], pred_belief.shape[1], -1).max(dim=-1).values
        loss_dict['rel/belief_peak_mean'] = peak_vals.mean().item()
        loss_dict['rel/belief_peak_min'] = peak_vals.min().item()

        total = l_rel
        loss_dict['rel/total'] = total.item()
        return total, loss_dict
