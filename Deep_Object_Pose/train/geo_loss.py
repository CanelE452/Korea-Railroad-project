"""Geometric loss for DOPE training — main entry + re-export shim.

Soft-argmax + BPnP (Backpropagatable PnP) 로 belief MSE 위에 3D 기하학적 제약 추가.
학습 시에만 사용, inference 시 완전 제거 가능.

v2: Structural losses 추가 (2026-04-07)
  - FlipEquivarianceLoss, SparseEdgeLoss, VP, coord_huber + Huber + diagonal normalize

Reference:
    BPnP: Chen et al., "End-to-End Learnable Geometric Vision by
    Backpropagating PnP Optimization", CVPR 2020

분리된 모듈:
  geo_loss_bpnp.py    rodrigues_batch / project_points / projection_jacobian /
                       SpatialSoftArgmax2D / BPnP
  geo_loss_pose.py    keypoint_l2 / diagonal_consistency / reprojection /
                       volume / add / huber / object_diagonal
  geo_loss_coord.py   coord_vis_loss / coord_huber_loss / coord_rel_loss /
                       VisibilityCoordLoss / ReliabilityLoss
  geo_loss_struct.py  Flip / SparseEdge / VanishingPoint / StructuralLoss
"""
import numpy as np
import torch
import torch.nn as nn

# Re-export — 기존 import 호환
from geo_loss_bpnp import (
    rodrigues_batch, project_points, projection_jacobian,
    SpatialSoftArgmax2D, BPnP,
)
from geo_loss_pose import (
    keypoint_l2_loss, CUBOID_FACES, diagonal_consistency_loss,
    reprojection_loss, volume_loss, add_loss, huber, object_diagonal,
)
from geo_loss_coord import (
    coord_vis_loss, coord_huber_loss, coord_rel_loss,
    VisibilityCoordLoss, ReliabilityLoss,
)
from geo_loss_struct import (
    FLIP_PERM, FlipEquivarianceLoss,
    CUBOID_EDGES, CUBOID_DIAGONALS, EDGE_WEIGHTS, SparseEdgeLoss,
    VP_EDGE_GROUPS, vanishing_point_loss, StructuralLoss,
)


class GeometricLoss(nn.Module):
    """Soft-argmax + BPnP 기반 통합 geometric loss.

    학습 파라미터 없음 — DOPE 파라미터만 업데이트. inference 시 완전히 제거 가능.

    Loss 구성:
      kp_l2     : soft-argmax coord MSE
      diagonal  : cuboid face diagonal midpoint consistency
      reproj    : BPnP pose → reprojection MSE
      volume    : 예측/GT cuboid 부피 ratio
      add       : Average Distance of model points
    """

    def __init__(self, kp3d_np, K_np,
                 belief_size=50, input_size=448, orig_size=(640, 480),
                 temperature=1.0, damping=1e-4, lambdas=None):
        super().__init__()
        self.soft_argmax = SpatialSoftArgmax2D(temperature)
        self.damping = damping

        self.register_buffer(
            'kp3d', torch.from_numpy(kp3d_np.astype(np.float64)).float())

        # K scaled to belief map space
        W_orig, H_orig = orig_size
        sx = belief_size / W_orig
        sy = belief_size / H_orig
        K_bel = np.array([
            [K_np[0, 0] * sx, 0, K_np[0, 2] * sx],
            [0, K_np[1, 1] * sy, K_np[1, 2] * sy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.register_buffer('K', torch.from_numpy(K_bel).float())

        default = {'kp_l2': 1.0, 'diagonal': 0.5,
                   'reprojection': 0.1, 'volume': 0.1, 'add': 0.1}
        self.lambdas = {**default, **(lambdas or {})}

    def forward(self, pred_belief, gt_belief, epoch=0, warmup=5):
        loss_dict = {}
        device = pred_belief.device

        pred_kp = self.soft_argmax(pred_belief)
        gt_kp = self.soft_argmax(gt_belief)

        l_kp = keypoint_l2_loss(pred_kp, gt_kp)
        loss_dict['geo/kp_l2'] = l_kp.item()

        l_diag = diagonal_consistency_loss(pred_kp[:, :8])
        loss_dict['geo/diagonal'] = l_diag.item()

        total = self.lambdas['kp_l2'] * l_kp + self.lambdas['diagonal'] * l_diag

        # PnP-based losses (after warmup)
        if epoch >= warmup:
            kp3d_8 = self.kp3d[:8]
            pred_kp_8 = pred_kp[:, :8]
            gt_kp_8 = gt_kp[:, :8].detach()

            try:
                pose_pred, valid_pred = BPnP.apply(
                    pred_kp_8, kp3d_8, self.K, self.damping)
                pose_gt, valid_gt = BPnP.apply(
                    gt_kp_8, kp3d_8, self.K, self.damping)

                valid = valid_pred * valid_gt
                rvec_p = pose_pred[:, :3].float()
                tvec_p = pose_pred[:, 3:].float()
                rvec_g = pose_gt[:, :3].float()
                tvec_g = pose_gt[:, 3:].float()

                if valid.sum() > 0:
                    l_reproj = reprojection_loss(
                        rvec_p, tvec_p, kp3d_8, self.K, gt_kp_8, valid)
                    if not (torch.isnan(l_reproj) or torch.isinf(l_reproj)):
                        loss_dict['geo/reproj'] = l_reproj.item()
                        total = total + self.lambdas['reprojection'] * l_reproj

                    l_vol = volume_loss(rvec_p, tvec_p, rvec_g, tvec_g,
                                        self.kp3d, valid)
                    if not (torch.isnan(l_vol) or torch.isinf(l_vol)):
                        loss_dict['geo/volume'] = l_vol.item()
                        total = total + self.lambdas['volume'] * l_vol

                    l_add = add_loss(rvec_p, tvec_p, rvec_g, tvec_g,
                                     self.kp3d, valid)
                    if not (torch.isnan(l_add) or torch.isinf(l_add)):
                        loss_dict['geo/add'] = l_add.item()
                        total = total + self.lambdas['add'] * l_add

                loss_dict['geo/pnp_valid_rate'] = valid.mean().item()
            except Exception:
                pass

        if torch.isnan(total) or torch.isinf(total):
            total = l_kp + l_diag
        loss_dict['geo/total'] = total.item()
        return total, loss_dict
