"""Measure VP loss magnitude on real training data using v8_A checkpoint.

λ_vp calibration을 위한 실측. v8_A로 1~2 batch 추론하고 pred soft-argmax에서
VP loss, coord huber loss 실제 크기를 비교.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

from models import DopeNetwork
from utils import CleanVisiiDopeLoader
from geo_loss import (
    SpatialSoftArgmax2D,
    vanishing_point_loss,
    coord_huber_loss,
    SparseEdgeLoss,
)


def main():
    weights = os.path.join(ROOT, "weights/v9_ablation_A_coord/final_net_epoch_0065.pth")
    data_dir = os.path.join(ROOT, "data/pallet/training_data/mixed_v8_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {weights}")
    net = DopeNetwork()
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device).eval()

    print(f"Loading dataset: {data_dir}")
    ds = CleanVisiiDopeLoader(
        path_dataset=[data_dir], objects=["pallet"], sigma=4.0,
        output_size=50, extensions=["png"],
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    soft_argmax = SpatialSoftArgmax2D(temperature=1.0)
    edge_loss = SparseEdgeLoss(delta=0.03)

    n_batches = 8
    vp_raw_vals, vp_x_vals, vp_y_vals, vp_z_vals = [], [], [], []
    coord_raw_vals = []
    edge_raw_vals = []
    belief_l2_vals = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            data = batch["img"].to(device)
            gt_belief = batch["beliefs"][:, :9].to(device)

            out_bel, _ = net(data)
            pred_belief = out_bel[-1][:, :9]

            belief_l2 = ((pred_belief - gt_belief) ** 2).mean()
            belief_l2_vals.append(belief_l2.item())

            pred_kp = soft_argmax(pred_belief)
            gt_kp = soft_argmax(gt_belief)

            l_coord, _ = coord_huber_loss(pred_kp, gt_kp, delta=0.03)
            coord_raw_vals.append(l_coord.item())

            l_edge, _ = edge_loss(pred_kp, gt_kp)
            edge_raw_vals.append(l_edge.item())

            l_vp, info_vp = vanishing_point_loss(pred_kp)
            vp_raw_vals.append(l_vp.item())
            vp_x_vals.append(info_vp["struct/vp_x"])
            vp_y_vals.append(info_vp["struct/vp_y"])
            vp_z_vals.append(info_vp["struct/vp_z"])

            print(f"[{bi+1}/{n_batches}] "
                  f"belief={belief_l2.item():.5f}  "
                  f"coord_raw={l_coord.item():.5f}  "
                  f"edge_raw={l_edge.item():.5f}  "
                  f"vp_raw={l_vp.item():.6f}  "
                  f"(vp_x={info_vp['struct/vp_x']:.6f} "
                  f"vp_y={info_vp['struct/vp_y']:.6f} "
                  f"vp_z={info_vp['struct/vp_z']:.6f})")

    print()
    print("=" * 70)
    print(f"Mean over {n_batches} batches:")
    print(f"  belief L2   : {np.mean(belief_l2_vals):.5f}")
    print(f"  coord_raw   : {np.mean(coord_raw_vals):.5f}")
    print(f"  edge_raw    : {np.mean(edge_raw_vals):.5f}")
    print(f"  vp_raw      : {np.mean(vp_raw_vals):.6f}")
    print(f"    vp_x      : {np.mean(vp_x_vals):.6f}")
    print(f"    vp_y      : {np.mean(vp_y_vals):.6f}")
    print(f"    vp_z      : {np.mean(vp_z_vals):.6f}")
    print()
    print("=" * 70)
    print("λ calibration (coord fixed at 0.005):")
    coord_weighted = 0.005 * np.mean(coord_raw_vals)
    vp_raw_mean = np.mean(vp_raw_vals)
    print(f"  coord_weighted = 0.005 * {np.mean(coord_raw_vals):.5f} = {coord_weighted:.6f}")
    print(f"  vp_raw_mean    = {vp_raw_mean:.6f}")
    print()
    for pct in [0.10, 0.30, 0.50]:
        target_vp_weighted = pct * coord_weighted
        lam = target_vp_weighted / max(vp_raw_mean, 1e-8)
        print(f"  VP @ {int(pct*100):>3}% of coord: λ_vp ≈ {lam:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
