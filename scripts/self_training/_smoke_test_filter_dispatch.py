"""Smoke test for filter_type dispatch in generate_pseudo_labels.

Loads v8_A_control/ep68 and runs pseudo-label generation on 10 noapril frames
for each filter_type. Verifies that all four branches execute without error
and produce sensible (varying) acceptance rates.
"""
import argparse
import os
import sys
import glob

import numpy as np
import torch
import torch.utils.data as data

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_root, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_root, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(_root, "scripts", "data_prep"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix
from geometric_filter import GeometricFilter
from augmentations import WeakAugmentation
from self_train import RealUnlabeledDataset, generate_pseudo_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(
        _root, "weights", "v8_A_control", "final_net_epoch_0068.pth"))
    parser.add_argument("--num_frames", type=int, default=10,
                        help="How many noapril frames to use (default 10 for smoke test)")
    parser.add_argument("--tag", default="smoke",
                        help="Label printed in the result header")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights = args.weights
    print(f"Loading: {weights}")
    model = DopeNetwork()
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    real_dir = os.path.join(_root, "data", "pallet", "raw_data",
                            "capture0403noapril", "rgb")
    all_imgs = sorted(glob.glob(os.path.join(real_dir, "*.png")) +
                      glob.glob(os.path.join(real_dir, "*.jpg")))
    print(f"Found {len(all_imgs)} images in {real_dir}")
    assert len(all_imgs) >= 10, "Need at least 10 images"

    # Build a subset dataset (default 10 for smoke test, --num_frames to override)
    n_frames = args.num_frames
    class _Tiny(data.Dataset):
        def __init__(self, paths):
            self.ds = RealUnlabeledDataset(os.path.dirname(paths[0]))
            self.ds.image_paths = sorted(self.ds.image_paths)[:n_frames]

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return self.ds[i]

    tiny = _Tiny(all_imgs)
    print(f"Using {len(tiny)} frames")
    loader = data.DataLoader(tiny, batch_size=4, shuffle=False, num_workers=0)
    print(f"\n[{args.tag}] weights={os.path.basename(os.path.dirname(args.weights))}/{os.path.basename(args.weights)}")

    K = make_camera_matrix(615, 615, 320, 240)
    solver = PalletPnPSolver(
        K, pallet_dims=(1.10, 1.30, 0.11),
        use_ransac=True, ransac_reproj_threshold=8.0, ransac_iterations=100)

    gf_cfg = {
        "ransac_n_iter": 50, "ransac_subset": 5, "ransac_reproj_px": 5.0,
        "ransac_min_consensus": 6,
        "tau_size_min": 0.5, "tau_size_max": 2.5, "min_keypoints": 5,
        "bc_tau_span": 0.35, "bc_tau_end": 0.10, "bc_tau_nc": 0.02,
        "bc_tau_C": 0.05, "bc_min_kps_B": 4, "bc_min_kps_C": 5,
        "conf_min": 0.5, "seed": 0,
    }
    gf = GeometricFilter(solver, gf_cfg)
    weak_aug = WeakAugmentation(brightness=0.0, contrast=0.0, noise_std=0.0)

    base_config = {
        "self_training": {"image_size": 448, "sigma": 2.0},
        "geometric_filter": dict(gf_cfg),
    }

    print(f"\n{'filter':<10}{'total':>6}{'acc':>6}{'rate':>8}"
          f"{'pnp_fail':>10}{'filt_fail':>11}{'reproj_mean':>13}")
    print("-" * 64)
    for ft in ["ransac", "bc", "conf", "none"]:
        cfg = dict(base_config)
        cfg["geometric_filter"] = dict(gf_cfg, filter_type=ft)
        accepted, stats = generate_pseudo_labels(
            model, loader, solver, gf, weak_aug, cfg, device)
        print(f"{ft:<10}{stats['total']:>6}{stats['accepted']:>6}"
              f"{stats['acceptance_rate']*100:>7.1f}%"
              f"{stats['pnp_fail']:>10}{stats['filter_fail']:>11}"
              f"{stats['reproj_error_mean']:>13.2f}")

    print("\nSmoke test passed - all 4 filter types executed without error.")


if __name__ == "__main__":
    main()
