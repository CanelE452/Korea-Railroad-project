"""Stage 3: FixMatch-based Self-Training with Geometric Filter.

Implements the self-training loop for unsupervised domain adaptation
of DOPE (Deep Object Pose Estimation) from synthetic to real data.

Pipeline per round:
  1. Generate pseudo-labels on unlabeled real images:
     weak aug -> DOPE inference -> peak extraction -> PnP -> geometric filter
  2. Train on synthetic (GT) + pseudo-labeled real (strong aug):
     loss = MSE(belief) + MSE(affinity) for synthetic
          + lambda_real * MSE(belief) for pseudo-labeled real

Usage:
    python scripts/self_training/self_train.py \\
        --config config/stage3_selftrain.yaml \\
        --pretrained weights/pallet_category/net_pallet_best.pth \\
        --synthetic_dir data/pallet/training_data/train \\
        --real_dir data/pallet/real_unlabeled \\
        --output_dir output/stage3_selftrain

See config/stage3_selftrain.yaml for all hyperparameters.
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml

# Add DOPE modules to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_project_root, "Deep_Object_Pose", "train"))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_project_root, "scripts", "data_prep"))

from models import DopeNetwork
from utils import CreateBeliefMap, GenerateMapAffinity
from pnp_solver import PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d
from geometric_filter import GeometricFilter
from augmentations import WeakAugmentation, StrongAugmentation
from canonical_filters import filter_B as canonical_filter_B
from canonical_filters import filter_C as canonical_filter_C
from pnp_solver import make_pallet_keypoints_3d


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class SyntheticDataset(data.Dataset):
    """Loads synthetic data with NDDS-format JSON annotations."""

    def __init__(self, data_dir, object_name="pallet", image_size=448,
                 output_size=56, sigma=2.0):
        self.data_dir = data_dir
        self.object_name = object_name.lower()
        self.image_size = image_size
        self.output_size = output_size
        self.sigma = sigma

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Find all image-json pairs
        self.samples = []
        if isinstance(data_dir, list):
            dirs = data_dir
        else:
            dirs = [data_dir]

        for d in dirs:
            for ext in ["*.png", "*.jpg"]:
                for img_path in sorted(glob.glob(os.path.join(d, ext))):
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path))

        print(f"SyntheticDataset: {len(self.samples)} samples from {dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, json_path = self.samples[index]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Load annotation
        with open(json_path) as f:
            ann = json.load(f)

        # Extract keypoints
        keypoints = self._extract_keypoints(ann, h_orig, w_orig)

        # Generate belief maps and affinity maps
        beliefs = self._generate_beliefs(keypoints)
        affinities = self._generate_affinities(keypoints)

        # Normalize image
        img_tensor = self.normalize(img)

        # Clean NaN/Inf
        beliefs[torch.isnan(beliefs) | torch.isinf(beliefs)] = 0
        affinities[torch.isnan(affinities) | torch.isinf(affinities)] = 0

        return {
            "img": img_tensor,
            "beliefs": torch.clamp(beliefs, 0, 1),
            "affinities": torch.clamp(affinities, -1, 1),
            "type": "synthetic",
        }

    def _extract_keypoints(self, ann, h_orig, w_orig):
        """Extract 9 keypoints from NDDS annotation, scaled to image_size."""
        keypoints = [[-100, -100]] * 9
        for obj in ann.get("objects", []):
            if obj.get("class", "").lower() != self.object_name:
                continue
            if obj.get("visibility", 0) <= 0:
                continue
            cuboid = obj.get("projected_cuboid", [])
            centroid = obj.get("projected_cuboid_centroid", [-100, -100])
            if len(cuboid) == 8:
                for i in range(8):
                    keypoints[i] = [
                        cuboid[i][0] * self.image_size / w_orig,
                        cuboid[i][1] * self.image_size / h_orig,
                    ]
                keypoints[8] = [
                    centroid[0] * self.image_size / w_orig,
                    centroid[1] * self.image_size / h_orig,
                ]
            break
        return [keypoints]

    def _generate_beliefs(self, all_keypoints):
        scale = self.output_size / self.image_size
        scaled_kps = [[[x * scale, y * scale] for x, y in kp] for kp in all_keypoints]
        beliefs = CreateBeliefMap(
            size=self.output_size,
            pointsBelief=scaled_kps,
            sigma=self.sigma,
            nbpoints=9,
            save=False,
        )
        return torch.from_numpy(np.array(beliefs)).float()

    def _generate_affinities(self, all_keypoints):
        scale = self.output_size / self.image_size
        scaled_kps = [[[x * scale, y * scale] for x, y in kp] for kp in all_keypoints]
        centroids = [kp[8] for kp in scaled_kps]
        affinities = GenerateMapAffinity(
            size=self.output_size,
            nb_vertex=8,
            pointsInterest=scaled_kps,
            objects_centroid=centroids,
            scale=1,
        )
        return affinities.float()


class RealUnlabeledDataset(data.Dataset):
    """Loads unlabeled real images for pseudo-label generation."""

    def __init__(self, data_dir, image_size=448):
        self.image_size = image_size

        self.image_paths = []
        if isinstance(data_dir, list):
            dirs = data_dir
        else:
            dirs = [data_dir]

        for d in dirs:
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_paths.extend(sorted(glob.glob(os.path.join(d, ext))))

        print(f"RealUnlabeledDataset: {len(self.image_paths)} images from {dirs}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Return as float tensor [0, 1] without normalization
        # (augmentation is applied before normalization)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "img": img_tensor,
            "path": img_path,
        }


class PseudoLabeledDataset(data.Dataset):
    """Dataset of pseudo-labeled real images accepted by geometric filter."""

    def __init__(self, entries, image_size=448, strong_aug=None):
        """
        Args:
            entries: list of dicts with keys:
                - img: (C, H, W) float tensor [0, 1]
                - beliefs: (9, H', W') tensor
            image_size: target image size.
            strong_aug: StrongAugmentation instance.
        """
        self.entries = entries
        self.image_size = image_size
        self.strong_aug = strong_aug

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        img = entry["img"].clone()  # (C, H, W) in [0, 1]

        # Apply strong augmentation
        if self.strong_aug is not None:
            img = self.strong_aug(img)

        # Normalize for DOPE
        img = self.normalize(img)

        return {
            "img": img,
            "beliefs": entry["beliefs"],
            "type": "pseudo_real",
        }


# ---------------------------------------------------------------------------
# Peak extraction from belief maps
# ---------------------------------------------------------------------------

def extract_peaks(belief_maps, threshold=0.3, image_size=448):
    """Extract 2D keypoint locations + peak confidences from belief maps.

    Returns:
        list of 9 elements, each (u, v, conf) in image coords or None.
        conf is the raw belief peak value (0 ~ 1 range typically).
    """
    if isinstance(belief_maps, torch.Tensor):
        belief_maps = belief_maps.cpu().numpy()

    num_kp, bh, bw = belief_maps.shape
    scale_x = image_size / bw
    scale_y = image_size / bh

    keypoints = []
    for i in range(num_kp):
        bmap = belief_maps[i]
        max_val = float(bmap.max())
        if max_val < threshold:
            keypoints.append(None)
            continue

        idx = bmap.argmax()
        py, px = divmod(int(idx), bw)

        win = 5
        half = win // 2
        y_start = max(0, py - half)
        y_end = min(bh, py + half + 1)
        x_start = max(0, px - half)
        x_end = min(bw, px + half + 1)

        patch = bmap[y_start:y_end, x_start:x_end]
        if patch.sum() < 1e-8:
            keypoints.append(None)
            continue

        ys, xs = np.meshgrid(
            np.arange(y_start, y_end),
            np.arange(x_start, x_end),
            indexing='ij',
        )
        cx = float(np.average(xs, weights=patch))
        cy = float(np.average(ys, weights=patch))

        u = cx * scale_x
        v = cy * scale_y
        keypoints.append((u, v, max_val))

    return keypoints


# ---------------------------------------------------------------------------
# Belief map generation from keypoints
# ---------------------------------------------------------------------------

def generate_belief_maps_from_keypoints(keypoints_2d, image_size=448,
                                        belief_map_size=56, sigma=2.0):
    """Generate belief maps from 2D keypoints for pseudo-labels.

    Args:
        keypoints_2d: list of 9 elements, each (u, v) in image_size coords or None.
        image_size: DOPE input image size.
        belief_map_size: belief map output resolution (DOPE output size).
        sigma: Gaussian sigma.

    Returns:
        torch.Tensor of shape (9, belief_map_size, belief_map_size).
    """
    scale = belief_map_size / image_size
    kps = []
    for pt in keypoints_2d:
        if pt is None:
            kps.append([-100, -100])
        else:
            kps.append([pt[0] * scale, pt[1] * scale])

    beliefs = CreateBeliefMap(
        size=belief_map_size,
        pointsBelief=[kps],
        sigma=sigma,
        nbpoints=9,
        save=False,
    )
    return torch.from_numpy(np.array(beliefs)).float()


# ---------------------------------------------------------------------------
# Main self-training loop
# ---------------------------------------------------------------------------

def _apply_filter(filter_type, keypoints_2d, pnp_solver, geo_filter,
                  gf_cfg, image_size):
    """Dispatch filter based on filter_type.

    Returns:
        is_valid: bool
        R, t: pose (for logging; None if filter does not solve)
        reproj_error_mean: float or None (for logging)
        reason: "pnp_fail" | "filter_fail" | None (None = accepted)
    """
    if filter_type == "none":
        return True, None, None, None

    if filter_type == "conf":
        conf_min = float(gf_cfg.get("conf_min", 0.5))
        confs = [float(p[2]) for p in keypoints_2d[:8]
                 if p is not None and len(p) >= 3]
        if not confs or min(confs) < conf_min:
            return False, None, None, "filter_fail"
        return True, None, None, None

    if filter_type == "ransac":
        is_valid, R, t, details = geo_filter.solve_and_validate(keypoints_2d)
        if R is None:
            return False, None, None, "pnp_fail"
        if not is_valid:
            return False, R, t, "filter_fail"
        return True, R, t, None

    if filter_type == "bc":
        success, R, t, _ = pnp_solver.solve(keypoints_2d)
        if not success or R is None:
            return False, None, None, "pnp_fail"
        b_pass, _ = canonical_filter_B(
            keypoints_2d, pnp_solver, R, t,
            tau_span=float(gf_cfg.get("bc_tau_span", 0.35)),
            tau_end=float(gf_cfg.get("bc_tau_end", 0.10)),
            tau_nc=float(gf_cfg.get("bc_tau_nc", 0.02)),
            min_kps=int(gf_cfg.get("bc_min_kps_B", 4)),
            img_size=(image_size, image_size),
        )
        if not b_pass:
            return False, R, t, "filter_fail"
        c_pass, _ = canonical_filter_C(
            keypoints_2d, pnp_solver, R, t,
            tau_C=float(gf_cfg.get("bc_tau_C", 0.05)),
            min_kps=int(gf_cfg.get("bc_min_kps_C", 5)),
        )
        if not c_pass:
            return False, R, t, "filter_fail"
        size_pass, _ = geo_filter._check_size(R, t)
        if not size_pass:
            return False, R, t, "filter_fail"
        return True, R, t, None

    if filter_type == "ransac_loo":
        is_valid, R, t, details = geo_filter.solve_and_validate(keypoints_2d)
        if R is None:
            return False, None, None, "pnp_fail"
        if not is_valid:
            return False, R, t, "filter_fail"
        loo_tau = float(gf_cfg.get("loo_tau", 0.05))
        loo_solver = gf_cfg.get("_loo_solver")
        if loo_solver is None:
            return True, R, t, None
        try:
            ok_loo, R_loo, t_loo, _, _ = loo_solver.solve_adaptive(keypoints_2d)
            if not ok_loo:
                return False, R, t, "filter_fail"
            _, loo_score = canonical_filter_C(
                keypoints_2d, loo_solver, R_loo, t_loo, tau_C=loo_tau)
            if loo_score >= loo_tau:
                return False, R, t, "filter_fail"
        except Exception:
            return False, R, t, "filter_fail"
        return True, R, t, None

    raise ValueError(f"Unknown filter_type: {filter_type}")


def generate_pseudo_labels(model, real_loader, pnp_solver, geo_filter,
                           weak_aug, config, device):
    """Generate pseudo-labels for real unlabeled data.

    filter_type dispatch (config.geometric_filter.filter_type):
        ransac (default) | bc | conf | none
    """
    image_size = config["self_training"]["image_size"]
    sigma = config["self_training"]["sigma"]
    peak_threshold = 0.3

    gf_cfg = config["geometric_filter"]
    filter_type = str(gf_cfg.get("filter_type", "ransac")).lower()
    min_kps = int(gf_cfg.get("min_keypoints", 5))

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    accepted = []
    total = 0
    num_pnp_fail = 0
    num_filter_fail = 0
    reproj_errors = []

    model.eval()
    with torch.no_grad():
        for batch in real_loader:
            imgs = batch["img"]

            for i in range(imgs.size(0)):
                total += 1
                img = imgs[i]

                img_weak = weak_aug(img.clone())
                img_input = normalize(img_weak).unsqueeze(0).to(device)
                out_bel, _ = model(img_input)
                belief_maps = out_bel[-1][0].cpu().numpy()

                keypoints_2d = extract_peaks(
                    belief_maps, threshold=peak_threshold, image_size=image_size)

                valid_count = sum(1 for kp in keypoints_2d if kp is not None)
                if valid_count < min_kps:
                    num_pnp_fail += 1
                    continue

                is_valid, R, t, reason = _apply_filter(
                    filter_type, keypoints_2d, pnp_solver, geo_filter,
                    gf_cfg, image_size,
                )

                if R is not None:
                    reproj_all = pnp_solver.reproject(R, t)[:8]
                    det_idx = [k for k, p in enumerate(keypoints_2d[:8]) if p is not None]
                    det_2d = np.array(
                        [[keypoints_2d[k][0], keypoints_2d[k][1]] for k in det_idx],
                        dtype=np.float64)
                    if len(det_idx) > 0:
                        errs = np.linalg.norm(
                            reproj_all[det_idx] - det_2d, axis=1)
                        reproj_errors.append(float(np.mean(errs)))

                if not is_valid:
                    if reason == "pnp_fail":
                        num_pnp_fail += 1
                    else:
                        num_filter_fail += 1
                    continue

                pseudo_beliefs = generate_belief_maps_from_keypoints(
                    keypoints_2d, image_size=image_size,
                    belief_map_size=config["self_training"]["belief_map_size"],
                    sigma=sigma)

                accepted.append({
                    "img": img.cpu(),
                    "beliefs": pseudo_beliefs,
                })

    num_accepted = len(accepted)
    stats = {
        "filter_type": filter_type,
        "total": total,
        "accepted": num_accepted,
        "acceptance_rate": num_accepted / total if total > 0 else 0,
        "pnp_fail": num_pnp_fail,
        "filter_fail": num_filter_fail,
        "reproj_error_mean": float(np.mean(reproj_errors)) if reproj_errors else 0,
    }
    return accepted, stats


def train_one_epoch(model, optimizer, syn_loader, pseudo_dataset,
                    config, device, epoch, round_idx):
    """Train one epoch mixing synthetic GT and pseudo-labeled real data.

    Args:
        model: DOPE network in train mode.
        optimizer: Adam optimizer.
        syn_loader: DataLoader for synthetic data.
        pseudo_dataset: PseudoLabeledDataset (may be empty).
        config: configuration dict.
        device: torch device.
        epoch: current epoch number.
        round_idx: current self-training round.

    Returns:
        dict with loss statistics.
    """
    lambda_real = config["self_training"]["lambda_real"]
    log_interval = config["logging"]["log_interval"]

    model.train()

    # Create pseudo-labeled loader
    pseudo_loader = None
    if len(pseudo_dataset) > 0:
        pseudo_loader = data.DataLoader(
            pseudo_dataset,
            batch_size=min(config["self_training"]["batch_size"], len(pseudo_dataset)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        pseudo_iter = iter(pseudo_loader)

    losses_syn = []
    losses_real = []
    losses_total = []

    for batch_idx, syn_batch in enumerate(syn_loader):
        optimizer.zero_grad()

        # --- Synthetic loss ---
        syn_img = syn_batch["img"].to(device)
        syn_beliefs = syn_batch["beliefs"].to(device)
        syn_affinities = syn_batch["affinities"].to(device)

        out_bel, out_aff = model(syn_img)

        loss_syn = torch.tensor(0.0, device=device)
        for stage in range(len(out_bel)):
            loss_syn += ((out_bel[stage] - syn_beliefs) ** 2).mean()
            loss_syn += ((out_aff[stage] - syn_affinities) ** 2).mean()
        losses_syn.append(loss_syn.item())

        # --- Pseudo-labeled real loss ---
        loss_real = torch.tensor(0.0, device=device)
        if pseudo_loader is not None and len(pseudo_dataset) > 0:
            try:
                real_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                real_batch = next(pseudo_iter)

            real_img = real_batch["img"].to(device)
            real_beliefs = real_batch["beliefs"].to(device)

            out_bel_real, _ = model(real_img)

            # Only belief map loss for pseudo-labels (no affinity GT)
            for stage in range(len(out_bel_real)):
                loss_real += ((out_bel_real[stage] - real_beliefs) ** 2).mean()

            loss_real = lambda_real * loss_real
        losses_real.append(loss_real.item())

        # --- Total loss ---
        total_loss = loss_syn + loss_real
        total_loss.backward()
        optimizer.step()
        losses_total.append(total_loss.item())

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Round {round_idx} Epoch {epoch} "
                  f"[{batch_idx + 1}/{len(syn_loader)}] "
                  f"loss_syn={loss_syn.item():.6f} "
                  f"loss_real={loss_real.item():.6f} "
                  f"total={total_loss.item():.6f}")

    return {
        "loss_syn": float(np.mean(losses_syn)),
        "loss_real": float(np.mean(losses_real)),
        "loss_total": float(np.mean(losses_total)),
    }


def self_training_loop(config, args):
    """Main self-training loop.

    Args:
        config: parsed YAML configuration.
        args: argparse namespace with CLI overrides.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    st_cfg = config["self_training"]
    output_dir = args.output_dir or config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- Load pre-trained model ---
    weights_path = args.pretrained or config["paths"]["pretrained_weights"]
    print(f"Loading pre-trained model: {weights_path}")
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)

    # --- Setup PnP solver and geometric filter ---
    cam_cfg = config["camera"]
    camera_matrix = make_camera_matrix(
        cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"])

    pallet_cfg = config["pallet"]
    pnp_solver = PalletPnPSolver(
        camera_matrix,
        pallet_dims=(pallet_cfg["width"], pallet_cfg["depth"], pallet_cfg["height"]),
        use_ransac=config["pnp"]["use_ransac"],
        ransac_reproj_threshold=config["pnp"]["ransac_reproj_threshold"],
        ransac_iterations=config["pnp"]["ransac_iterations"],
    )

    geo_filter = GeometricFilter(pnp_solver, config["geometric_filter"])

    # LOO filter needs default-ordering PnP solver with correct pallet dims
    filter_type = config["geometric_filter"].get("filter_type", "ransac")
    if filter_type == "ransac_loo":
        loo_solver = PalletPnPSolver(
            camera_matrix,
            pallet_dims=(pallet_cfg["width"], pallet_cfg["depth"], pallet_cfg["height"]))
        config["geometric_filter"]["_loo_solver"] = loo_solver

    # --- Setup augmentations ---
    weak_aug = WeakAugmentation(
        brightness=config["augmentation"]["weak"]["brightness"],
        contrast=config["augmentation"]["weak"]["contrast"],
        noise_std=config["augmentation"]["weak"]["gaussian_noise_std"],
    )
    strong_aug = StrongAugmentation(config["augmentation"]["strong"])

    # --- Setup datasets ---
    syn_dir = args.synthetic_dir or config["paths"]["synthetic_data"]
    real_dir = args.real_dir or config["paths"]["real_unlabeled"]

    syn_dataset = SyntheticDataset(
        syn_dir,
        image_size=st_cfg["image_size"],
        output_size=st_cfg["belief_map_size"],
        sigma=st_cfg["sigma"],
    )
    syn_loader = data.DataLoader(
        syn_dataset,
        batch_size=st_cfg["batch_size"],
        shuffle=True,
        num_workers=st_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    real_dataset = RealUnlabeledDataset(real_dir, image_size=st_cfg["image_size"])
    real_loader = data.DataLoader(
        real_dataset,
        batch_size=st_cfg["batch_size"],
        shuffle=False,
        num_workers=st_cfg["num_workers"],
        pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=st_cfg["learning_rate"],
        weight_decay=st_cfg.get("weight_decay", 0),
    )

    # --- Training loop ---
    num_rounds = st_cfg["num_rounds"]
    epochs_per_round = st_cfg["epochs_per_round"]
    convergence_threshold = st_cfg.get("convergence_threshold", 0.01)
    convergence_patience = st_cfg.get("convergence_patience", 3)

    history = []
    acceptance_rates = []
    converged_count = 0

    print(f"\nStarting self-training: {num_rounds} rounds, "
          f"{epochs_per_round} epochs/round")
    print(f"Synthetic: {len(syn_dataset)} samples, "
          f"Real unlabeled: {len(real_dataset)} images")
    print("=" * 60)

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

        # Step 1: Generate pseudo-labels
        filter_type = config["geometric_filter"].get("filter_type", "ransac")
        print(f"Generating pseudo-labels (filter_type={filter_type})...")
        accepted, pl_stats = generate_pseudo_labels(
            model, real_loader, pnp_solver, geo_filter,
            weak_aug, config, device,
        )
        print(f"  Accepted: {pl_stats['accepted']}/{pl_stats['total']} "
              f"({pl_stats['acceptance_rate']:.1%})")
        print(f"  PnP failures: {pl_stats['pnp_fail']}, "
              f"Filter rejections: {pl_stats['filter_fail']}")
        if pl_stats['reproj_error_mean'] > 0:
            print(f"  Mean reproj error: {pl_stats['reproj_error_mean']:.2f} px")

        # Check convergence
        acceptance_rates.append(pl_stats['acceptance_rate'])
        if len(acceptance_rates) >= 2:
            rate_change = abs(acceptance_rates[-1] - acceptance_rates[-2])
            if rate_change < convergence_threshold:
                converged_count += 1
                if converged_count >= convergence_patience:
                    print(f"\nConverged: acceptance rate stable for "
                          f"{convergence_patience} rounds.")
                    break
            else:
                converged_count = 0

        # Step 2: Train with mixed data
        pseudo_dataset = PseudoLabeledDataset(
            accepted,
            image_size=st_cfg["image_size"],
            strong_aug=strong_aug,
        )
        print(f"Training ({epochs_per_round} epochs, "
              f"{len(syn_dataset)} syn + {len(pseudo_dataset)} pseudo-real)...")

        round_losses = []
        for epoch in range(epochs_per_round):
            epoch_stats = train_one_epoch(
                model, optimizer, syn_loader, pseudo_dataset,
                config, device, epoch, round_idx + 1,
            )
            round_losses.append(epoch_stats)

        # Log round results
        round_time = time.time() - round_start
        avg_loss = np.mean([s["loss_total"] for s in round_losses])
        round_result = {
            "round": round_idx + 1,
            "pseudo_labels": pl_stats,
            "avg_loss": float(avg_loss),
            "time_seconds": round_time,
        }
        history.append(round_result)
        print(f"  Avg loss: {avg_loss:.6f}, Time: {round_time:.1f}s")

        # Save checkpoint
        if config["logging"].get("save_every_round", True):
            ckpt_path = os.path.join(
                output_dir, f"round_{round_idx + 1:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Save final model and history
    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Self-Training Summary")
    print("=" * 60)
    for r in history:
        print(f"  Round {r['round']:2d}: "
              f"accepted={r['pseudo_labels']['accepted']:4d}/"
              f"{r['pseudo_labels']['total']:4d} "
              f"({r['pseudo_labels']['acceptance_rate']:.1%}), "
              f"loss={r['avg_loss']:.6f}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: FixMatch-based Self-Training with Geometric Filter")

    parser.add_argument("--config", type=str,
                        default="config/stage3_selftrain.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pre-trained DOPE weights (overrides config)")
    parser.add_argument("--synthetic_dir", type=str, default=None,
                        help="Path to synthetic training data (overrides config)")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Path to unlabeled real images (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--num_rounds", type=int, default=None,
                        help="Number of self-training rounds (overrides config)")
    parser.add_argument("--epochs_per_round", type=int, default=None,
                        help="Epochs per round (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--filter_type", type=str, default=None,
                        choices=["ransac", "bc", "conf", "none", "ransac_loo"],
                        help="Pseudo-label filter (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.num_rounds is not None:
        config["self_training"]["num_rounds"] = args.num_rounds
    if args.epochs_per_round is not None:
        config["self_training"]["epochs_per_round"] = args.epochs_per_round
    if args.lr is not None:
        config["self_training"]["learning_rate"] = args.lr
    if args.filter_type is not None:
        config["geometric_filter"]["filter_type"] = args.filter_type

    # Run
    self_training_loop(config, args)


if __name__ == "__main__":
    main()
