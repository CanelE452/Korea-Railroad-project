"""Standalone PL extractor (R0/R1/R2 공용).

DOPE 추론 + RANSAC subset consensus 필터 + NDDS 형식 dump.
Phase 1 self-training 실험용 (3차 발표 준비).

사용:
    python scripts/data_prep/inference/extract_pl_v5.py \\
        --weights weights/v8_ablation_A_coord/final_net_epoch_0065.pth \\
        --img_dirs data/outside/capturepallet07/rgb data/outside/capturepallet08/rgb \\
        --output_dir output/pl_outside_r0_ransac \\
        --filter_type ransac
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "self_training"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "data_prep"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix
from geometric_filter import GeometricFilter
from self_train_pseudo import extract_peaks, _apply_filter


def collect_images(img_dirs):
    paths = []
    for d in img_dirs:
        if os.path.isdir(d):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                paths.extend(sorted(glob.glob(os.path.join(d, ext))))
        else:
            paths.extend(sorted(glob.glob(d)))
    return paths


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--img_dirs", required=True, nargs="+",
                   help="One or more image dirs (or glob patterns)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", default="config/stage3_selftrain.yaml")
    p.add_argument("--filter_type", default="ransac",
                   choices=["ransac", "ransac_loo", "bc", "conf", "none"])
    p.add_argument("--ransac_min_consensus", type=int, default=None,
                   help="Override c-threshold (default from config: 6)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Peak threshold")
    p.add_argument("--fx", type=float, default=605.906494140625)
    p.add_argument("--fy", type=float, default=605.9697875976562)
    p.add_argument("--cx", type=float, default=317.59619140625)
    p.add_argument("--cy", type=float, default=256.29229736328125)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--copy_images", action="store_true",
                   help="Copy images (default: symlink)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Limit total frames (for quick test)")
    args = p.parse_args()

    # config 로드 + override
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.filter_type:
        config["geometric_filter"]["filter_type"] = args.filter_type
    if args.ransac_min_consensus is not None:
        config["geometric_filter"]["ransac_min_consensus"] = args.ransac_min_consensus

    # 이미지 수집
    img_paths = collect_images(args.img_dirs)
    if args.max_frames:
        img_paths = img_paths[: args.max_frames]
    print(f"Collected {len(img_paths)} images from {len(args.img_dirs)} sources")
    if len(img_paths) == 0:
        print("No images found. Abort.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.weights}")
    model = load_model(args.weights, device)

    # 카메라 + PnP + filter
    cam_matrix = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pallet = config["pallet"]
    pnp_solver = PalletPnPSolver(
        cam_matrix,
        pallet_dims=(pallet["width"], pallet["depth"], pallet["height"]),
        use_ransac=config["pnp"]["use_ransac"],
        ransac_reproj_threshold=config["pnp"]["ransac_reproj_threshold"],
        ransac_iterations=config["pnp"]["ransac_iterations"],
    )
    geo_filter = GeometricFilter(pnp_solver, config["geometric_filter"])

    if args.filter_type == "ransac_loo":
        loo_solver = PalletPnPSolver(
            cam_matrix,
            pallet_dims=(pallet["width"], pallet["depth"], pallet["height"]))
        config["geometric_filter"]["_loo_solver"] = loo_solver

    image_size = 448
    min_kps = int(config["geometric_filter"].get("min_keypoints", 5))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 추론 루프
    accepted_count = 0
    pnp_fail = 0
    filter_fail = 0
    accepted_log = []  # list of dicts

    t_start = time.time()
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (image_size, image_size))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()

        # extract_peaks 는 image_size(448) 좌표계 반환
        keypoints_2d = extract_peaks(belief, threshold=args.threshold, image_size=image_size)
        valid_count = sum(1 for kp in keypoints_2d if kp is not None)
        if valid_count < min_kps:
            pnp_fail += 1
            continue

        is_valid, R, t, reason = _apply_filter(
            args.filter_type, keypoints_2d, pnp_solver, geo_filter,
            config["geometric_filter"], image_size,
        )

        if not is_valid:
            if reason == "pnp_fail":
                pnp_fail += 1
            else:
                filter_fail += 1
            continue

        # 통과 → NDDS dump
        # keypoints_2d 는 448 기준 → 원본 좌표로 스케일
        # None keypoint 는 [-100,-100] (CleanVisiiDopeLoader 가 image 밖으로
        # 판정해 belief 학습에서 자동 제외)
        sx = w0 / image_size
        sy = h0 / image_size
        kps_orig = []
        for kp in keypoints_2d:
            if kp is None:
                kps_orig.append([-100.0, -100.0])
            else:
                kps_orig.append([float(kp[0]) * sx, float(kp[1]) * sy])
        # centroid (9번째) 는 추론으로 가능하면 그대로, 아니면 평균
        projected_cuboid = kps_orig[:9]

        # pose 4x4 행렬
        pose_transform = np.eye(4)
        if R is not None and t is not None:
            pose_transform[:3, :3] = R
            pose_transform[:3, 3] = np.array(t).flatten()

        out_basename = f"{accepted_count:06d}"
        out_png = os.path.join(args.output_dir, out_basename + ".png")
        out_json = os.path.join(args.output_dir, out_basename + ".json")

        # 이미지 (원본 640x480 그대로 저장)
        if args.copy_images:
            shutil.copy2(img_path, out_png)
        else:
            # symlink 가 윈도우에서 안 될 수 있어 cv2 로 다시 저장이 안전
            cv2.imwrite(out_png, img)

        # NDDS JSON
        ndds = {
            "camera_data": {
                "width": w0,
                "height": h0,
                "intrinsics": {
                    "fx": args.fx, "fy": args.fy,
                    "cx": args.cx, "cy": args.cy,
                },
            },
            "objects": [{
                "class": "pallet",
                "name": "real_pallet",
                "visibility": 1,
                "pose_transform": pose_transform.tolist(),
                "projected_cuboid": projected_cuboid,
                "projected_cuboid_centroid": projected_cuboid[8] if len(projected_cuboid) >= 9 else [0.0, 0.0],
            }],
        }
        with open(out_json, "w") as f:
            json.dump(ndds, f, indent=2)

        accepted_log.append({
            "src_image": img_path,
            "out_basename": out_basename,
        })
        accepted_count += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(img_paths)}] accepted={accepted_count} "
                  f"pnp_fail={pnp_fail} filter_fail={filter_fail} "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t_start
    total = len(img_paths)
    summary = {
        "total": total,
        "accepted": accepted_count,
        "acceptance_rate": accepted_count / total if total > 0 else 0.0,
        "pnp_fail": pnp_fail,
        "filter_fail": filter_fail,
        "filter_type": args.filter_type,
        "ransac_min_consensus": config["geometric_filter"].get("ransac_min_consensus"),
        "weights": args.weights,
        "img_dirs": args.img_dirs,
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(args.output_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "_accepted_log.json"), "w") as f:
        json.dump(accepted_log, f, indent=2)

    print()
    print("=" * 60)
    print(f"PL extraction complete: {args.output_dir}")
    print(f"  Total       : {total}")
    print(f"  Accepted    : {accepted_count} ({accepted_count/total*100:.1f}%)")
    print(f"  PnP fail    : {pnp_fail}")
    print(f"  Filter fail : {filter_fail}")
    print(f"  Elapsed     : {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
