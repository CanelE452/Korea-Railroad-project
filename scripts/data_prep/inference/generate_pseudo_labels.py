"""Filter-passed 이미지에서 pseudo-label NDDS JSON 생성 + 합성 데이터와 병합.

1) filter_details.csv에서 all_passed=True인 이미지 목록 확인
2) 해당 이미지를 re-infer → PnP → reproject → NDDS JSON 저장
3) 합성 데이터 + pseudo-label을 하나의 학습 디렉토리로 병합

사용법:
    python scripts/data_prep/generate_pseudo_labels.py \
        --weights weights/mixed_v1/final_net_epoch_0060.pth \
        --filter_csv data/pallet/real_data_results_mixed_v1/filter_details.csv \
        --img_dir data/pallet/real_data \
        --syn_dir data/pallet/training_data/mixed_v1_train \
        --output_dir data/pallet/training_data/selftrain_r1
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # data_prep for shared libs

from visualize_inference import load_model, infer, extract_keypoints
from pnp_solver import PalletPnPSolver, make_camera_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--filter_csv", required=True)
    parser.add_argument("--img_dir", required=True, help="원본 이미지 디렉토리")
    parser.add_argument("--syn_dir", required=True, help="합성 학습 데이터 디렉토리")
    parser.add_argument("--output_dir", required=True, help="병합된 학습 데이터 출력")
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) filter_details.csv에서 passed 목록 읽기
    passed_files = []
    with open(args.filter_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["all_passed"] == "True":
                passed_files.append(row["filename"])
    print(f"Filter-passed images: {len(passed_files)}")

    # 2) 합성 데이터 복사 (symlink 대신 copy — Windows 호환)
    syn_pngs = sorted(glob.glob(os.path.join(args.syn_dir, "*.png")))
    syn_count = 0
    print(f"Copying {len(syn_pngs)} synthetic images...")
    for png_path in syn_pngs:
        json_path = os.path.splitext(png_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
        out_id = f"{syn_count:06d}"
        shutil.copy2(png_path, os.path.join(args.output_dir, out_id + ".png"))
        shutil.copy2(json_path, os.path.join(args.output_dir, out_id + ".json"))
        syn_count += 1

    print(f"Synthetic copied: {syn_count}")

    # 3) Pseudo-label 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    pseudo_count = 0
    idx = syn_count  # 합성 데이터 뒤부터 번호 이어감

    for basename in passed_files:
        # 원본 이미지 찾기
        img_path = None
        for ext in [".jpg", ".png"]:
            candidate = os.path.join(args.img_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 추론
        belief = infer(model, img, device)
        pred_kps = extract_keypoints(belief, 0.3)

        # 원본 해상도 변환
        h, w = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = w / bw, h / bh
        pred_orig = []
        for kp in pred_kps:
            if kp is None:
                pred_orig.append(None)
            else:
                pred_orig.append((kp[0] * sx, kp[1] * sy))

        # PnP
        success, R, t, _ = pnp.solve(pred_orig)
        if not success:
            continue

        # Reproject → annotation
        reproj = pnp.reproject(R, t)

        annotation = {
            "camera_data": {
                "width": w, "height": h,
                "intrinsics": {
                    "fx": args.fx, "fy": args.fy,
                    "cx": args.cx, "cy": args.cy
                },
            },
            "objects": [{
                "class": "pallet",
                "name": "pseudo_label",
                "visibility": 1.0,
                "projected_cuboid": reproj[:8].tolist(),
                "projected_cuboid_centroid": reproj[8].tolist(),
            }],
        }

        out_id = f"{idx:06d}"
        cv2.imwrite(os.path.join(args.output_dir, out_id + ".png"), img)
        with open(os.path.join(args.output_dir, out_id + ".json"), "w") as f:
            json.dump(annotation, f, indent=2)

        idx += 1
        pseudo_count += 1

    print(f"Pseudo-labels generated: {pseudo_count}")
    print(f"\nTotal training data: {syn_count} syn + {pseudo_count} pseudo = {syn_count + pseudo_count}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
