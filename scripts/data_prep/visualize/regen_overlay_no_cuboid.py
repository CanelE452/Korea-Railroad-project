"""기존 overlay 이미지를 PnP cuboid(노란선) 없이 재생성.

heatmap + keypoint 점 + 필터 텍스트만 유지, PnP cuboid wireframe 제거.

사용법:
    python scripts/data_prep/visualize/regen_overlay_no_cuboid.py \
        --weights weights/v8_ablation_A_coord/final_net_epoch_0065.pth \
        --filter_dir data/pallet/eval_results/v8A_coord_noapril_filters/2_ransac \
        --img_dir data/pallet/raw_data/capture0403noapril/rgb \
        --tag v8A_coord
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PREP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "self_training"))
sys.path.insert(0, DATA_PREP)

from visualize_inference import (load_model, infer, extract_keypoints,
                                  belief_to_heatmap, KP_COLORS)


def draw_overlay_no_cuboid(img, kps_belief, belief_maps, label):
    """heatmap + keypoint 점만 그림 (PnP cuboid 없음)."""
    h, w = img.shape[:2]
    bh, bw = belief_maps.shape[1], belief_maps.shape[2]
    sx, sy = w / bw, h / bh

    vis = img.copy()

    # Belief heatmap overlay (30% opacity)
    heatmap = belief_to_heatmap(belief_maps, img.shape[:2])
    vis = cv2.addWeighted(vis, 0.7, heatmap, 0.3, 0)

    # Predicted keypoints (원본 해상도로 변환)
    for i, kp in enumerate(kps_belief):
        if kp is None:
            continue
        x_orig = kp[0] * sx
        y_orig = kp[1] * sy
        pt = (int(x_orig), int(y_orig))
        cv2.circle(vis, pt, 6, KP_COLORS[i], -1)
        cv2.circle(vis, pt, 7, (0, 0, 0), 1)
        cv2.putText(vis, f"{i}", (pt[0]+8, pt[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLORS[i], 1)

    # 정보 텍스트
    detected = sum(1 for kp in kps_belief if kp is not None)
    info = f"{label} | {detected}/9 kps"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--filter_dir", required=True,
                    help="overlay 파일이 있는 필터 폴더 (예: 2_ransac)")
    ap.add_argument("--img_dir", required=True,
                    help="원본 RGB 이미지 폴더")
    ap.add_argument("--tag", default="v8A_coord")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out_dir", default=None,
                    help="출력 폴더 (기본: filter_dir 내 덮어쓰기)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.filter_dir
    os.makedirs(out_dir, exist_ok=True)

    # 기존 overlay에서 원본 파일명 추출
    overlays = sorted(glob.glob(os.path.join(args.filter_dir, "*_overlay.jpg")))
    stems = [os.path.basename(f).replace("_overlay.jpg", "") for f in overlays]
    print(f"재생성 대상: {len(stems)}장")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    print(f"Model: {args.weights} ({device})")

    for stem in stems:
        # 원본 이미지 찾기
        img_path = None
        for ext in [".jpg", ".png"]:
            cand = os.path.join(args.img_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"  SKIP {stem} — 원본 없음")
            continue

        img_bgr = cv2.imread(img_path)
        belief_maps = infer(model, img_bgr, device)
        kps = extract_keypoints(belief_maps, args.threshold)

        vis = draw_overlay_no_cuboid(img_bgr, kps, belief_maps, args.tag)

        out_path = os.path.join(out_dir, f"{stem}_overlay.jpg")
        cv2.imwrite(out_path, vis)
        detected = sum(1 for k in kps if k is not None)
        print(f"  저장: {out_path} ({detected}/9 kps)")

    print(f"\n완료: {len(stems)}장 → {out_dir}")


if __name__ == "__main__":
    main()
