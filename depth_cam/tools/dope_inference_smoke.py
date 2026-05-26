"""DOPE 모델 실제 로드 + 1프레임 추론 smoke.

depth_cam 통합 후 Perception() 가 정상 작동하는지 검증.
- weight 로드 + Cuboid3d Z180 inject 확인
- 1프레임 추론 → fsm_inputs_from_pose 변환 → 결과 출력

사용:
    python depth_cam/tools/dope_inference_smoke.py \\
        --seq data/outside/capturepallet02 \\
        --frame_idx 0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DEPTH_CAM_DIR)

from calib.config import PALLET_WIDTH_M, PALLET_DEPTH_M, MODEL_PATH
from calib.geometry import fsm_inputs_from_pose
from calib.perception import Perception


def load_frame(seq_dir: str, frame_idx: int):
    """capturepallet* 형식에서 (rgb, depth_frame_stub, K) 로드."""
    rgb_paths = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB in {seq_dir}")

    rgb_path = rgb_paths[min(frame_idx, len(rgb_paths) - 1)]
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load {rgb_path}")

    # depth (uint16 mm)
    depth_frame = None
    if depth_paths:
        depth_path = depth_paths[min(frame_idx, len(depth_paths) - 1)]
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d is not None and d.dtype == np.uint16:
            # NpDepthFrame stub (challenge/scripts/run_live_io.py 에 있음)
            sys.path.insert(0, os.path.join(_DEPTH_CAM_DIR, "..", "challenge", "scripts"))
            from run_live_io import NpDepthFrame
            depth_frame = NpDepthFrame(d)

    # K (cam_K.txt 있으면 사용)
    K_path = os.path.join(seq_dir, "cam_K.txt")
    if os.path.isfile(K_path):
        K = np.loadtxt(K_path).reshape(3, 3)
    else:
        # fallback: config.py
        from calib.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
        K = np.array([[CAMERA_FX, 0, CAMERA_CX],
                      [0, CAMERA_FY, CAMERA_CY],
                      [0, 0, 1]], dtype=np.float64)

    print(f"[INFO] frame: {os.path.basename(rgb_path)}")
    print(f"[INFO] img shape: {img.shape}")
    print(f"[INFO] depth: {'OK' if depth_frame else 'none'}")
    print(f"[INFO] K: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return img, depth_frame, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/outside/capturepallet02")
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1, help="여러 프레임 시도")
    args = ap.parse_args()

    repo_root = os.path.dirname(_DEPTH_CAM_DIR)
    seq_dir = args.seq if os.path.isabs(args.seq) else os.path.join(repo_root, args.seq)

    print("=" * 70)
    print("[smoke] DOPE 모델 로드 + 추론 검증")
    print("=" * 70)
    print(f"[INFO] weights: {MODEL_PATH}")
    print(f"[INFO] pallet dim (W,D): ({PALLET_WIDTH_M}, {PALLET_DEPTH_M}) m")
    print()

    # 1. 모델 로드
    print("--- Step 1: Perception() 인스턴스화 (weight load + Cuboid3d Z180) ---")
    perception = Perception()
    print()

    # 2. 1프레임 (또는 여러 프레임) 추론
    print(f"--- Step 2: 추론 {args.n_frames} 프레임 ({seq_dir}) ---")
    for i in range(args.n_frames):
        idx = args.frame_idx + i
        try:
            img, depth_frame, K = load_frame(seq_dir, idx)
        except Exception as e:
            print(f"[ERROR] frame {idx}: {e}")
            continue

        pose = perception.infer(img, depth_frame=depth_frame, K=K)

        print()
        print(f"[Result frame {idx}]")
        print(f"  ok        : {pose['ok']}")
        print(f"  reason    : {pose['reason']}")
        print(f"  confirmed : {pose['confirmed']}")
        if pose['ok']:
            R = pose['R_pallet']
            t_cm = pose['t_pallet_cm']
            print(f"  R_pallet  : ")
            for row in R:
                print(f"              [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
            print(f"  t (cm)    : [{t_cm[0]:+.2f}, {t_cm[1]:+.2f}, {t_cm[2]:+.2f}]")

            offset, dist_z, yaw, width = fsm_inputs_from_pose(
                R, t_cm, PALLET_DEPTH_M, PALLET_WIDTH_M
            )
            print()
            print(f"  >>> FSM inputs <<<")
            print(f"  offset_smooth: ({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}) m")
            print(f"  dist_z       : {dist_z:.3f} m")
            print(f"  yaw_smooth   : {yaw:+.2f}°")
            print(f"  width        : {width:.3f} m")

    print()
    print("=" * 70)
    print("[smoke] 완료")


if __name__ == "__main__":
    main()
