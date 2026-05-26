"""manual GT 데이터로 YAW_OFFSET_DEG, DEPTH_FRONT_SIGN 직접 산출.

challenge/data/capturepallet07_manual_gt/ 의 27 프레임은 사용자가 직접 라벨링한
정확한 R/t (face-flip ambiguity 해결됨). 이 데이터로:

1. **가장 정렬에 가까운 frame** 찾기 (R[2,2] 절대값 최대)
2. 그 frame 의 raw_yaw 계산 → YAW_OFFSET_DEG 후보
3. corners {0,1,2,3} (라벨러의 "front face") 의 카메라 z 평균이 centroid 보다
   가까운지/먼지로 → DEPTH_FRONT_SIGN 결정

추가: 모든 frame의 raw_yaw 분포로 식 후보 비교 (atan2(x,-z) vs atan2(x,+z)).

사용:
    python depth_cam/tools/calibrate_from_manual_gt.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_DEPTH_CAM_DIR)


def load_manual_gt_frames():
    """27 프레임 manual GT 로드."""
    gt_dir = os.path.join(_REPO_ROOT, "challenge", "data", "capturepallet07_manual_gt")
    frames = []
    for jp in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        with open(jp) as f:
            j = json.load(f)
        if not j.get("objects"):
            continue
        obj = j["objects"][0]
        R = np.array(obj["pose_transform"])[:3, :3]
        t = np.array(obj["pose_transform"])[:3, 3]
        proj8 = np.array(obj["projected_cuboid"])  # (8, 2)
        frames.append({
            "name": os.path.basename(jp),
            "R": R,
            "t_m": t,
            "proj8": proj8,
        })
    return frames


def analyze_alignment(frames):
    """각 frame 의 R[:,2] (팔레트 +Z 의 카메라 좌표) + raw yaw 두 식 비교."""
    print(f"\n[INFO] {len(frames)} 프레임 분석\n")
    print(f"{'frame':30s}  R[:,2] (front axis cam)            "
          f"yaw_A=atan2(x,-z)  yaw_B=atan2(x,+z)  t_z")
    print("-" * 130)

    rows = []
    for f in frames:
        R = f["R"]
        front = R @ np.array([0, 0, 1])
        yaw_A = float(np.degrees(np.arctan2(front[0], -front[2])))
        yaw_B = float(np.degrees(np.arctan2(front[0], +front[2])))
        rows.append({
            "name": f["name"],
            "front": front,
            "yaw_A": yaw_A,
            "yaw_B": yaw_B,
            "t_z": float(f["t_m"][2]),
            "R": R,
            "t_m": f["t_m"],
            "proj8": f["proj8"],
        })
        print(f"  {f['name'][:25]:30s}  "
              f"[{front[0]:+.3f}, {front[1]:+.3f}, {front[2]:+.3f}]   "
              f"{yaw_A:+8.2f}         {yaw_B:+8.2f}        {f['t_m'][2]:.2f}")
    return rows


def find_most_aligned(rows):
    """R[2,2] 절대값이 최대인 frame = 카메라가 가장 정면에서 본 frame."""
    rows_sorted = sorted(rows, key=lambda r: abs(r["front"][2]), reverse=True)
    return rows_sorted[:3]


def determine_depth_sign(rows):
    """corners {0,1,2,3} 의 카메라 z 평균 vs centroid z 비교.

    가정: 라벨러는 "팔레트 정면" 을 corners {0,1,2,3} 으로 라벨링.
    카메라가 팔레트 정면을 본 frame 에서:
      - corners 0-3 z < corners 4-7 z → 라벨러 정면 = 카메라 쪽 (DEPTH_FRONT_SIGN = -1 in default frame)
      - corners 0-3 z > corners 4-7 z → 라벨러 정면 = 카메라 반대 쪽 (DEPTH_FRONT_SIGN = +1)
    """
    # 가장 정렬된 frame 들에서만 평가
    aligned = find_most_aligned(rows)
    front_z_means = []
    back_z_means = []
    for r in aligned:
        # NDDS projected_cuboid 만 있으므로 카메라 좌표 8 corner 가 직접 없음.
        # 대신 default_z180 contract 사용해서 카메라 좌표 8 corner 추정:
        default_z180_local = np.array([
            [-0.5, +0.075, +0.6],  # 0
            [+0.5, +0.075, +0.6],  # 1
            [+0.5, -0.075, +0.6],  # 2
            [-0.5, -0.075, +0.6],  # 3
            [-0.5, +0.075, -0.6],  # 4
            [+0.5, +0.075, -0.6],  # 5
            [+0.5, -0.075, -0.6],  # 6
            [-0.5, -0.075, -0.6],  # 7
        ]) * np.array([1.0, 0.15/0.15, 1.2/1.2])  # 실제 dim 반영 (W=1.0, H=0.15, L=1.2)
        # dim 보정: 위 default_z180_local 은 이미 (W=1.0, H=0.15, L=1.2) 기준이라 그대로.
        R = r["R"]
        t = r["t_m"]
        corners_cam = (R @ default_z180_local.T).T + t
        front_z = corners_cam[:4, 2].mean()
        back_z  = corners_cam[4:, 2].mean()
        front_z_means.append(front_z)
        back_z_means.append(back_z)
        print(f"  [{r['name'][:15]}] front face z={front_z:.3f}m  back face z={back_z:.3f}m  "
              f"({'front closer' if front_z < back_z else 'back closer'})")
    front_mean = np.mean(front_z_means)
    back_mean  = np.mean(back_z_means)
    return front_mean, back_mean


def main():
    print("=" * 80)
    print(" Calibrate YAW_OFFSET_DEG, DEPTH_FRONT_SIGN from manual GT")
    print("=" * 80)
    frames = load_manual_gt_frames()
    if not frames:
        print("[ERROR] manual GT 없음")
        return

    rows = analyze_alignment(frames)

    print()
    print("=" * 80)
    print(" 가장 정렬에 가까운 3 frame (|R[2,2]| max)")
    print("=" * 80)
    most_aligned = find_most_aligned(rows)
    for r in most_aligned:
        print(f"  {r['name']:30s}  "
              f"R[2,2]={r['front'][2]:+.4f}  yaw_A={r['yaw_A']:+.2f}  yaw_B={r['yaw_B']:+.2f}")

    print()
    print("-" * 80)
    print(" front/back face z 비교 (라벨러의 'front face' = 카메라 쪽인가?)")
    print("-" * 80)
    front_z, back_z = determine_depth_sign(rows)
    print(f"\n평균: front face z = {front_z:.3f}m, back face z = {back_z:.3f}m")

    print()
    print("=" * 80)
    print(" 결론 (config 값 권장)")
    print("=" * 80)

    # 정렬 frame 의 yaw_A, yaw_B 중 값이 작은 것이 정답 식
    avg_yaw_A = np.mean([abs(r["yaw_A"]) for r in most_aligned])
    avg_yaw_B = np.mean([abs(r["yaw_B"]) for r in most_aligned])
    print(f"  정렬된 frame 들의 |yaw_A| 평균: {avg_yaw_A:.2f}° (atan2(x,-z))")
    print(f"  정렬된 frame 들의 |yaw_B| 평균: {avg_yaw_B:.2f}° (atan2(x,+z))")

    if avg_yaw_B < avg_yaw_A:
        print()
        print("  → 식 B (atan2(front[0], +front[2])) 사용 권장.")
        print("    geometry.py: yaw_rad = np.arctan2(front_axis_cam[0], +front_axis_cam[2])")
        print(f"  → YAW_OFFSET_DEG = {np.mean([r['yaw_B'] for r in most_aligned]):+.2f}")
    else:
        print()
        print("  → 식 A (atan2(front[0], -front[2])) 사용 (현재 코드).")
        print(f"  → YAW_OFFSET_DEG = {np.mean([r['yaw_A'] for r in most_aligned]):+.2f}")

    # DEPTH_FRONT_SIGN
    print()
    if front_z < back_z:
        print("  라벨러의 'front face' (corners 0-3) 가 카메라에 더 가까움.")
        print("  → DEPTH_FRONT_SIGN = -1.0  (front_center = R @ (0,0,-depth/2) + t)")
    else:
        print("  라벨러의 'front face' (corners 0-3) 가 카메라에서 더 멀음.")
        print("  → DEPTH_FRONT_SIGN = +1.0  (front_center = R @ (0,0,+depth/2) + t)")
    print("=" * 80)


if __name__ == "__main__":
    main()
