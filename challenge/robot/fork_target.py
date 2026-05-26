"""
challenge/robot/fork_target.py

DOPE 6D pose → forklift fork entry pose 변환 (스켈레톤).

입력:
  - R_world_pallet (3x3): 팔레트 → 카메라 좌표 (DOPE quaternion에서 변환)
  - t_world_pallet (3,):  팔레트 중심 위치 (m, 카메라 기준)
  - challenge/config/task.yaml 의 robot.fork.* 설정

출력:
  - approach_pose: 진입 전 정지 위치 (정면 0.6m, fork height)
  - insert_pose:   포크 삽입 완료 위치 (0.5m 진입)
  - yaw_world:     팔레트 정면 방향 (rad)

본 파일은 6D pose가 안정적으로 detection되었을 때 forklift 제어 stack
(또는 sim)에 전달할 target을 계산하는 entry point. 실제 모션 플래닝은
별도 구현 또는 외부 ROS node에 위임.
"""

from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import yaml


@dataclass
class ForkTarget:
    approach_pose_m: np.ndarray   # (3,) — 진입 전 정지 위치 (m, world)
    insert_pose_m:   np.ndarray   # (3,) — 포크 삽입 완료 위치 (m, world)
    yaw_world_rad:   float        # 팔레트 정면을 향한 yaw
    left_entry_m:    np.ndarray   # (3,) — 좌측 포크 삽입 진입점 (m, world)
    right_entry_m:   np.ndarray   # (3,) — 우측 포크 삽입 진입점 (m, world)

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in asdict(self).items()}


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_fork_target(R_pallet: np.ndarray,
                        t_pallet_m: np.ndarray,
                        cfg: dict) -> ForkTarget:
    """팔레트 6D pose → fork 진입 target.

    R_pallet: 3x3, 팔레트 로컬 축 → 카메라 좌표
    t_pallet_m: (3,) 팔레트 중심 (m, 카메라 좌표)
    cfg: task.yaml dict
    """
    fc = cfg["robot"]["fork"]
    left_local  = np.array(fc["left_entry"],  dtype=np.float64)
    right_local = np.array(fc["right_entry"], dtype=np.float64)
    approach_d  = float(fc["approach_offset_m"])
    insert_d    = float(fc["insertion_depth_m"])

    # 팔레트 정면 방향 (로컬 +Z) → world
    front_dir = R_pallet @ np.array([0.0, 0.0, 1.0])
    front_dir[1] = 0.0  # 수평면 사영 (Y=UP)
    n = np.linalg.norm(front_dir)
    if n < 1e-6:
        front_dir = np.array([0.0, 0.0, 1.0])
    else:
        front_dir /= n

    yaw_world_rad = float(np.arctan2(front_dir[0], front_dir[2]))

    # 좌/우 포크 진입점 (팔레트 로컬 → world)
    left_world  = R_pallet @ left_local  + t_pallet_m
    right_world = R_pallet @ right_local + t_pallet_m

    # forklift는 양 포크 중심선을 따라 진입한다고 가정
    mid_entry = 0.5 * (left_world + right_world)
    approach_pose = mid_entry + front_dir * approach_d
    insert_pose   = mid_entry - front_dir * insert_d  # -front_dir = 팔레트 내부

    return ForkTarget(
        approach_pose_m=approach_pose,
        insert_pose_m=insert_pose,
        yaw_world_rad=yaw_world_rad,
        left_entry_m=left_world,
        right_entry_m=right_world,
    )


def _smoke_test(cfg):
    # 식별: 팔레트가 카메라 정면 2m, yaw=0
    R = np.eye(3)
    t = np.array([0.0, 0.0, 2.0])
    target = compute_fork_target(R, t, cfg)
    print("[smoke] approach :", target.approach_pose_m)
    print("[smoke] insert   :", target.insert_pose_m)
    print("[smoke] yaw[rad] :", target.yaw_world_rad)
    print("[smoke] left/rt  :", target.left_entry_m, target.right_entry_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="challenge/config/task.yaml 경로 (기본: repo 자동탐색)")
    ap.add_argument("--smoke", action="store_true", help="더미 입력으로 동작만 확인")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = args.config or os.path.join(repo_root, "challenge", "config", "task.yaml")
    cfg = load_cfg(cfg_path)

    if args.smoke:
        _smoke_test(cfg)
        return

    print("[TODO] 실시간 detection 결과(R, t)를 입력받아 ForkTarget을 publish하는 루프.")
    print("       run_live.py와 통합하거나, ROS topic으로 받아오도록 확장.")


if __name__ == "__main__":
    main()
