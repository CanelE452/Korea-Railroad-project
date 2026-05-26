# calib/geometry.py
# 3D 포인트 구성, 평면 적합으로 yaw 계산, bbox 클램프

import numpy as np
import pyrealsense2 as rs
# sklearn은 legacy fit_plane_yaw_from_points() 에서만 사용 → lazy import
from typing import Tuple, Optional
from .config import PLANE_INLIER_THRESH, PLANE_MAX_TRIALS, MIN_POINTS

def compute_yaw_deg_from_plane(a: float, b: float) -> float:
    # z = a x + b y + c → 법선(nx,ny,nz)=(-a,-b,1), 여기서 yaw는 x-z 투영을 사용
    nx, nz = -a, 1.0
    return float(np.degrees(np.arctan2(nx, nz)))

def robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask_or_roi,
                                   stride=3, z_inlier_thresh=0.06, min_points=120) -> Tuple[bool, Optional[np.ndarray]]:
    ys, xs = np.where(mask_or_roi > 0)
    if len(xs) == 0:
        return False, None
    xs = xs[::stride].astype(np.int32)
    ys = ys[::stride].astype(np.int32)

    dists = np.array([depth_frame.get_distance(int(u), int(v)) for u, v in zip(xs, ys)],
                     dtype=np.float32)
    valid = np.isfinite(dists) & (dists > 0)
    if not np.any(valid):
        return False, None
    xs_v, ys_v, d_v = xs[valid], ys[valid], dists[valid]
    if len(d_v) < min_points:
        return False, None

    pts = np.array([
        rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(d))
        for u, v, d in zip(xs_v, ys_v, d_v)
    ], dtype=np.float32)

    z_med = np.median(pts[:, 2])
    inliers = np.abs(pts[:, 2] - z_med) <= float(z_inlier_thresh)
    pts_in = pts[inliers]
    if len(pts_in) < min_points:
        return False, None
    return True, pts_in

def fit_plane_yaw_from_points(pts_in: np.ndarray):
    from sklearn.linear_model import RANSACRegressor  # lazy import (legacy only)
    X = pts_in[:, :2]; Z = pts_in[:, 2]
    ransac = RANSACRegressor(residual_threshold=PLANE_INLIER_THRESH,
                             max_trials=PLANE_MAX_TRIALS, random_state=0)
    ransac.fit(X, Z)
    if ransac.inlier_mask_ is None or ransac.inlier_mask_.sum() < max(30, MIN_POINTS//2):
        return False, None, None, None
    a, b = ransac.estimator_.coef_
    yaw = compute_yaw_deg_from_plane(a, b)
    return True, yaw, a, b

def clamp_bbox(x1,y1,x2,y2,W,H):
    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(0, min(W-1, int(x2)))
    y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1,y1,x2,y2

# ============================================================================
# [6D 통합, 2026-05-22]
# DOPE pose (R_pallet, t_pallet_cm) → FSM 입력 (offset, dist_z, yaw, width)
# ----------------------------------------------------------------------------
# 핵심 변환:
#   1) DOPE location 은 cuboid centroid (cm). m 로 변환.
#   2) 기존 RANSAC 흐름은 팔레트 앞면 중앙 기준. 무게중심 → 앞면 중앙 변환:
#        front_center_local = (0, 0, +depth/2)
#        front_center_cam   = R @ front_center_local + t_m
#   3) yaw_smooth = atan2(front_axis_cam[0], -front_axis_cam[2])
#      where front_axis_cam = R @ [0, 0, 1]
#      정렬 시 yaw=0, 팔레트가 카메라 시점 오른쪽으로 돌면 yaw>0 → FSM ROT_RIGHT.
#   4) width 는 detection ok 시 PALLET_WIDTH_M 상수 (실시간 측정 불필요).
#
# Twin-PnP 검증 결과 (depth_cam/tools/twin_pnp_check.py, 2026-05-22):
#   - dim = (1.0, 1.2, 0.15) m
#   - PnP contract = default Cuboid3d @ diag([-1,-1,+1]) (Z축 180°)
#   - 50/50 프레임 reproj 2.89px, |dt| 0.085m, R 완벽 일치
# ============================================================================

def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def fsm_inputs_from_pose(R_pallet: np.ndarray,
                         t_pallet_cm: np.ndarray,
                         pallet_depth_m: float,
                         pallet_width_m: float,
                         yaw_offset_deg: float = 0.0,
                         depth_front_sign: float = +1.0) -> Tuple[Tuple[float, float, float], float, float, float]:
    """DOPE 6D pose → FSM 입력값.

    Args:
        R_pallet:        (3,3) 팔레트 로컬 → 카메라 좌표 회전 (default_z180 적용 완료 가정)
        t_pallet_cm:     (3,)  팔레트 centroid 의 카메라 좌표 (cm)
        pallet_depth_m:  팔레트 Z 방향 길이 (m). 무게중심 → 앞면 중앙 변환에 사용.
        pallet_width_m:  팔레트 X 방향 길이 (m). detected_length 로 반환.
        yaw_offset_deg:  시연 환경 보정 (정렬 시 yaw=0 되도록). 시연 직전에 측정.
        depth_front_sign: +1=front_local +Z 방향, -1=-Z 방향. 시연에서 dist_z 부호로 확인.

    Returns:
        offset_now_m: (x, y, z) — 앞면 중앙의 카메라 좌표 (m)
        dist_z_m:    앞면 중앙의 z (m)
        yaw_deg:     팔레트 정면 vs 카메라 정면 각도 (deg, +는 오른쪽 회전)
        width_m:     pallet_width_m 그대로 반환 (detection ok 시)
    """
    R = np.asarray(R_pallet, dtype=np.float64)
    t_m = np.asarray(t_pallet_cm, dtype=np.float64) / 100.0  # cm → m

    # (1) 무게중심 → 앞면 중앙 (depth_front_sign 으로 +/-Z 토글 가능)
    front_center_local_m = np.array(
        [0.0, 0.0, depth_front_sign * pallet_depth_m / 2.0], dtype=np.float64
    )
    front_center_cam_m   = R @ front_center_local_m + t_m

    # (2) yaw_smooth — atan2(front[0], -front[2]) + offset 보정
    front_axis_cam = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw_rad = float(np.arctan2(front_axis_cam[0], -front_axis_cam[2]))
    yaw_deg_raw = float(np.degrees(yaw_rad))
    yaw_deg = wrap_to_180(yaw_deg_raw - yaw_offset_deg)

    return (
        (float(front_center_cam_m[0]),
         float(front_center_cam_m[1]),
         float(front_center_cam_m[2])),  # offset_now
        float(front_center_cam_m[2]),     # dist_z
        yaw_deg,
        float(pallet_width_m),
    )


def ema_scalar(prev: Optional[float], cur: float, alpha: float) -> float:
    """간단한 1D EMA. prev 가 None 이면 cur 그대로."""
    if prev is None:
        return float(cur)
    return float((1.0 - alpha) * prev + alpha * cur)


def ema_tuple(prev: Optional[Tuple[float, ...]],
              cur: Tuple[float, ...],
              alpha: float) -> Tuple[float, ...]:
    """N-tuple EMA."""
    if prev is None:
        return tuple(float(v) for v in cur)
    return tuple(float((1.0 - alpha) * p + alpha * c) for p, c in zip(prev, cur))


# ----------------------------------------------------------------------
# [legacy] offset/width 계산 유틸 (YOLO + RANSAC 흐름 — 6D 통합 후 호출 안 됨)
#  - 요구사항: offset은 중앙값(median) 기반으로 계산
#  - width는 기존 관행 유지(min-max 범위)
# ----------------------------------------------------------------------
def compute_offset_and_width(pts_in: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    pts_in: (N,3) ndarray of 3D points in camera coords (x, y, z)
    return:
      ok: bool
      offset_x: float (중앙값 기반)
      width_x:  float (min-max 기반)
    """
    if pts_in is None or len(pts_in) == 0:
        return False, None, None

    X = pts_in[:, 0].astype(np.float32)

    # ---- offset: 중앙값(median) ----
    offset_x = float(np.median(X))

    # ---- width: 기존대로 min-max ----
    if len(X) > 1:
        width_x = float(np.max(X) - np.min(X))
    else:
        width_x = 0.0

    return True, offset_x, width_x
