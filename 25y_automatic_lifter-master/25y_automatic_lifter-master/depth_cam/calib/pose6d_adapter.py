# calib/pose6d_adapter.py
# -----------------------------------------------------------------------------
# DOPE / FoundationPose 6D pose → ALIGN FSM 변수 (ψ_pallet, d_lateral, d_forward).
#
# 좌표계: OpenCV (+X right, +Y down, +Z forward / optical axis).
#
# DOPE 9 keypoint convention (v4):
#   0: front-top-LEFT   1: front-top-RIGHT
#   2: front-bot-RIGHT  3: front-bot-LEFT
#   4: rear-top-LEFT    5: rear-top-RIGHT
#   6: rear-bot-RIGHT   7: rear-bot-LEFT
#   8: centroid (무게중심)  ← ALIGN 의 기준점
#
# ALIGN 변수 정의 (모두 centroid 기준):
#   ψ_pallet (deg) : forklift heading 과 pallet forward 축 (centroid 를 지남)
#                    사이 각도. ψ=0 일 때 forklift.yaw == pallet.yaw.
#                    R 의 column (팔레트 +Z 축의 camera frame 표현) 으로 추출.
#                    회전이라 기준점에 무관 (centroid든 어디든 같은 값).
#   d_lateral (m)  : 카메라 +X (right) 방향 centroid 오프셋. 목표 0.
#   d_forward (m)  : 카메라 +Z (forward) 방향 centroid 거리. 목표 ALIGN_DIST_M
#                    (centroid 기준).
#
# FSM 정렬 목표:
#   ψ_pallet ≈ 0 AND |d_lateral| ≤ OFF_TOL_M AND |d_forward - ALIGN_DIST_M| ≤ ALIGN_BAND_M
#   → READY_TO_DONE → INSERT (compute Δt_fwd from d_forward) → DONE
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple, Optional, Sequence
import math

# 팔레트 실측 크기 (scan_cleanup/pallet_full.obj, KS T-11 변형)
PALLET_WIDTH_M  = 1.10
PALLET_DEPTH_M  = 1.30
PALLET_HEIGHT_M = 0.12


def _to_3x3(R) -> list:
    """numpy ndarray / list-of-lists 둘 다 받기."""
    try:
        return [[float(R[i][j]) for j in range(3)] for i in range(3)]
    except Exception:
        import numpy as np
        Rn = np.asarray(R, dtype=float).reshape(3, 3)
        return Rn.tolist()


def _to_3vec(t) -> Tuple[float, float, float]:
    try:
        return float(t[0]), float(t[1]), float(t[2])
    except Exception:
        import numpy as np
        tn = np.asarray(t, dtype=float).reshape(3)
        return float(tn[0]), float(tn[1]), float(tn[2])


def pose6d_to_align_vars(R, t, anchor: str = "entry_face") -> Tuple[float, float, float]:
    """OpenCV (R, t) → (psi_pallet_deg, d_lateral_m, d_forward_m).

    입력:
        R: 3x3, pallet local axes 의 camera frame 표현 (DOPE/FoundationPose 출력).
        t: 3-vec (m), 카메라 frame 에서 본 pallet model origin = centroid 위치.
        anchor:
          "entry_face"  : (default) d_lateral/d_forward 를 entry face 중심 기준.
                          centroid + R @ (0, 0, +depth/2) 변환. 기존 perception.py 와
                          호환되어 config 의 ALIGN_DIST_M, OFF_TOL_M 그대로 사용 가능.
          "centroid"    : centroid 그대로 (t).

    Returns:
        psi_pallet_deg : ψ_pallet (deg). + 이면 팔레트 정면이 카메라 우측.
        d_lateral_m   : anchor 의 카메라 +X 좌표 (m).
        d_forward_m   : anchor 의 카메라 +Z 좌표 (m).
    """
    Rm = _to_3x3(R)
    tx, _, tz = _to_3vec(t)

    psi_rad = math.atan2(Rm[0][2], Rm[2][2])
    psi_deg = math.degrees(psi_rad)

    if anchor == "entry_face":
        # entry face = centroid + R @ (0, 0, +depth/2)
        offset = PALLET_DEPTH_M / 2.0
        ax = tx + Rm[0][2] * offset
        az = tz + Rm[2][2] * offset
        return psi_deg, float(ax), float(az)
    return psi_deg, float(tx), float(tz)


def keypoints9_to_align_vars(
    keypoints_2d: Sequence[Sequence[float]],
    camera_matrix,
    dist_coeffs=None,
    pallet_width_m: float = PALLET_WIDTH_M,
    pallet_depth_m: float = PALLET_DEPTH_M,
    pallet_height_m: float = PALLET_HEIGHT_M,
) -> Optional[Tuple[float, float, float]]:
    """DOPE 9 keypoint (image px) + 카메라 intrinsic → ALIGN 변수.

    내부에서 PnP (EPnP + iterative refine) 로 (R, t) 복원 후 centroid 기준 변환.
    centroid (keypoint 8) 는 PnP 입력에도 포함되어 추정 안정성을 높임.

    Args:
        keypoints_2d  : [(u, v)] x 9, v4 convention 순서.
                        invisible keypoint 는 (NaN, NaN) 또는 (-1, -1) 로 표기.
        camera_matrix : 3x3 K (fx, fy, cx, cy).
        dist_coeffs   : OpenCV distortion (k1, k2, p1, p2[, k3]). None=무왜곡.
        pallet_*_m    : 모델 실측 (default = pallet_full.obj 1.10×1.30×0.12).

    Returns:
        (psi_pallet_deg, d_lateral_m, d_forward_m) 또는 PnP 실패 시 None.
    """
    try:
        import numpy as np
        import cv2
    except Exception:
        return None

    # v4 convention 의 모델 좌표 (centroid 가 origin, 단위 m).
    # +X = pallet right(width), +Y = up(height), +Z = forward(depth).
    hw, hh, hd = pallet_width_m / 2.0, pallet_height_m / 2.0, pallet_depth_m / 2.0
    object_points = np.array([
        [-hw, +hh, +hd],   # 0 front-top-LEFT
        [+hw, +hh, +hd],   # 1 front-top-RIGHT
        [+hw, -hh, +hd],   # 2 front-bot-RIGHT
        [-hw, -hh, +hd],   # 3 front-bot-LEFT
        [-hw, +hh, -hd],   # 4 rear-top-LEFT
        [+hw, +hh, -hd],   # 5 rear-top-RIGHT
        [+hw, -hh, -hd],   # 6 rear-bot-RIGHT
        [-hw, -hh, -hd],   # 7 rear-bot-LEFT
        [0.0, 0.0, 0.0],   # 8 centroid
    ], dtype=np.float64)

    kps = np.array(keypoints_2d, dtype=np.float64)
    if kps.shape != (9, 2):
        return None

    # invisible keypoint 제거
    visible = ~np.isnan(kps).any(axis=1) & (kps[:, 0] >= 0) & (kps[:, 1] >= 0)
    if visible.sum() < 4:
        return None

    obj_pts = object_points[visible].reshape(-1, 1, 3)
    img_pts = kps[visible].reshape(-1, 1, 2)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    D = np.zeros(5, dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).ravel()

    flag = cv2.SOLVEPNP_EPNP if visible.sum() >= 6 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flag)
    if not ok:
        return None
    # iterative refine
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
    except Exception:
        pass

    R, _ = cv2.Rodrigues(rvec)
    return pose6d_to_align_vars(R, tvec.ravel())


def pose6d_to_align_vars_safe(R, t) -> Optional[Tuple[float, float, float]]:
    """예외 안전 wrapper. 변환 실패 시 None."""
    try:
        return pose6d_to_align_vars(R, t)
    except Exception:
        return None
