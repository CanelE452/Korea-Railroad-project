# calib/truck/truck_adapter.py
# =============================================================================
# SMOKE Detection → FSM 상태변수 어댑터 (다이어그램 T3~T6).
#
# pose6d_adapter.py (파렛트) 와 동일 패턴/부호 규약:
#   ψ = 0°       ⇔ 지게차가 목표면(트럭 적재면 옆면)을 정면으로 봄
#   d_lateral    ⇔ 부호 반전된 camera +X (파렛트 어댑터의 실차 버그 수정 계승)
#   d_forward    ⇔ 목표점의 전방(+Z) 거리
#
# SMOKE(KITTI) 카메라 좌표: X우 / Y하 / Z전방 (OpenCV 와 동일).
# Detection:
#   location_xyz   : 3D 박스 '바닥 중심' (카메라 frame, m)
#   rotation_y     : Y축(연직) 회전 (rad)
#   dimensions_hwl : (height, width, length) (m)
#
# 적재면 옆면부 중앙 (T4): 박스의 ±width/2 면 중 카메라에 가까운 쪽 면의
# 높이 중간점. 높이축 제거(T5) 후 (forward, lateral, yaw) 를 계산(T6).
#
# Camera2 는 포크 장착 → extrinsic CAM2_TO_FORK_T/RPY (config).
# ⚠ 포크 높이가 변하면 extrinsic 도 변함 — 트럭 pose 는 포크가 알려진 초기
#   높이일 때(T14 상승 전) 만 신뢰. TruckMachine 이 이를 강제한다.
# =============================================================================
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from calib.pose6d_adapter import apply_cam_to_fork


def _wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _rot_y(yaw_rad: float):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # KITTI: R_y = [[c,0,s],[0,1,0],[-s,0,c]]
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _matvec(R, v):
    return [sum(R[i][k] * v[k] for k in range(3)) for i in range(3)]


def truck_detection_to_align_vars(
    location_xyz: Sequence[float],
    rotation_y: float,
    dimensions_hwl: Sequence[float],
    cam_to_fork_t: Optional[Sequence[float]] = None,
    cam_to_fork_rpy_deg: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """SMOKE Detection → (psi_truck_deg, d_lateral_m, d_forward_m).

    Returns:
        psi_truck_deg : 적재면 옆면 법선 기준 yaw (deg). 0 = 정면 대향.
        d_lateral_m   : 옆면 중앙의 측방 오프셋 (부호 반전 규약, 파렛트와 동일).
        d_forward_m   : 옆면 중앙까지 전방 거리 (m).
    """
    h, w, l = (float(v) for v in dimensions_hwl)
    loc = [float(v) for v in location_xyz]
    R = _rot_y(float(rotation_y))

    # 두 옆면(±width/2, 박스 z축) 중앙 — location 은 바닥 중심이므로 높이 중간은 -h/2 (Y하 방향이 +)
    candidates = []
    for s in (+1.0, -1.0):
        off_box = [0.0, -h / 2.0, s * w / 2.0]
        center = [loc[i] + v for i, v in enumerate(_matvec(R, off_box))]
        normal = _matvec(R, [0.0, 0.0, s])   # 옆면 바깥쪽 법선 (카메라 frame)
        dist2 = sum(c * c for c in center)
        candidates.append((dist2, center, normal))

    # 카메라(지게차)에 가까운 옆면 채택
    _, face_center, face_normal = min(candidates, key=lambda c: c[0])

    # ---- Camera2 → 포크 frame extrinsic (기본 항등 — 회귀 없음) ----
    R_f, t_f = apply_cam_to_fork(R, face_center, cam_to_fork_t, cam_to_fork_rpy_deg)
    # 법선도 회전만 적용: n_f = Rcf @ n. apply_cam_to_fork 의 R_f = Rcf@R 이므로
    # 박스 frame 법선 [0,0,±1] 을 R_f 로 돌리면 동일 결과.
    s_face = +1.0 if face_normal == _matvec(R, [0.0, 0.0, 1.0]) else -1.0
    n_f = _matvec(R_f, [0.0, 0.0, s_face])

    # ---- T5: 높이축 제거 (BEV) → T6: 상태변수 ----
    # ψ: 법선이 카메라를 향할 때(정면 대향) 0° — 파렛트 어댑터의 +180 wrap 과 동일 규약
    psi_deg = _wrap_to_180(math.degrees(math.atan2(n_f[0], n_f[2])) + 180.0)
    d_lateral = -float(t_f[0])   # 부호 반전 (파렛트 어댑터 실차 버그 수정 계승)
    d_forward = float(t_f[2])
    return psi_deg, d_lateral, d_forward


class TruckStateGate:
    """시간적 게이팅 — 연속 N 프레임 유효 검출 시에만 상태 확정 (파렛트 confirm 패턴).

    사용:
        gate = TruckStateGate(score_thr, confirm_n)
        state = gate.update(det)   # det = None 또는 (score, loc, rot_y, dims)
        # state = None 또는 (psi, d_lat, d_fwd)
    """

    def __init__(self, score_thr: float, confirm_n: int,
                 cam_to_fork_t=None, cam_to_fork_rpy_deg=None):
        self.score_thr = float(score_thr)
        self.confirm_n = int(confirm_n)
        self.cam_to_fork_t = cam_to_fork_t
        self.cam_to_fork_rpy_deg = cam_to_fork_rpy_deg
        self._consecutive = 0
        self._last_state: Optional[Tuple[float, float, float]] = None

    def reset(self):
        self._consecutive = 0
        self._last_state = None

    def update(self, detection) -> Optional[Tuple[float, float, float]]:
        """detection: None 또는 (score, location_xyz, rotation_y, dimensions_hwl)."""
        if detection is None:
            self._consecutive = 0
            self._last_state = None
            return None
        score, loc, rot_y, dims = detection
        if float(score) < self.score_thr:
            self._consecutive = 0
            self._last_state = None
            return None
        self._consecutive += 1
        self._last_state = truck_detection_to_align_vars(
            loc, rot_y, dims, self.cam_to_fork_t, self.cam_to_fork_rpy_deg,
        )
        if self._consecutive >= self.confirm_n:
            return self._last_state
        return None
