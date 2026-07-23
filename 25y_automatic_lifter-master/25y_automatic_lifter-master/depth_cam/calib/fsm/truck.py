# calib/fsm/truck.py
# =============================================================================
# TruckMachine — Phase B (트럭 적재) FSM. truck_loading/다이어그램.txt T0~T27.
#
# align.py 와 동일한 snapshot 원칙: "한 번 보고 눈을 감고 동작".
#   TRUCK_CHECK 에서 (ψ_truck, d_lateral, d_forward) 를 1회 캡처해 Manhattan
#   접근 체인 전체를 사전 계산하고, cmd 상태 진행 중에는 perception 을 무시
#   (회전 종료는 IMU rel_yaw, 전진 종료는 시간, 모서리/안착은 레이저).
#
# 상태 흐름:
#   WAIT_SENSORS (T0~TD0)  레이저 L/R 유효 대기
#   TRUCK_CHECK  (T1~T6)   트럭 상태 확정 (시간적 게이팅 후 snapshot)
#   LAT_ROT / LAT_FWD / LAT_ROT_BACK  (T7~T10, T12)  측면 이동 체인 (|d_lat|>tol)
#   FACE_ROT     (T11)     잔여 ψ 보정 (|ψ|>tol)
#   APPROACH_FWD (T13)     전진: d_forward - SAFETY_MARGIN
#   RAISE_FORK   (T14/TD4) 포크 상승 → TRUCK_HEIGHT (시간 기반)
#   EDGE_SEARCH  (T15~T20) 저속 전진 + 좌/우 레이저 동시 급감 → 정지
#   LOWER        (T21~T24) 저속 하강 + laser L/R < RELEASE_THRESHOLD → 정지
#   RELEASE      (T25)     안착 확인 (리프트 해제)
#   BACKOUT      (T26/TD7) 후진 FORK_LENGTH + SAFETY_MARGIN
#   DONE         (T27)
#   FAULT                  레이저 stale / 탐색 타임아웃 → STOP 유지
#
# 모든 cmd 전환 사이 STOP 인터록 (STOP_SEC) — 파렛트 FSM 과 동일한 안전 규약.
# =============================================================================
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import calib.config as cfg
from calib.config import (
    STOP_SEC,
    TRUCK_YAW_TOL_DEG, TRUCK_OFF_TOL_M,
    TRUCK_HEIGHT_M, TRUCK_SAFETY_MARGIN_M, BACKOUT_DIST_M,
    RELEASE_THRESHOLD_M, FORK_LENGTH_M,
    LASER_DROP_THRESH_M, LASER_DROP_SYNC_S, LASER_CONFIRM_N,
    SLOW_FWD_MPS, LOWER_SPEED_MPS,
    LATERAL_BACK_YAW_DEG, YAW_TURN_MIN_DEG,
    COLOR_META, COLOR_ALERT, COLOR_STATUS_OK, COLOR_STATUS_TRK,
)
from calib.motion_models import fwd_sec_from_offset_piecewise
from calib.truck.lasers import EdgeDropDetector, ReleaseDetector
from .mission import lift_sec_for_height
from .commands import CommandExecutor


def _wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _drive_sec_for_distance(dist_m: float) -> float:
    """전/후진 시간 모델 — align.py 와 동일하게 piecewise fit 재사용."""
    return max(0.1, float(fwd_sec_from_offset_piecewise(abs(float(dist_m)))))


class TruckMachine:
    """Phase B FSM. step() 은 매 tick 호출.

    Args:
        execu  : CommandExecutor (mission 과 공유 권장)
        now_fn : 시간 함수 주입 (테스트용)
    """

    CMD_STATES = frozenset({
        "LAT_ROT", "LAT_FWD", "LAT_ROT_BACK",
        "FACE_ROT", "APPROACH_FWD",
        "RAISE_FORK", "EDGE_SEARCH", "LOWER", "BACKOUT",
    })

    def __init__(self, execu: Optional[CommandExecutor] = None, now_fn=time.time):
        self.execu = execu or CommandExecutor()
        self.now = now_fn
        self.state: str = "WAIT_SENSORS"

        # 인터록
        self._interlock_until: float = 0.0
        self._after_interlock: Optional[str] = None

        # Manhattan 접근 plan (TRUCK_CHECK snapshot 에서 계산)
        self._plan: Optional[dict] = None

        # cmd 진행 상태
        self._deadline: Optional[float] = None
        self._rel_yaw_ref: Optional[float] = None
        self._rot_dir: int = 0          # +1=ROT_RIGHT, -1=ROT_LEFT
        self._rot_target_deg: float = 0.0

        # 레이저 감지기 (순수 로직)
        self.edge_det = EdgeDropDetector(
            LASER_DROP_THRESH_M, LASER_DROP_SYNC_S, LASER_CONFIRM_N)
        self.release_det = ReleaseDetector(RELEASE_THRESHOLD_M, LASER_CONFIRM_N)

        # 안전 타임아웃 (open-loop 폭주 방지)
        self._edge_timeout_s = (
            (TRUCK_SAFETY_MARGIN_M + FORK_LENGTH_M + 0.5) / max(1e-3, SLOW_FWD_MPS)
        ) * 1.5
        self._lower_timeout_s = (TRUCK_HEIGHT_M / max(1e-3, LOWER_SPEED_MPS)) * 1.5

        self.fault_reason: str = ""

    # ------------------------------------------------------------------ utils
    def _interlock_then(self, next_state: str):
        """STOP 인터록 시작 후 next_state 진입 (align.py 규약)."""
        self.execu.exec("STOP")
        self.execu.reset_last()   # 인터록 후 동일 명령 재송신 허용
        self._interlock_until = self.now() + STOP_SEC
        self._after_interlock = next_state
        self._deadline = None

    def _in_interlock(self, lines) -> bool:
        if self._after_interlock is None:
            return False
        if self.now() < self._interlock_until:
            lines.append(("[PHASE_B] STOP 인터록", COLOR_META))
            return True
        self.state = self._after_interlock
        self._after_interlock = None
        return False

    def _fault(self, reason: str, lines):
        self.fault_reason = reason
        self.execu.exec("STOP")
        self.state = "FAULT"
        lines.append((f"[PHASE_B→FAULT] {reason}", COLOR_ALERT))

    def _rel_delta(self, rel_yaw: Optional[float]) -> Optional[float]:
        if rel_yaw is None or self._rel_yaw_ref is None:
            return None
        return _wrap_to_180(float(rel_yaw) - self._rel_yaw_ref)

    def _start_rotation(self, direction: int, target_deg: float, rel_yaw: Optional[float]):
        """direction: +1=ROT_RIGHT, -1=ROT_LEFT. IMU rel_yaw 기준 회전 시작."""
        self._rot_dir = direction
        self._rot_target_deg = max(YAW_TURN_MIN_DEG, abs(float(target_deg)))
        self._rel_yaw_ref = float(rel_yaw) if rel_yaw is not None else None
        self.execu.exec("ROT_RIGHT" if direction > 0 else "ROT_LEFT")

    def _rotation_done(self, rel_yaw: Optional[float]) -> Optional[bool]:
        """회전 종료 판정. None = rel_yaw 불가."""
        d = self._rel_delta(rel_yaw)
        if d is None:
            return None
        if self._rot_dir > 0:
            return d >= +self._rot_target_deg
        return d <= -self._rot_target_deg

    # ------------------------------------------------------------------ step
    def step(self,
             truck_state: Optional[Tuple[float, float, float]] = None,
             rel_yaw: Optional[float] = None,
             laser_l: Optional[float] = None,
             laser_r: Optional[float] = None,
             ) -> List[Tuple[str, tuple]]:
        """1 tick.

        Args:
            truck_state : 게이팅 통과한 (ψ_truck_deg, d_lateral_m, d_forward_m)
                          또는 None (미확정) — TruckStateGate.update() 출력.
            rel_yaw     : IMU 상대 yaw (deg)
            laser_l/r   : 좌/우 레이저 거리 (m) 또는 None (stale)
        """
        lines: List[Tuple[str, tuple]] = []

        if self._in_interlock(lines):
            return lines

        # ------------------------- WAIT_SENSORS (T0/TD0) -------------------------
        if self.state == "WAIT_SENSORS":
            self.execu.exec("STOP")
            if laser_l is not None and laser_r is not None:
                self.state = "TRUCK_CHECK"
                lines.append(("[WAIT_SENSORS→TRUCK_CHECK] 레이저 L/R 유효", COLOR_META))
            else:
                lines.append(("[WAIT_SENSORS] 레이저 대기 "
                              f"(L={'ok' if laser_l is not None else '-'} "
                              f"R={'ok' if laser_r is not None else '-'})", COLOR_STATUS_TRK))
            return lines

        # ------------------------- TRUCK_CHECK (T1~T6) -------------------------
        if self.state == "TRUCK_CHECK":
            self.execu.exec("STOP")
            if truck_state is None:
                lines.append(("[TRUCK_CHECK] 트럭 pose 게이팅 대기", COLOR_STATUS_TRK))
                return lines

            psi, d_lat, d_fwd = (float(v) for v in truck_state)
            approach = max(0.0, d_fwd - TRUCK_SAFETY_MARGIN_M)
            self._plan = {
                "psi": psi, "d_lat": d_lat, "d_fwd": d_fwd,
                "lat_needed": abs(d_lat) > TRUCK_OFF_TOL_M,
                "psi_needed": abs(psi) > TRUCK_YAW_TOL_DEG,
                # d_lat<0 → 우측 보정(ROT_RIGHT 먼저) / d_lat>0 → 좌측 (align 규약)
                "lat_dir": +1 if d_lat < 0 else -1,
                "lat_fwd_sec": _drive_sec_for_distance(d_lat),
                "approach_sec": _drive_sec_for_distance(approach),
                "approach_m": approach,
            }
            lines.append((f"[TRUCK_CHECK] snapshot ψ={psi:+.1f}° "
                          f"d_lat={d_lat:+.2f}m d_fwd={d_fwd:.2f}m", COLOR_META))

            if self._plan["lat_needed"]:
                self._start_rotation(self._plan["lat_dir"], LATERAL_BACK_YAW_DEG, rel_yaw)
                self.state = "LAT_ROT"
                lines.append((f"[TRUCK_CHECK→LAT_ROT] 측면 이동 체인 "
                              f"({'우' if self._plan['lat_dir']>0 else '좌'} 90°)", COLOR_META))
            elif self._plan["psi_needed"]:
                self._start_rotation(+1 if psi > 0 else -1, abs(psi), rel_yaw)
                self.state = "FACE_ROT"
                lines.append((f"[TRUCK_CHECK→FACE_ROT] ψ 보정 {psi:+.1f}°", COLOR_META))
            else:
                self.state = "APPROACH_FWD"
                lines.append(("[TRUCK_CHECK→APPROACH_FWD] 정렬 OK", COLOR_META))
            return lines

        # ------------------------- LAT_ROT (T9) -------------------------
        if self.state == "LAT_ROT":
            self.execu.exec("ROT_RIGHT" if self._rot_dir > 0 else "ROT_LEFT")
            done = self._rotation_done(rel_yaw)
            if done is None:
                lines.append(("[LAT_ROT] rel_yaw N/A", COLOR_ALERT))
            elif done:
                self._interlock_then("LAT_FWD")
                lines.append(("[LAT_ROT→LAT_FWD] (STOP)", COLOR_META))
            else:
                d = self._rel_delta(rel_yaw)
                lines.append((f"[LAT_ROT] {d:+.1f}°/{self._rot_target_deg * self._rot_dir:+.1f}°",
                              COLOR_STATUS_TRK))
            return lines

        # ------------------------- LAT_FWD (T10) -------------------------
        if self.state == "LAT_FWD":
            self.execu.exec("FWD")
            if self._deadline is None:
                self._deadline = self.now() + self._plan["lat_fwd_sec"]
            remain = self._deadline - self.now()
            if remain <= 0.0:
                # 복귀 회전 — 반대 방향 90°
                self._interlock_then("LAT_ROT_BACK")
                lines.append(("[LAT_FWD→LAT_ROT_BACK] (STOP)", COLOR_META))
            else:
                lines.append((f"[LAT_FWD] 측면 이동 ({remain:.1f}s)", COLOR_STATUS_TRK))
            return lines

        # ------------------------- LAT_ROT_BACK (T12) -------------------------
        if self.state == "LAT_ROT_BACK":
            if self._rel_yaw_ref is None or self._rot_dir == self._plan["lat_dir"]:
                # 인터록 직후 첫 진입: 반대 방향 회전 시작
                self._start_rotation(-self._plan["lat_dir"], LATERAL_BACK_YAW_DEG, rel_yaw)
            self.execu.exec("ROT_RIGHT" if self._rot_dir > 0 else "ROT_LEFT")
            done = self._rotation_done(rel_yaw)
            if done is None:
                lines.append(("[LAT_ROT_BACK] rel_yaw N/A", COLOR_ALERT))
            elif done:
                if self._plan["psi_needed"]:
                    psi = self._plan["psi"]
                    self._interlock_then("FACE_ROT")
                    # FACE_ROT 진입 시 회전 파라미터 재설정 필요 표식
                    self._rot_dir = 0
                    lines.append(("[LAT_ROT_BACK→FACE_ROT] (STOP)", COLOR_META))
                else:
                    self._interlock_then("APPROACH_FWD")
                    lines.append(("[LAT_ROT_BACK→APPROACH_FWD] (STOP)", COLOR_META))
            else:
                lines.append(("[LAT_ROT_BACK] 복귀 회전 중", COLOR_STATUS_TRK))
            return lines

        # ------------------------- FACE_ROT (T11) -------------------------
        if self.state == "FACE_ROT":
            if self._rot_dir == 0:
                psi = self._plan["psi"]
                self._start_rotation(+1 if psi > 0 else -1, abs(psi), rel_yaw)
            self.execu.exec("ROT_RIGHT" if self._rot_dir > 0 else "ROT_LEFT")
            done = self._rotation_done(rel_yaw)
            if done is None:
                lines.append(("[FACE_ROT] rel_yaw N/A", COLOR_ALERT))
            elif done:
                self._interlock_then("APPROACH_FWD")
                lines.append(("[FACE_ROT→APPROACH_FWD] (STOP)", COLOR_META))
            else:
                lines.append(("[FACE_ROT] ψ 보정 중", COLOR_STATUS_TRK))
            return lines

        # ------------------------- APPROACH_FWD (T13) -------------------------
        if self.state == "APPROACH_FWD":
            if self._plan["approach_m"] <= 0.01:
                self._interlock_then("RAISE_FORK")
                lines.append(("[APPROACH_FWD→RAISE_FORK] 이미 안전점", COLOR_META))
                return lines
            self.execu.exec("FWD")
            if self._deadline is None:
                self._deadline = self.now() + self._plan["approach_sec"]
            remain = self._deadline - self.now()
            if remain <= 0.0:
                self._interlock_then("RAISE_FORK")
                lines.append(("[APPROACH_FWD→RAISE_FORK] 안전점 도달 (STOP)", COLOR_META))
            else:
                lines.append((f"[APPROACH_FWD] 전진 ({remain:.1f}s, "
                              f"{self._plan['approach_m']:.2f}m)", COLOR_STATUS_TRK))
            return lines

        # ------------------------- RAISE_FORK (T14/TD4) -------------------------
        if self.state == "RAISE_FORK":
            if self._deadline is None:
                self.execu.exec("LIFT_UP")
                self._deadline = self.now() + lift_sec_for_height(TRUCK_HEIGHT_M)
            remain = self._deadline - self.now()
            if remain <= 0.0:
                self.execu.exec("LIFT_STOP")
                self.edge_det.reset()
                self._interlock_then("EDGE_SEARCH")
                lines.append(("[RAISE_FORK→EDGE_SEARCH] 포크 1.50m (STOP)", COLOR_META))
            else:
                lines.append((f"[RAISE_FORK] 상승 중 ({remain:.1f}s)", COLOR_STATUS_TRK))
            return lines

        # ------------------------- EDGE_SEARCH (T15~T20) -------------------------
        if self.state == "EDGE_SEARCH":
            if self._deadline is None:
                self._deadline = self.now() + self._edge_timeout_s   # 안전 타임아웃
            if self.now() > self._deadline:
                self._fault("EDGE_SEARCH 타임아웃 (적재면 미감지)", lines)
                return lines

            verdict = self.edge_det.update(self.now(), laser_l, laser_r)
            if verdict == EdgeDropDetector.FAULT:
                self._fault("레이저 stale (EDGE_SEARCH)", lines)
                return lines
            if verdict == EdgeDropDetector.EDGE:
                self.release_det.reset()
                self._interlock_then("LOWER")
                lines.append(("[EDGE_SEARCH→LOWER] 적재면 모서리 감지! (STOP)", COLOR_STATUS_OK))
                return lines

            self.execu.exec("FWD_SLOW")
            lines.append((f"[EDGE_SEARCH] 저속 전진 — L={laser_l:.2f} R={laser_r:.2f}",
                          COLOR_STATUS_TRK))
            return lines

        # ------------------------- LOWER (T21~T24/TD6) -------------------------
        if self.state == "LOWER":
            if self._deadline is None:
                self.execu.exec("LIFT_DOWN")
                self._deadline = self.now() + self._lower_timeout_s
            if self.now() > self._deadline:
                self.execu.exec("LIFT_STOP")
                self._fault("LOWER 타임아웃 (안착 미확인)", lines)
                return lines

            if self.release_det.update(laser_l, laser_r):
                self.execu.exec("LIFT_STOP")
                self._interlock_then("RELEASE")
                lines.append(("[LOWER→RELEASE] 안착 감지 "
                              f"(L={laser_l:.2f} R={laser_r:.2f} < {RELEASE_THRESHOLD_M})",
                              COLOR_STATUS_OK))
            else:
                l_txt = "-" if laser_l is None else f"{laser_l:.2f}"
                r_txt = "-" if laser_r is None else f"{laser_r:.2f}"
                lines.append((f"[LOWER] 하강 중 — L={l_txt} R={r_txt}", COLOR_STATUS_TRK))
            return lines

        # ------------------------- RELEASE (T25) -------------------------
        if self.state == "RELEASE":
            self._deadline = None
            self._interlock_then("BACKOUT")
            lines.append(("[RELEASE→BACKOUT] 팔레트 안착 완료", COLOR_STATUS_OK))
            return lines

        # ------------------------- BACKOUT (T26/TD7) -------------------------
        if self.state == "BACKOUT":
            self.execu.exec("BACK")
            if self._deadline is None:
                self._deadline = self.now() + _drive_sec_for_distance(BACKOUT_DIST_M)
            remain = self._deadline - self.now()
            if remain <= 0.0:
                self.execu.exec("STOP")
                self.state = "DONE"
                lines.append((f"[BACKOUT→DONE] 후진 {BACKOUT_DIST_M:.1f}m 완료 (T27)",
                              COLOR_STATUS_OK))
            else:
                lines.append((f"[BACKOUT] 후진 이탈 ({remain:.1f}s)", COLOR_STATUS_TRK))
            return lines

        # ------------------------- DONE / FAULT -------------------------
        if self.state == "DONE":
            self.execu.exec("STOP")
            lines.append(("[PHASE_B DONE] 트럭 적재 완료 유지", COLOR_STATUS_OK))
            return lines

        if self.state == "FAULT":
            self.execu.exec("STOP")
            lines.append((f"[FAULT] {self.fault_reason} — STOP 유지 (수동 개입 필요)",
                          COLOR_ALERT))
            return lines

        # fallback
        self.execu.exec("STOP")
        lines.append((f"[{self.state}] 알 수 없는 상태 — STOP", COLOR_ALERT))
        return lines
