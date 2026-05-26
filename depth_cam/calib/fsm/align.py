# calib/fsm/align.py
from __future__ import annotations
from typing import Optional, List, Tuple
import time

import calib.config as cfg  # 하이퍼파라미터 안전 접근(getattr용)

from calib.config import (
    YAW_TOL_DEG, OFF_TOL_M,
    ALIGN_DIST_M, ALIGN_BAND_M,
    REL_YAW_TARGET_DEG,     # 보통 90.0
    STOP_SEC,
    USE_PIECEWISE_FWD_FIT,

    COLOR_STATUS_TRK, COLOR_ALERT, COLOR_META, COLOR_STATUS_OK,
    CMD_STABLE_THR,
)
from calib.motion_models import fwd_sec_from_offset_piecewise
from .utils import within_band, Stabilizer, SimpleTimer
from .status_helper import StatusHelper
from .commands import CommandExecutor


def _wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _fwd_sec_for_insertion(dist_z: Optional[float]) -> Optional[float]:
    """
    파렛트 포켓 삽입용 전진 시간 추정 함수.
    total_dist = dist_z + PALLET_POCKET_M
    sec = clamp( total_dist / INSERT_FWD_MPS, INS_FWD_MIN_SEC, INS_FWD_MAX_SEC )

    하이퍼파라미터(없으면 기본값 사용):
      - PALLET_POCKET_M: float = 1.0
      - INSERT_FWD_MPS : float = 0.25
      - INS_FWD_MIN_SEC: float = 0.5
      - INS_FWD_MAX_SEC: float = 10.0
    """
    if dist_z is None:
        return None
    pocket_m = float(getattr(cfg, "PALLET_POCKET_M", 0.0))
    v_mps    = float(getattr(cfg, "INSERT_FWD_MPS", 0.25))
    t_min    = float(getattr(cfg, "INS_FWD_MIN_SEC", 0.5))
    t_max    = float(getattr(cfg, "INS_FWD_MAX_SEC", 10.0))
    total = max(0.0, float(dist_z) + pocket_m)
    v = max(1e-3, v_mps)
    sec = total / v
    return max(t_min, min(t_max, sec))


class AlignMachine:
    """
    다이어그램 준수 + 포켓 삽입:
      DIST_CHECK → YAW_CHECK → OFFSET_CHECK →
        (RIGHT 체인) ALIGN_ROTATE_RIGHT(+90) → [STOP] → FORWARD_AFTER_RIGHT(FWD_SEC) → [STOP] → ALIGN_ROTATE_LEFT_90(−90) → YAW_CHECK
        (LEFT  체인) ALIGN_ROTATE_LEFT(−90)  → [STOP] → FORWARD_AFTER_LEFT(FWD_SEC)  → [STOP] → ALIGN_ROTATE_RIGHT_90(+90) → YAW_CHECK
      OFFSET_CHECK에서 |offset|≤tol & |yaw|≤tol ⇒ (STOP) ⇒ INSERT_FORWARD(포켓삽입 전진) ⇒ (STOP) ⇒ READY_TO_DONE

    규칙(요청 반영):
      - ALIGN_ROTATE_RIGHT/LEFT 및 *_90 진입 직전에 rel_yaw 기준각 초기화(재초기화).
    """

    def __init__(self, execu: CommandExecutor, status: StatusHelper):
        self.sub: str = "DIST_CHECK"
        self.execu = execu
        self.status = status

        self._stb = Stabilizer(CMD_STABLE_THR)

        self._timer = SimpleTimer()
        self._interlock_active: bool = False
        self._after_interlock_sub: Optional[str] = None

        # 90° 회전 기준각 (rel_yaw 기준)
        self._rel_yaw_ref: Optional[float] = None

        # OFFSET_CHECK 산출 전진 시간 (오프셋 보정용)
        self._fwd_sec_cached: float = 0.0
        self._fwd_deadline_ts: Optional[float] = None

        # 포켓 삽입 전진 시간
        self._insert_sec_cached: Optional[float] = None
        self._insert_deadline_ts: Optional[float] = None

        # 최근 dist_z (READY_TO_DONE 시 사용)
        self._latest_dist_z: Optional[float] = None

        # 명령 스팸 억제
        self._last_cmd: Optional[str] = None
        self._last_cmd_ts: float = 0.0

    # ----------------- 유틸 -----------------
    def _exec(self, code: str):
        now = time.time()
        if self._last_cmd == code and (now - self._last_cmd_ts) < 0.10:
            return
        self.execu.exec(code)
        self._last_cmd, self._last_cmd_ts = code, now

    def _start_interlock_then(self, next_sub: str):
        self._interlock_active = True
        self._after_interlock_sub = next_sub
        self._exec("STOP")
        self._timer.start(STOP_SEC)
        self.status.start_timed("STOP", STOP_SEC)

    def _maybe_finish_interlock(self, lines) -> bool:
        if self._interlock_active:
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append(("[ALIGN] STOP 인터록 진행", COLOR_META))
            if not self._timer.active():
                self._interlock_active = False
                if self._after_interlock_sub:
                    self.sub = self._after_interlock_sub
                    self._after_interlock_sub = None
            return True
        return False

    def reset(self, sub: str = "DIST_CHECK"):
        self.sub = sub
        self._stb.reset()
        self._interlock_active = False
        self._after_interlock_sub = None
        self._rel_yaw_ref = None
        self._fwd_deadline_ts = None
        self._insert_sec_cached = None
        self._insert_deadline_ts = None
        self._latest_dist_z = None

    # ----------------- 메인 스텝 -----------------
    def step(self,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw: Optional[float],   # 비전 yaw (YAW_CHECK용)
             ox: Optional[float],
             rel_yaw: Optional[float]  # IMU 상대 yaw(도) — 회전 체인에서 사용
             ) -> List[Tuple[str, tuple]]:

        lines: List[Tuple[str, tuple]] = []

        # 최신 dist_z 업데이트(포켓 삽입에 사용)
        self._latest_dist_z = dist_z if dist_z is not None else self._latest_dist_z

        # HUD 진행도(|yaw|, |offset_x|) 갱신; rel_yaw 진행도는 회전 상태에서 설정
        try:
            self.status.update_until_metric(yaw, ox)
        except Exception:
            pass

        yaw_ok = (yaw is not None) and (abs(yaw) <= YAW_TOL_DEG)
        band_ok = within_band(dist_z, ALIGN_DIST_M, ALIGN_BAND_M)

        # 미탐지 시: 회전/체인 중이 아닐 때만 스핀 대기
        if not det_ok and self.sub not in (
            "ALIGN_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "ALIGN_ROTATE_LEFT_90",
            "ALIGN_ROTATE_LEFT",  "FORWARD_AFTER_LEFT",  "ALIGN_ROTATE_RIGHT_90",
            "ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL",
            "INSERT_FORWARD",
        ):
            code = "ROT_LEFT" if self.execu.last_dir > 0 else "ROT_RIGHT"
            self._exec(code)
            self.status.start_timed(code, 0.0)
            lines.append(("[ALIGN] 미탐지 → 스핀 대기", COLOR_ALERT))
            return lines

        # 인터록 우선 처리
        if self._maybe_finish_interlock(lines):
            return lines

        # ---------- DIST_CHECK ----------
        if self.sub == "DIST_CHECK":
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)

            if band_ok:
                if self._stb.stable("DIST_BAND_OK"):
                    self.reset("YAW_CHECK")
                    lines.append(("[DIST_CHECK→YAW_CHECK] 밴드 OK", COLOR_META))
                else:
                    lines.append((f"[DIST_CHECK] 밴드 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines

            if (dist_z is not None) and (dist_z > ALIGN_DIST_M):
                if self._stb.stable("DIST_FWD"):
                    self.reset("ALIGN_FWD_ADJUST")
                    lines.append(("[DIST_CHECK→ALIGN_FWD_ADJUST]", COLOR_META))
                else:
                    lines.append((f"[DIST_CHECK] 전진 판정 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines

            if (dist_z is not None) and (dist_z < ALIGN_DIST_M):
                if self._stb.stable("DIST_BWD"):
                    self.reset("ALIGN_BWD_ADJUST")
                    lines.append(("[DIST_CHECK→ALIGN_BWD_ADJUST]", COLOR_META))
                else:
                    lines.append((f"[DIST_CHECK] 후진 판정 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines

            lines.append(("[DIST_CHECK] distance N/A (STOP 유지)", COLOR_ALERT))
            return lines

        # ---------- ALIGN_FWD_ADJUST ----------
        if self.sub == "ALIGN_FWD_ADJUST":
            if (dist_z is not None) and (abs(dist_z - ALIGN_DIST_M) > ALIGN_BAND_M):
                self._exec("FWD")
                self.status.start_timed("FWD", 0.0)
                lines.append(("[ALIGN_FWD_ADJUST] 전진", COLOR_STATUS_TRK))
            else:
                self._start_interlock_then("DIST_CHECK")
                lines.append(("[ALIGN_FWD_ADJUST→DIST_CHECK] (STOP 인터록)", COLOR_META))
            return lines

        # ---------- ALIGN_BWD_ADJUST ----------
        if self.sub == "ALIGN_BWD_ADJUST":
            if (dist_z is not None) and (abs(dist_z - ALIGN_DIST_M) > ALIGN_BAND_M):
                self._exec("BACK")
                self.status.start_timed("BACK", 0.0)
                lines.append(("[ALIGN_BWD_ADJUST] 후진", COLOR_STATUS_TRK))
            else:
                self._start_interlock_then("DIST_CHECK")
                lines.append(("[ALIGN_BWD_ADJUST→DIST_CHECK] (STOP 인터록)", COLOR_META))
            return lines

        # ---------- YAW_CHECK (vision yaw 기준) ----------
        if self.sub == "YAW_CHECK":
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)

            if (yaw is not None) and (abs(yaw) > YAW_TOL_DEG):
                if yaw > 0:
                    if self._stb.stable("YAW_POS"):
                        self.sub = "ROTATE_RIGHT_UNTIL_YAW_TOL"
                        self._exec("ROT_RIGHT")
                        self.status.start_until("ROT_RIGHT", "|yaw|", abs(yaw), YAW_TOL_DEG)
                        lines.append(("[YAW_CHECK→ROTATE_RIGHT_UNTIL_YAW_TOL]", COLOR_ALERT))
                    else:
                        lines.append((f"[YAW_CHECK] yaw>+tol 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                else:
                    if self._stb.stable("YAW_NEG"):
                        self.sub = "ROTATE_LEFT_UNTIL_YAW_TOL"
                        self._exec("ROT_LEFT")
                        self.status.start_until("ROT_LEFT", "|yaw|", abs(yaw), YAW_TOL_DEG)
                        lines.append(("[YAW_CHECK→ROTATE_LEFT_UNTIL_YAW_TOL]", COLOR_ALERT))
                    else:
                        lines.append((f"[YAW_CHECK] yaw<-tol 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines
            else:
                if self._stb.stable("YAW_OK"):
                    self.reset("OFFSET_CHECK")
                    lines.append(("[YAW_CHECK→OFFSET_CHECK] yaw tol 이내", COLOR_META))
                else:
                    lines.append((f"[YAW_CHECK] tol 이내 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines

        # ---------- ROTATE_*_UNTIL_YAW_TOL ----------
        if self.sub in ("ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL"):
            code = "ROT_RIGHT" if self.sub.startswith("ROTATE_RIGHT") else "ROT_LEFT"
            self._exec(code)
            if yaw is not None:
                self.status.start_until(code, "|yaw|", abs(yaw), YAW_TOL_DEG)
            if yaw_ok:
                self._start_interlock_then("OFFSET_CHECK")
                lines.append((f"[{self.sub}→OFFSET_CHECK] (STOP 인터록)", COLOR_META))
            else:
                lines.append((f"[{self.sub}] 자세 보정 중", COLOR_STATUS_TRK))
            return lines

        # ---------- OFFSET_CHECK ----------
        if self.sub == "OFFSET_CHECK":
            if ox is None:
                if self._stb.stable("OFF_NA"):
                    self.reset("DIST_CHECK")
                    lines.append(("[OFFSET_CHECK→DIST_CHECK] offset N/A", COLOR_META))
                else:
                    lines.append((f"[OFFSET_CHECK] N/A 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                return lines

            if abs(ox) > OFF_TOL_M:
                # (기존) 오프셋 기반 회전+전진 체인
                self._fwd_sec_cached = fwd_sec_from_offset_piecewise(ox) if USE_PIECEWISE_FWD_FIT else 2.0
                self._fwd_sec_cached = max(0.1, float(self._fwd_sec_cached))

                if ox > 0:
                    if self._stb.stable("OFF_RIGHT"):
                        # RIGHT 90 체인 시작 직전 기준각 초기화
                        self._rel_yaw_ref = rel_yaw if rel_yaw is not None else None
                        self.sub = "ALIGN_ROTATE_RIGHT"
                        self._exec("ROT_RIGHT")
                        if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                            delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                            self.status.start_until("ROT_RIGHT", "rel_yaw", delta, +REL_YAW_TARGET_DEG)
                        self._fwd_deadline_ts = None
                        lines.append((f"[OFFSET_CHECK→ALIGN_ROTATE_RIGHT] (FWD_SEC={self._fwd_sec_cached:.2f}s)", COLOR_META))
                    else:
                        lines.append((f"[OFFSET_CHECK] offset>+tol 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                    return lines
                else:
                    if self._stb.stable("OFF_LEFT"):
                        # LEFT 90 체인 시작 직전 기준각 초기화
                        self._rel_yaw_ref = rel_yaw if rel_yaw is not None else None
                        self.sub = "ALIGN_ROTATE_LEFT"
                        self._exec("ROT_LEFT")
                        if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                            delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                            self.status.start_until("ROT_LEFT", "rel_yaw", delta, -REL_YAW_TARGET_DEG)
                        self._fwd_deadline_ts = None
                        lines.append((f"[OFFSET_CHECK→ALIGN_ROTATE_LEFT] (FWD_SEC={self._fwd_sec_cached:.2f}s)", COLOR_META))
                    else:
                        lines.append((f"[OFFSET_CHECK] offset<-tol 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                    return lines
            else:
                # |offset| ≤ tol
                if (yaw is not None) and (abs(yaw) > YAW_TOL_DEG):
                    if self._stb.stable("OFF_OK_YAW_NOK"):
                        self.reset("DIST_CHECK")
                        lines.append(("[OFFSET_CHECK→DIST_CHECK] yaw 재보정", COLOR_META))
                    else:
                        lines.append((f"[OFFSET_CHECK] yaw>tol 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                    return lines
                else:
                    # ★ 포켓 삽입 준비: STOP 인터록 후 INSERT_FORWARD로 전이
                    if self._stb.stable("READY_TO_DONE"):
                        # dist_z + 포켓깊이 → 전진 시간 추정
                        self._insert_sec_cached = _fwd_sec_for_insertion(self._latest_dist_z)
                        if (self._insert_sec_cached is not None) and (self._insert_sec_cached > 0.0):
                            self._start_interlock_then("INSERT_FORWARD")
                            lines.append((f"[OFFSET_CHECK→INSERT_FORWARD] pocket={getattr(cfg,'PALLET_POCKET_M',1.0):.2f}m, "
                                          f"FWD_SEC={self._insert_sec_cached:.2f}s (STOP 인터록)", COLOR_META))
                        else:
                            # dist 정보 없음 → 삽입 생략하고 READY_TO_DONE
                            self.sub = "READY_TO_DONE"
                            self._exec("STOP")
                            self.status.start_timed("STOP", 0.0)
                            lines.append(("[OFFSET_CHECK→READY_TO_DONE] dist N/A: 삽입 스킵", COLOR_STATUS_TRK))
                    else:
                        lines.append((f"[OFFSET_CHECK] 완료 대기 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
                    return lines

        # ---------- RIGHT branch ----------
        if self.sub == "ALIGN_ROTATE_RIGHT":
            self._exec("ROT_RIGHT")
            if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                self.status.start_until("ROT_RIGHT", "rel_yaw", delta, +REL_YAW_TARGET_DEG)
                if delta >= +REL_YAW_TARGET_DEG:
                    self._start_interlock_then("FORWARD_AFTER_RIGHT")
                    lines.append(("[ALIGN_ROTATE_RIGHT→FORWARD_AFTER_RIGHT] (STOP 인터록)", COLOR_META))
            else:
                lines.append(("[ALIGN_ROTATE_RIGHT] rel_yaw N/A", COLOR_ALERT))
            return lines

        if self.sub == "FORWARD_AFTER_RIGHT":
            self._exec("FWD")
            if self._fwd_deadline_ts is None:
                self._fwd_deadline_ts = time.time() + self._fwd_sec_cached
            remain = max(0.0, self._fwd_deadline_ts - time.time())
            self.status.start_timed("FWD", remain)
            lines.append((f"[FORWARD_AFTER_RIGHT] 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))
            if remain <= 0.0:
                # ALIGN_ROTATE_LEFT_90 진입 직전 기준각 재초기화
                self._rel_yaw_ref = rel_yaw if rel_yaw is not None else None
                self._fwd_deadline_ts = None
                self._start_interlock_then("ALIGN_ROTATE_LEFT_90")
                lines.append(("[FORWARD_AFTER_RIGHT→ALIGN_ROTATE_LEFT_90] (STOP 인터록)", COLOR_META))
            return lines

        if self.sub == "ALIGN_ROTATE_LEFT_90":
            self._exec("ROT_LEFT")
            if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                self.status.start_until("ROT_LEFT", "rel_yaw", delta, -REL_YAW_TARGET_DEG)
                if delta <= -REL_YAW_TARGET_DEG:
                    self._rel_yaw_ref = None
                    self._start_interlock_then("YAW_CHECK")
                    lines.append(("[ALIGN_ROTATE_LEFT_90→YAW_CHECK] (STOP 인터록)", COLOR_META))
            else:
                lines.append(("[ALIGN_ROTATE_LEFT_90] rel_yaw N/A", COLOR_ALERT))
            return lines

        # ---------- LEFT branch ----------
        if self.sub == "ALIGN_ROTATE_LEFT":
            self._exec("ROT_LEFT")
            if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                self.status.start_until("ROT_LEFT", "rel_yaw", delta, -REL_YAW_TARGET_DEG)
                if delta <= -REL_YAW_TARGET_DEG:
                    self._start_interlock_then("FORWARD_AFTER_LEFT")
                    lines.append(("[ALIGN_ROTATE_LEFT→FORWARD_AFTER_LEFT] (STOP 인터록)", COLOR_META))
            else:
                lines.append(("[ALIGN_ROTATE_LEFT] rel_yaw N/A", COLOR_ALERT))
            return lines

        if self.sub == "FORWARD_AFTER_LEFT":
            self._exec("FWD")
            if self._fwd_deadline_ts is None:
                self._fwd_deadline_ts = time.time() + self._fwd_sec_cached
            remain = max(0.0, self._fwd_deadline_ts - time.time())
            self.status.start_timed("FWD", remain)
            lines.append((f"[FORWARD_AFTER_LEFT] 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))
            if remain <= 0.0:
                # ALIGN_ROTATE_RIGHT_90 진입 직전 기준각 재초기화
                self._rel_yaw_ref = rel_yaw if rel_yaw is not None else None
                self._fwd_deadline_ts = None
                self._start_interlock_then("ALIGN_ROTATE_RIGHT_90")
                lines.append(("[FORWARD_AFTER_LEFT→ALIGN_ROTATE_RIGHT_90] (STOP 인터록)", COLOR_META))
            return lines

        if self.sub == "ALIGN_ROTATE_RIGHT_90":
            self._exec("ROT_RIGHT")
            if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                delta = _wrap_to_180(rel_yaw - self._rel_yaw_ref)
                self.status.start_until("ROT_RIGHT", "rel_yaw", delta, +REL_YAW_TARGET_DEG)
                if delta >= +REL_YAW_TARGET_DEG:
                    self._rel_yaw_ref = None
                    self._start_interlock_then("YAW_CHECK")
                    lines.append(("[ALIGN_ROTATE_RIGHT_90→YAW_CHECK] (STOP 인터록)", COLOR_META))
            else:
                lines.append(("[ALIGN_ROTATE_RIGHT_90] rel_yaw N/A", COLOR_ALERT))
            return lines

        # ---------- INSERT_FORWARD (포켓 삽입 전진) ----------
        if self.sub == "INSERT_FORWARD":
            self._exec("FWD")
            # 최초 진입 시 데드라인 설정
            if self._insert_deadline_ts is None:
                # 안전: 캐시가 None이면 즉시 READY_TO_DONE으로 스킵
                if (self._insert_sec_cached is None) or (self._insert_sec_cached <= 0.0):
                    self._start_interlock_then("READY_TO_DONE")
                    lines.append(("[INSERT_FORWARD→READY_TO_DONE] 삽입시간 N/A: 스킵 (STOP 인터록)", COLOR_META))
                    return lines
                self._insert_deadline_ts = time.time() + float(self._insert_sec_cached)

            remain = max(0.0, self._insert_deadline_ts - time.time())
            self.status.start_timed("FWD", remain)
            lines.append((f"[INSERT_FORWARD] 포켓 삽입 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))

            if remain <= 0.0:
                # 완료 → READY_TO_DONE으로 복귀
                self._insert_deadline_ts = None
                self._start_interlock_then("READY_TO_DONE")
                lines.append(("[INSERT_FORWARD] 완료 → READY_TO_DONE (STOP 인터록)", COLOR_META))
            return lines

        # ---------- READY_TO_DONE ----------
        if self.sub == "READY_TO_DONE":
            # 상위 FSM이 ALIGN→DONE 전이시킬 상태. 정지 유지.
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append(("[READY_TO_DONE] 정렬 및 포켓 삽입 완료(상위로 DONE 전이)", COLOR_STATUS_OK))
            return lines

        # fallback
        self._exec("STOP")
        self.status.start_timed("STOP", 0.0)
        lines.append((f"[{self.sub}] 대기", COLOR_META))
        return lines
