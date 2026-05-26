# calib/fsm/top.py
# -----------------------------------------------------------------------------
# Top-level FSM for pallet-front alignment using RGB-D perception.
# 상태 요약:
#   SEARCH → DETECTED → (ALIGN | RECOVER) → CHECK → (ALIGN | DONE)
# - SEARCH: 탐지 전 대기(요구사항에 따라 이제 STOP을 내리며 정지 대기)
# - DETECTED: 탐지 성공 시, 폭/품질 기준으로 ALIGN 또는 RECOVER 선택
# - ALIGN: 세부 정렬 하위 상태기(거리/야우/오프셋 보정)
# - RECOVER: 시야 확보/전면 마스크 폭 확장 동작 후 HOLD
# - CHECK: 최종 허용 오차(yaw/offset) 안정화 검사
# - DONE: 정렬 완료 상태 유지(정지)
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Tuple
import time

from calib.config import (
    WIDTH_MIN_FULL,                      # 전면이 충분히 보이는 최소 폭 기준
    COLOR_META, COLOR_ALERT, COLOR_STATUS_OK, COLOR_STATUS_TRK,  # HUD 색상
    YAW_TOL_DEG, OFF_TOL_M,              # 최종 허용 오차
    ALIGN_DIST_M, ALIGN_BAND_M,          # 거리 밴드(목표 거리 ± 밴드)
    CMD_STABLE_THR                       # 안정화에 필요한 연속 프레임 수
)
from .commands import CommandExecutor      # CAN 제어 명령 실행기
from .status_helper import StatusHelper    # HUD용 현재 명령/진행도 관리
from .utils import Stabilizer, within_band # 안정화 카운터, 밴드 체크
from .align import AlignMachine            # 정렬 하위 상태기
from .recover import RecoverMachine        # 리커버(시야확보) 하위 상태기


class CalibrationFSM:
    """
    Top-level:
      SEARCH → DETECTED → (ALIGN | RECOVER) → CHECK → (ALIGN | DONE)
    - SEARCH: 미탐지 시 정지 상태로 대기(요구사항 반영)
    - DETECTED: 탐지 성공 시 다음 단계 분기
    - ALIGN: 거리/야우/오프셋을 순차적으로 정렬
    - RECOVER: 전면 폭 부족 시 회전/대기 등으로 시야 확보 후 HOLD
    - CHECK: yaw/offset 최종 안정화 검사(CMD_STABLE_THR 프레임)
    - DONE: 정렬 완료(정지 유지)
    """
    def __init__(self):
        # FSM 현재 상태
        self.state: str = "SEARCH"

        # 하위 모듈: 제어, HUD, 서브 상태기
        self.execu = CommandExecutor()
        self.status = StatusHelper()
        self.align = AlignMachine(self.execu, self.status)
        self.recover = RecoverMachine(self.execu, self.status)

        # CHECK 안정화용 카운터
        self._stb = Stabilizer(CMD_STABLE_THR)

        # STOP 연속 송신 억제를 위한 마지막 전송 기록(0.2s 스로틀에 사용)
        self._last_cmd: Optional[str] = None
        self._last_ts: float = 0.0

    # -------------------------------------------------------------------------
    # 내부 유틸: 검사/대기 프레임 동안 STOP을 명시적으로 송신
    # - 같은 프레임/아주 짧은 시간 간격으로 STOP이 과도 송신되지 않도록 0.2s 스로틀
    # - HUD도 함께 STOP으로 업데이트하여 표시-제어 일치 보장
    # -------------------------------------------------------------------------
    def _ensure_stop(self) -> None:
        now = time.time()
        if self._last_cmd != "STOP" or (now - self._last_ts) > 0.2:
            self.execu.exec("STOP")
            self.status.start_timed("STOP", 0.0)
            self._last_cmd, self._last_ts = "STOP", now

    # 외부(HUD) 조회용: 현재 명령 상태(라벨/진행도) 객체 — 함수형 API(하위 호환)
    def get_command_status(self):
        return self.status.cmd_status

    # 속성형 API — main_rec.py에서 fsm.cmd_status로 직접 접근 가능
    @property
    def cmd_status(self):
        return self.status.cmd_status

    # -------------------------------------------------------------------------
    # 주 상태 전이 함수
    # - det_ok            : front 탐지 여부
    # - detected_length   : 전면(마스크/박스)의 가로 길이(시야 확보 판단)
    # - dist_z            : 목표물과의 Z 거리
    # - yaw_smooth        : 평활화된 yaw(법선 기반, 팔레트 면 기준 정렬용)
    # - offset_smooth     : (offset_x, offset_y, ...) 튜플(여기선 x만 사용)
    # - rel_yaw           : ★ IMU 기반 상대 yaw(gyro-Y 적분, deg) — 회전 90° 체인 종료 조건/진행률
    # 반환: HUD에 표시할 로그 텍스트/색상 리스트
    # -------------------------------------------------------------------------
    def step(self,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw_smooth: Optional[float],
             offset_smooth: Optional[tuple],
             rel_yaw: Optional[float] = None
             ) -> List[Tuple[str, tuple]]:

        lines: List[Tuple[str, tuple]] = []

        # offset_smooth가 (ox, oy, ...) 형태일 수 있으므로 안전하게 추출
        ox: Optional[float] = None
        if offset_smooth is not None:
            try:
                ox = float(offset_smooth[0])
            except Exception:
                ox = None

        # 허용 오차 판정(최종 체크/결정에 사용)
        yaw_ok = (yaw_smooth is not None) and (abs(yaw_smooth) <= YAW_TOL_DEG)
        off_ok = (ox is not None) and (abs(ox) <= OFF_TOL_M)
        band_ok = within_band(dist_z, ALIGN_DIST_M, ALIGN_BAND_M)

        # HUD의 until 진행도(예: |yaw|/tol, |offset|/tol)를 최신으로 갱신
        self.status.update_until_metric(yaw_smooth, ox)

        # ------------------------------ SEARCH ------------------------------
        if self.state == "SEARCH":
            # 서브 상태기 초기화(탐지 유무와 관계 없이 SEARCH 진입 시 리셋)
            self.align.reset("DIST_CHECK")
            self.recover.reset()
            self.execu.reset_last()  # 마지막 방향 등 내부 상태 초기화

            if det_ok:
                # 탐지 성공 → DETECTED 단계로 진입
                self.state = "DETECTED"
                lines.append(("[SEARCH→DETECTED]", COLOR_META))
            else:
                # (정책) 탐지 전에는 STOP 유지
                self._ensure_stop()
                lines.append(("[SEARCH] 탐지 대기: 정지(Stop)", COLOR_STATUS_TRK))
            return lines

        # ----------------------------- DETECTED -----------------------------
        if self.state == "DETECTED":
            if not det_ok:
                # 탐지 유실 → SEARCH로 복귀(정지)
                self.state = "SEARCH"
                self.execu.exec("STOP")
                self.status.start_timed("STOP", 0.0)
                lines.append(("[DETECTED→SEARCH] 미탐지", COLOR_ALERT))
                return lines

            # 전면 폭 충분 → ALIGN, 부족 → RECOVER
            if (detected_length is not None) and (detected_length >= WIDTH_MIN_FULL):
                self.state = "ALIGN"
                self.align.reset("DIST_CHECK")
                lines.append(("[DETECTED→ALIGN]", COLOR_META))
            else:
                self.state = "RECOVER"
                self.recover.reset()
                lines.append(("[DETECTED→RECOVER] 시야확보 필요", COLOR_ALERT))
            return lines

        # ------------------------------ RECOVER -----------------------------
        if self.state == "RECOVER":
            rec_lines = self.recover.step(det_ok, detected_length, ox)
            lines.extend(rec_lines)

            if self.recover.sub == "HOLD":
                self.state = "CHECK"
                self._stb.reset()
                lines.append(("[RECOVER→CHECK]", COLOR_META))
            return lines

        # ------------------------------- CHECK ------------------------------
        if self.state == "CHECK":
            self._ensure_stop()

            both_ok = ((yaw_smooth is not None and abs(yaw_smooth) <= YAW_TOL_DEG)
                       and (ox is not None and abs(ox) <= OFF_TOL_M))
            tag = "CHECK_OK" if both_ok else "CHECK_NOK"

            if self._stb.stable(tag):
                if both_ok:
                    self.state = "DONE"
                    lines.append(("[CHECK→DONE] 정렬 완료", COLOR_STATUS_OK))
                else:
                    self.state = "ALIGN"
                    self.align.reset("DIST_CHECK")
                    lines.append(("[CHECK→ALIGN] 정렬 필요", COLOR_STATUS_TRK))
            else:
                lines.append((f"[CHECK] 검수 중 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
            return lines

        # ------------------------------- ALIGN ------------------------------
        if self.state == "ALIGN":
            # 정렬 서브머신 실행 — ★ rel_yaw 전달 (90° 체인 종료/진행률에 사용)
            aln_lines = self.align.step(
                det_ok, detected_length, dist_z,
                yaw_smooth,  # 팔레트 면 기준 yaw (YAW_CHECK)
                ox,          # offset_x
                rel_yaw      # ★ IMU 기반 상대 yaw(gyro-Y)
            )
            lines.extend(aln_lines)

            if self.align.sub == "READY_TO_DONE":
                self.state = "DONE"
                self._ensure_stop()
                lines.append(("[ALIGN→DONE] 정렬 완료", COLOR_STATUS_OK))
            return lines

        # -------------------------------- DONE ------------------------------
        if self.state == "DONE":
            self._ensure_stop()
            lines.append(("[DONE] 정렬 완료 상태 유지", COLOR_STATUS_OK))
            return lines

        # ------------------------------ Fallback ----------------------------
        self._ensure_stop()
        lines.append((f"[{self.state}] 대기", COLOR_META))
        return lines
