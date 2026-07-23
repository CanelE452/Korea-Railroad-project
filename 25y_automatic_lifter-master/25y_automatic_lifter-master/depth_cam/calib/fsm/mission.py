# calib/fsm/mission.py
# =============================================================================
# Mission-level FSM — 전체 작업 흐름 (truck_loading/다이어그램.txt 의 최상위).
#
#   PHASE_A (파렛트 삽입, 기존 CalibrationFSM: SEARCH→ALIGN→INSERT→DONE)
#     → LIFT_PALLET (P14: lift_fork(+PALLET_LIFT_M), 시간 기반 open-loop)
#     → PHASE_B (트럭 적재, TruckMachine: T0~T27)  [truck_machine 없으면 스킵]
#     → DONE
#
# 다이어그램 P15 == "mode = TRUCK_LOADING" ==> T0 의 코드화.
# 기존 CalibrationFSM 은 수정하지 않고 감싼다 (파렛트 동작 회귀 방지).
# =============================================================================
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from calib.config import (
    COLOR_META, COLOR_ALERT, COLOR_STATUS_OK, COLOR_STATUS_TRK,
    PALLET_LIFT_M, LIFT_SPEED_MPS, LIFT_MIN_SEC, LIFT_MAX_SEC,
)
from .top import CalibrationFSM


def lift_sec_for_height(height_m: float) -> float:
    """시간 기반 리프트: t = h / v (안전 클램프). LIFT_SPEED_MPS 는 현장 캘리브레이션."""
    v = max(1e-3, float(LIFT_SPEED_MPS))
    sec = float(height_m) / v
    return max(LIFT_MIN_SEC, min(LIFT_MAX_SEC, sec))


class MissionFSM:
    """PHASE_A → LIFT_PALLET → PHASE_B → DONE.

    Args:
        truck_machine : TruckMachine 인스턴스 (없으면 LIFT_PALLET 후 DONE)
        now_fn        : 시간 함수 주입 (테스트용)
    """

    PHASES = ("PHASE_A", "LIFT_PALLET", "PHASE_B", "DONE")

    def __init__(self, truck_machine=None, now_fn=time.time):
        self.now = now_fn
        self.pallet_fsm = CalibrationFSM()
        self.truck = truck_machine
        self.phase: str = "PHASE_A"

        # LIFT_PALLET 상태
        self._lift_deadline: Optional[float] = None
        self._lift_sec: float = lift_sec_for_height(PALLET_LIFT_M)

        # 공유 실행기 (중복 송신 억제 일원화)
        self.execu = self.pallet_fsm.execu

    # ------------------------------------------------------------------ 상태
    @property
    def cmd_status(self):
        return self.pallet_fsm.cmd_status

    @property
    def state(self) -> str:
        """HUD 용 결합 상태 문자열."""
        if self.phase == "PHASE_A":
            return f"A:{self.pallet_fsm.state}"
        if self.phase == "PHASE_B" and self.truck is not None:
            return f"B:{self.truck.state}"
        return self.phase

    # ------------------------------------------------------------------ step
    def step(self,
             pallet_inputs: Optional[dict] = None,
             truck_inputs: Optional[dict] = None,
             ) -> List[Tuple[str, tuple]]:
        """1 tick 진행.

        Args:
            pallet_inputs : CalibrationFSM.step kwargs (PHASE_A 에서 사용)
            truck_inputs  : TruckMachine.step kwargs (PHASE_B 에서 사용)
        """
        lines: List[Tuple[str, tuple]] = []

        # --------------------------- PHASE_A ---------------------------
        if self.phase == "PHASE_A":
            if pallet_inputs is None:
                pallet_inputs = {"det_ok": False, "detected_length": None,
                                 "dist_z": None, "yaw_smooth": None,
                                 "offset_smooth": None}
            lines.extend(self.pallet_fsm.step(**pallet_inputs))

            if self.pallet_fsm.state == "DONE":
                self.phase = "LIFT_PALLET"
                self._lift_deadline = None
                lines.append((f"[PHASE_A→LIFT_PALLET] 삽입 완료 — 파렛트 상승 "
                              f"(+{PALLET_LIFT_M:.2f}m ≈ {self._lift_sec:.1f}s)", COLOR_META))
            return lines

        # ------------------------- LIFT_PALLET -------------------------
        if self.phase == "LIFT_PALLET":
            if self._lift_deadline is None:
                # LIFT_UP 은 내부에서 정지→lift_mode→settle→lift_up 시퀀스 수행
                self.execu.exec("LIFT_UP")
                self._lift_deadline = self.now() + self._lift_sec

            remain = max(0.0, self._lift_deadline - self.now())
            if remain > 0.0:
                lines.append((f"[LIFT_PALLET] 상승 중 ({remain:.1f}s)", COLOR_STATUS_TRK))
                return lines

            # 상승 완료 → 리프트 해제 + 주행 모드 복귀
            self.execu.exec("LIFT_STOP")
            self._lift_deadline = None
            if self.truck is not None:
                self.phase = "PHASE_B"
                lines.append(("[LIFT_PALLET→PHASE_B] mode = TRUCK_LOADING", COLOR_META))
            else:
                self.phase = "DONE"
                lines.append(("[LIFT_PALLET→DONE] (트럭 단계 비활성)", COLOR_STATUS_OK))
            return lines

        # --------------------------- PHASE_B ---------------------------
        if self.phase == "PHASE_B":
            if self.truck is None:
                self.phase = "DONE"
                return lines
            lines.extend(self.truck.step(**(truck_inputs or {})))
            if self.truck.state == "DONE":
                self.phase = "DONE"
                lines.append(("[PHASE_B→DONE] 트럭 적재 완료", COLOR_STATUS_OK))
            return lines

        # ---------------------------- DONE -----------------------------
        self.execu.exec("STOP")
        lines.append(("[MISSION DONE] 전체 작업 완료", COLOR_STATUS_OK))
        return lines
