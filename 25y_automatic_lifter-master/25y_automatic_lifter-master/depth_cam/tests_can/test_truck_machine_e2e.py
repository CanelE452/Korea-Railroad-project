# tests_can/test_truck_machine_e2e.py
# =============================================================================
# TruckMachine E2E 시뮬레이션 — fake clock + 합성 센서 + 기록 채널.
# 다이어그램 T0→T27 전 구간을 하드웨어 없이 완주하고,
# 상태 궤적과 CAN 명령 시퀀스를 검증한다.
# =============================================================================
from __future__ import annotations

import pytest

import calib.config as cfg
from calib.can import commands as canc
from calib.can import protocol as P
from calib.can.mockcan import RecordingChannel
from calib.fsm.commands import CommandExecutor
from calib.fsm.truck import TruckMachine
from calib.fsm.mission import lift_sec_for_height


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self):
        return self.t

    def advance(self, dt: float):
        self.t += dt


class SimRig:
    """TruckMachine + 합성 센서 시뮬레이터.

    회전: ROT 명령 중 rel_yaw 가 60°/s 로 누적.
    레이저: 시나리오 단계별 값 주입.
    """

    def __init__(self, monkeypatch):
        monkeypatch.setattr(canc, "_sleep", lambda s: None)
        self.rec = RecordingChannel()
        canc.BUS.inject_test_channel(self.rec)
        canc.BUS.reset_sync_for_test()

        self.clock = FakeClock()
        self.execu = CommandExecutor()
        self.m = TruckMachine(execu=self.execu, now_fn=self.clock)

        self.rel_yaw = 0.0
        self.laser_l = 1.50   # 바닥까지 (포크 하부)
        self.laser_r = 1.50
        self.truck_state = None
        self.trace = [self.m.state]   # 상태 궤적 (초기 상태 포함)

    def close(self):
        canc.BUS.inject_test_channel(None)

    def tick(self, dt: float = 0.1):
        # ROT 명령 중이면 IMU 적분 시뮬레이션 (±60°/s)
        last = self.execu.last_cmd
        if last == "ROT_RIGHT":
            self.rel_yaw += 60.0 * dt
        elif last == "ROT_LEFT":
            self.rel_yaw -= 60.0 * dt
        self.clock.advance(dt)
        lines = self.m.step(
            truck_state=self.truck_state,
            rel_yaw=self.rel_yaw,
            laser_l=self.laser_l,
            laser_r=self.laser_r,
        )
        if not self.trace or self.trace[-1] != self.m.state:
            self.trace.append(self.m.state)
        return lines

    def run_until(self, state: str, max_ticks: int = 5000, dt: float = 0.1):
        for _ in range(max_ticks):
            if self.m.state == state:
                return True
            self.tick(dt)
        return False


@pytest.fixture
def rig(monkeypatch):
    r = SimRig(monkeypatch)
    yield r
    r.close()


def test_full_phase_b_run(rig):
    """T0→T27 완주: 측면 이동 + ψ 보정 + 접근 + 상승 + 모서리 + 하강 + 후진."""
    m = rig.m
    assert m.state == "WAIT_SENSORS"

    # T0: 레이저 유효 → TRUCK_CHECK
    rig.tick()
    assert m.state == "TRUCK_CHECK"

    # T6: 트럭 상태 확정 (ψ=+10°, d_lat=-0.5m → 우측 체인, d_fwd=3.0m)
    rig.truck_state = (10.0, -0.5, 3.0)
    rig.tick()
    assert m.state == "LAT_ROT"
    rig.truck_state = None   # snapshot 이후 perception 무시 (눈 감고 동작)

    # 측면 체인: LAT_ROT → LAT_FWD → LAT_ROT_BACK → FACE_ROT
    assert rig.run_until("LAT_FWD"), f"trace={rig.trace}"
    assert rig.run_until("LAT_ROT_BACK"), f"trace={rig.trace}"
    assert rig.run_until("FACE_ROT"), f"trace={rig.trace}"

    # ψ 보정 후 접근
    assert rig.run_until("APPROACH_FWD"), f"trace={rig.trace}"
    assert rig.run_until("RAISE_FORK"), f"trace={rig.trace}"

    # T14: 상승 (시간 기반) → EDGE_SEARCH
    assert rig.run_until("EDGE_SEARCH"), f"trace={rig.trace}"

    # T15~T20: 저속 전진 중 적재면 진입 — 양쪽 레이저 동시 급감 (1.5→0.4)
    for _ in range(5):
        rig.tick()               # baseline 형성
    rig.laser_l = rig.laser_r = 0.40
    assert rig.run_until("LOWER", max_ticks=50), f"trace={rig.trace}"

    # T21~TD6: 하강 — 레이저 0.4 → 0.03 (안착)
    for _ in range(3):
        rig.tick()
    rig.laser_l = rig.laser_r = 0.03
    assert rig.run_until("RELEASE", max_ticks=50), f"trace={rig.trace}"

    # T25→T26→T27
    assert rig.run_until("BACKOUT"), f"trace={rig.trace}"
    assert rig.run_until("DONE", max_ticks=3000), f"trace={rig.trace}"

    # ---- 상태 궤적 검증 (다이어그램 순서) ----
    expected_order = ["WAIT_SENSORS", "TRUCK_CHECK", "LAT_ROT", "LAT_FWD",
                      "LAT_ROT_BACK", "FACE_ROT", "APPROACH_FWD", "RAISE_FORK",
                      "EDGE_SEARCH", "LOWER", "RELEASE", "BACKOUT", "DONE"]
    core = [s for s in rig.trace if s in expected_order]
    # 중복 제거 (인터록 재진입 등) 후 순서 확인
    dedup = [s for i, s in enumerate(core) if i == 0 or core[i - 1] != s]
    assert dedup == expected_order, f"궤적 불일치: {dedup}"

    # ---- CAN 명령 검증 ----
    ctrl_codes = [r["data"][3] for r in rig.rec.records if r["id"] == P.CAN_CONTROL_ID]
    assert 0x05 in ctrl_codes, "lift_mode 프레임 필요 (상승/하강)"
    assert P.LIFT_UP_CODE in ctrl_codes, "lift_up 프레임 필요 (T14)"
    assert P.LIFT_DOWN_CODE in ctrl_codes, "lift_down 프레임 필요 (T21)"

    movs = [tuple(r["data"]) for r in rig.rec.records if r["id"] == P.CAN_MOVEMENT_ID]
    assert tuple(P.MOVEMENT_TEMPLATES["forward_slow"]) in movs, "저속 전진 필요 (T16)"
    assert tuple(P.MOVEMENT_TEMPLATES["backward"]) in movs, "후진 필요 (T26)"
    assert movs[-1] == tuple(P.MOVEMENT_TEMPLATES["stop"]), "마지막은 정지"


def test_no_lateral_no_yaw_shortcut(rig):
    """정렬돼 있으면 측면/ψ 체인 스킵 — TRUCK_CHECK → APPROACH_FWD 직행."""
    m = rig.m
    rig.tick()
    rig.truck_state = (0.5, 0.02, 2.0)   # 허용치 내
    rig.tick()
    assert m.state == "APPROACH_FWD", f"state={m.state}"


def test_edge_search_timeout_faults(rig):
    """적재면을 끝내 못 찾으면 타임아웃 → FAULT + STOP (폭주 방지)."""
    m = rig.m
    rig.tick()
    rig.truck_state = (0.0, 0.0, cfg.TRUCK_SAFETY_MARGIN_M)  # 접근 불필요
    rig.tick()
    assert rig.run_until("EDGE_SEARCH"), f"trace={rig.trace}"
    # 레이저 급감 없이 시간만 흐름
    for _ in range(int(m._edge_timeout_s / 0.1) + 20):
        rig.tick()
        if m.state == "FAULT":
            break
    assert m.state == "FAULT"
    assert "타임아웃" in m.fault_reason

    movs = [tuple(r["data"]) for r in rig.rec.records if r["id"] == P.CAN_MOVEMENT_ID]
    assert movs[-1] == tuple(P.MOVEMENT_TEMPLATES["stop"])


def test_laser_stale_during_edge_search_faults(rig):
    """모서리 탐색 중 레이저 stale → 즉시 FAULT (전진 금지)."""
    m = rig.m
    rig.tick()
    rig.truck_state = (0.0, 0.0, cfg.TRUCK_SAFETY_MARGIN_M)
    rig.tick()
    assert rig.run_until("EDGE_SEARCH"), f"trace={rig.trace}"
    rig.tick()
    rig.laser_l = None                 # 왼쪽 레이저 끊김
    rig.tick()
    assert m.state == "FAULT"
    assert "stale" in m.fault_reason


def test_mission_phase_b_integration(rig, monkeypatch):
    """MissionFSM 에 TruckMachine 연결 — LIFT_PALLET 후 PHASE_B 진입."""
    from calib.fsm.mission import MissionFSM
    mission = MissionFSM(truck_machine=rig.m, now_fn=rig.clock)
    mission.pallet_fsm.state = "DONE"

    # PHASE_A tick → LIFT_PALLET 전이
    mission.step(pallet_inputs={"det_ok": False, "detected_length": None,
                                "dist_z": None, "yaw_smooth": None,
                                "offset_smooth": None})
    assert mission.phase == "LIFT_PALLET"

    # 리프트 완료까지
    mission.step()
    rig.clock.advance(lift_sec_for_height(cfg.PALLET_LIFT_M) + 1.0)
    mission.step()
    assert mission.phase == "PHASE_B"
    assert mission.state == "B:WAIT_SENSORS"

    # PHASE_B 1 tick — 레이저 전달 확인
    mission.step(truck_inputs={"truck_state": None, "rel_yaw": 0.0,
                               "laser_l": 1.5, "laser_r": 1.5})
    assert rig.m.state == "TRUCK_CHECK"
