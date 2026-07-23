# tests_can/test_mission_lift.py
# =============================================================================
# MissionFSM LIFT_PALLET 단계 테스트 — 주입식 (fake clock + 기록 채널).
# tests/eval 의 injection 스타일: 실제 sleep/CAN/센서 없음.
# =============================================================================
from __future__ import annotations

import pytest

from calib.can import commands as canc
from calib.can import protocol as P
from calib.can.mockcan import RecordingChannel
from calib.fsm.mission import MissionFSM, lift_sec_for_height
import calib.config as cfg


class FakeClock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


@pytest.fixture
def rig(monkeypatch):
    """기록 채널 + settle sleep 무력화 + fake clock."""
    monkeypatch.setattr(canc, "_sleep", lambda s: None)
    rec = RecordingChannel()
    canc.BUS.inject_test_channel(rec)
    canc.BUS.reset_sync_for_test()
    clock = FakeClock()
    yield rec, clock
    canc.BUS.inject_test_channel(None)


def _ctrl_codes(rec):
    return [r["data"][3] for r in rec.records if r["id"] == P.CAN_CONTROL_ID]


def _drive_mission_to_lift(m: MissionFSM):
    """PHASE_A 를 즉시 DONE 으로 만들기 위해 pallet_fsm 상태를 직접 세팅."""
    m.pallet_fsm.state = "DONE"
    m.step(pallet_inputs={"det_ok": False, "detected_length": None,
                          "dist_z": None, "yaw_smooth": None,
                          "offset_smooth": None})
    assert m.phase == "LIFT_PALLET"


def test_lift_sec_formula():
    v = cfg.LIFT_SPEED_MPS
    assert lift_sec_for_height(cfg.PALLET_LIFT_M) == pytest.approx(
        max(cfg.LIFT_MIN_SEC, min(cfg.LIFT_MAX_SEC, cfg.PALLET_LIFT_M / v))
    )


def test_lift_pallet_sequence(rig):
    rec, clock = rig
    m = MissionFSM(truck_machine=None, now_fn=clock)
    _drive_mission_to_lift(m)

    rec.clear()
    # 첫 tick: LIFT_UP 시퀀스 송신 + 데드라인 설정
    m.step()
    codes = _ctrl_codes(rec)
    assert 0x05 in codes, "lift_mode 프레임 없음"
    assert P.LIFT_UP_CODE in codes, "lift_up 프레임 없음"
    assert m.phase == "LIFT_PALLET"

    # 시간 경과 전: 유지 (추가 LIFT_UP 재송신은 TX 스레드 몫 — 즉시 프레임 없음)
    rec.clear()
    clock.advance(m._lift_sec * 0.5)
    m.step()
    assert m.phase == "LIFT_PALLET"

    # 시간 경과 후: LIFT_STOP (lift_mode 해제 → driving_mode 복귀) → DONE
    rec.clear()
    clock.advance(m._lift_sec)
    m.step()
    codes = _ctrl_codes(rec)
    assert 0x05 in codes and 0x0A in codes, "해제(lift_mode)+복귀(driving_mode) 필요"
    assert m.phase == "DONE"


def test_lift_pallet_holds_until_deadline(rig):
    rec, clock = rig
    m = MissionFSM(truck_machine=None, now_fn=clock)
    _drive_mission_to_lift(m)

    m.step()                       # LIFT_UP 시작
    for _ in range(5):             # 데드라인 전 반복 tick — 상태 유지
        clock.advance(m._lift_sec / 10.0)
        m.step()
        assert m.phase == "LIFT_PALLET"


def test_phase_a_regression_untouched():
    """MissionFSM 도입이 CalibrationFSM 자체를 변경하지 않았는지 스모크."""
    m = MissionFSM(truck_machine=None)
    assert m.pallet_fsm.state == "SEARCH"
    assert m.phase == "PHASE_A"
    assert m.state == "A:SEARCH"


def test_executor_lift_codes_exist():
    """CommandExecutor 가 LIFT 코드를 실제로 dispatch 하는지 (mock 채널)."""
    from calib.fsm.commands import CommandExecutor, HUMAN_NAME
    for code in ("LIFT_UP", "LIFT_DOWN", "LIFT_STOP", "FWD_SLOW"):
        assert code in HUMAN_NAME
    ex = CommandExecutor()
    rec = RecordingChannel()
    canc.BUS.inject_test_channel(rec)
    canc.BUS.reset_sync_for_test()
    import calib.can.commands as C
    orig_sleep = C._sleep
    C._sleep = lambda s: None
    try:
        ex.exec("FWD_SLOW")
        movs = [tuple(r["data"]) for r in rec.records if r["id"] == P.CAN_MOVEMENT_ID]
        assert tuple(P.MOVEMENT_TEMPLATES["forward_slow"]) in movs
    finally:
        C._sleep = orig_sleep
        canc.BUS.inject_test_channel(None)
