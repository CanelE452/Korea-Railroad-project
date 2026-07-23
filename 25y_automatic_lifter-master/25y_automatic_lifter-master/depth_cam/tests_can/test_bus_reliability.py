# tests_can/test_bus_reliability.py
# =============================================================================
# calib/can/bus.py 신뢰성 계층 단위 테스트 (mock 채널, 하드웨어 불필요).
#   - movement 재송신 루프
#   - 워치독 (keepalive 끊김 → 정지 프레임 강등)
#   - TX 에러 카운터 → bus_healthy() False
#   - 리프트 모드 시퀀스 프레임 순서
# =============================================================================
from __future__ import annotations

import time

import pytest

from calib.can.bus import CanBus, TX_ERR_REINIT_THRESHOLD
from calib.can import protocol as P
from calib.can.mockcan import RecordingChannel, FailingChannel


def _ids(rec):
    return [r["id"] for r in rec.records]


def _mov_frames(rec):
    return [tuple(r["data"]) for r in rec.records if r["id"] == P.CAN_MOVEMENT_ID]


# --------------------------------------------------------------------- 재송신
def test_tx_thread_resends_movement():
    bus = CanBus()
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.send_movement("forward")           # active_movement = forward
    rec.clear()

    bus.start_tx_thread()
    time.sleep(0.25)                        # 50ms 주기 → 4~5회 기대
    bus.stop_tx_thread()

    movs = _mov_frames(rec)
    assert len(movs) >= 3, f"movement 재송신 부족: {len(movs)}회"
    fwd = tuple(P.MOVEMENT_TEMPLATES["forward"])
    assert all(m == fwd for m in movs), "재송신 프레임이 forward 템플릿과 다름"
    # 하트비트도 주기 송신
    assert any(r["id"] == P.HEARTBEAT_ID for r in rec.records)


def test_stop_replaces_resend_frame():
    bus = CanBus()
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.send_movement("forward")
    bus.send_movement("stop")
    rec.clear()

    bus.start_tx_thread()
    time.sleep(0.15)
    bus.stop_tx_thread()

    movs = _mov_frames(rec)
    stop = tuple(P.MOVEMENT_TEMPLATES["stop"])
    assert movs and all(m == stop for m in movs)


# --------------------------------------------------------------------- 워치독
def test_watchdog_degrades_to_stop():
    bus = CanBus()
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.send_movement("forward")

    # 워치독 armed + keepalive 시각을 과거로 조작 → 즉시 트립
    bus.arm_watchdog()
    bus._last_keepalive = time.monotonic() - 10.0

    rec.clear()
    bus.start_tx_thread()
    time.sleep(0.15)
    bus.stop_tx_thread()

    movs = _mov_frames(rec)
    stop = tuple(P.MOVEMENT_TEMPLATES["stop"])
    assert movs and all(m == stop for m in movs), "워치독 트립 시 정지 프레임이어야 함"


def test_watchdog_keepalive_keeps_movement():
    bus = CanBus()
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.send_movement("forward")
    bus.arm_watchdog()          # keepalive 방금 → 트립 안 함

    rec.clear()
    bus.start_tx_thread()
    time.sleep(0.15)
    bus.stop_tx_thread()

    movs = _mov_frames(rec)
    fwd = tuple(P.MOVEMENT_TEMPLATES["forward"])
    assert movs and all(m == fwd for m in movs)


# ----------------------------------------------------------------- 에러 처리
def test_tx_errors_mark_unhealthy():
    bus = CanBus()
    bus.inject_test_channel(FailingChannel())
    assert bus.bus_healthy()
    for _ in range(TX_ERR_REINIT_THRESHOLD):
        bus.send_movement("forward")
    assert not bus.bus_healthy(), "연속 TX 실패 후 unhealthy 여야 함"


def test_tx_recovers_on_success():
    bus = CanBus()
    fail = FailingChannel()
    bus.inject_test_channel(fail)
    for _ in range(TX_ERR_REINIT_THRESHOLD):
        bus.send_movement("forward")
    assert not bus.bus_healthy()

    bus.inject_test_channel(RecordingChannel())
    bus.send_movement("stop")
    assert bus.bus_healthy(), "성공 송신 후 healthy 복구되어야 함"


# ------------------------------------------------------------ 리프트 시퀀스
def test_lift_up_mode_sequence(monkeypatch):
    """issue_lift_up: stop → lift_mode×N → lift_up×N 순서 검증 (settle sleep 무력화)."""
    from calib.can import commands as C
    monkeypatch.setattr(C, "_sleep", lambda s: None)

    bus = C.BUS
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.reset_sync_for_test()
    try:
        C.issue_lift_up()
    finally:
        recs = list(rec.records)
        C.issue_lift_stop()   # active control 복원
        bus.inject_test_channel(None)

    # movement stop 이 가장 먼저
    mov = [r for r in recs if r["id"] == P.CAN_MOVEMENT_ID]
    assert mov and tuple(mov[0]["data"]) == tuple(P.MOVEMENT_TEMPLATES["stop"])

    # control 프레임: lift_mode (code 0x05) 들이 lift_up (LIFT_UP_CODE) 보다 먼저
    ctrls = [r["data"][3] for r in recs if r["id"] == P.CAN_CONTROL_ID]
    assert 0x05 in ctrls, "lift_mode 프레임 없음"
    assert P.LIFT_UP_CODE in ctrls, "lift_up 프레임 없음"
    assert ctrls.index(0x05) < ctrls.index(P.LIFT_UP_CODE), "모드 프레임이 동작 프레임보다 먼저여야 함"

    # 동작 프레임 반복 송신 (수신 보장)
    assert ctrls.count(P.LIFT_UP_CODE) >= C.MODE_FRAME_REPEAT


def test_lift_stop_restores_driving(monkeypatch):
    from calib.can import commands as C
    monkeypatch.setattr(C, "_sleep", lambda s: None)

    bus = C.BUS
    rec = RecordingChannel()
    bus.inject_test_channel(rec)
    bus.reset_sync_for_test()
    try:
        C.issue_lift_stop()
    finally:
        bus.inject_test_channel(None)

    ctrls = [r["data"][3] for r in rec.records if r["id"] == P.CAN_CONTROL_ID]
    # lift_mode (해제) → driving_mode (복귀) 순서
    assert 0x05 in ctrls and 0x0A in ctrls
    assert ctrls.index(0x05) < len(ctrls) - 1 - ctrls[::-1].index(0x0A), \
        "driving_mode 복귀가 마지막에 와야 함"
    assert bus._active_control == "driving_mode"


# ------------------------------------------------------------- 프로토콜 플래그
def test_lift_codes_default():
    """기본값: v1 + forklift_ctrl/controller.py 다수결 (up=0x25/down=0x15)."""
    assert P.LIFT_UP_CODE == 0x25
    assert P.LIFT_DOWN_CODE == 0x15
    assert P.CONTROL_TEMPLATES["lift_up"][3] == P.LIFT_UP_CODE
    assert P.CONTROL_TEMPLATES["lift_down"][3] == P.LIFT_DOWN_CODE


def test_id_family_default():
    assert P.CAN_ID_FAMILY == 0xE3
    assert P.CAN_MOVEMENT_ID == 0x01E3
    assert P.CAN_CONTROL_ID == 0x02E3


def test_slow_forward_template():
    t = P.MOVEMENT_TEMPLATES["forward_slow"]
    assert t[2] == P.AN_FORWARD_SLOW == 102   # 127 - 25
    # 나머지 바이트는 중립
    assert all(b == P.AN_NEUTRAL for i, b in enumerate(t) if i != 2)
