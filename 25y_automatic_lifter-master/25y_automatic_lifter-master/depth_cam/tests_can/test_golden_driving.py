# tests_can/test_golden_driving.py
# =============================================================================
# 골든 캡처 테스트 — CAN 리팩터 계약 (Step 0).
#
# 목적:
#   calib.control 의 issue_command_* 가 실제로 송신하는 CAN 프레임 바이트
#   시퀀스를 기록해 golden/*.json 과 비교한다.
#   - golden 파일이 없으면: 현재 동작을 캡처해 골든으로 저장 (최초 1회).
#   - golden 파일이 있으면: 현재 동작과 바이트 단위 비교 → 리팩터 회귀 감지.
#
# 리팩터 전(구 control.py 단일 파일)과 후(calib/can 패키지 + shim) 모두에서
# 동작하도록 주입 지점을 이중으로 지원한다:
#   - 신규: control.inject_test_channel(ch) / control.reset_sync_for_test()
#   - 구형: control._CTX.ch = ch / control._CTX.sync_counter = 0x0A
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import pytest

import calib.control as control

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_PATH = GOLDEN_DIR / "driving_commands.json"

SYNC_SEED = 0x0A  # control.py 초기값과 동일 — 캡처 결정성 보장


class RecordingChannel:
    """canlib.Channel 대역 — write 된 Frame 을 (id, data, flags) 로 기록."""

    def __init__(self):
        self.records = []

    def write(self, frame):
        self.records.append({
            "id": int(frame.id),
            "data": [int(b) for b in frame.data],
            "flags": int(getattr(frame, "flags", 0)),
        })

    # canlib.Channel 호환 no-op (혹시 호출되어도 무해)
    def setBusParams(self, *a, **k):
        pass

    def busOn(self):
        pass

    def busOff(self):
        pass

    def close(self):
        pass


def _install_recorder(rec: RecordingChannel):
    """리팩터 전/후 겸용 채널 주입."""
    if hasattr(control, "inject_test_channel"):
        control.inject_test_channel(rec)
    else:
        control._CTX.ch = rec


def _uninstall_recorder():
    if hasattr(control, "inject_test_channel"):
        control.inject_test_channel(None)
    else:
        control._CTX.ch = None


def _reset_sync():
    """sync 카운터를 고정 시드로 리셋 — 캡처 결정성."""
    if hasattr(control, "reset_sync_for_test"):
        control.reset_sync_for_test(SYNC_SEED)
    else:
        control._CTX.sync_counter = SYNC_SEED


# 캡처 대상: (골든 키, 호출)
COMMANDS = [
    ("forward", lambda: control.issue_command_forward()),
    ("backward", lambda: control.issue_command_backward()),
    ("forward_and_turn_left", lambda: control.issue_command_forward_and_turn(+1)),
    ("forward_and_turn_right", lambda: control.issue_command_forward_and_turn(-1)),
    ("backward_and_turn_left", lambda: control.issue_command_backward_and_turn(+1)),
    ("backward_and_turn_right", lambda: control.issue_command_backward_and_turn(-1)),
    ("rotate_in_place_left", lambda: control.issue_command_rotate_in_place(+1)),
    ("rotate_in_place_right", lambda: control.issue_command_rotate_in_place(-1)),
    ("stop", lambda: control.issue_command_stop()),
]


def _capture_all() -> dict:
    captured = {}
    for name, fn in COMMANDS:
        rec = RecordingChannel()
        _install_recorder(rec)
        _reset_sync()
        try:
            fn()
        finally:
            _uninstall_recorder()
        captured[name] = rec.records
    return captured


def test_driving_commands_match_golden():
    captured = _capture_all()

    if not GOLDEN_PATH.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(captured, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pytest.skip(f"golden 최초 생성: {GOLDEN_PATH} — 다음 실행부터 비교")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert set(captured.keys()) == set(golden.keys()), (
        f"명령 집합 변경: {sorted(set(captured) ^ set(golden))}"
    )
    for name in golden:
        assert captured[name] == golden[name], (
            f"[{name}] CAN 프레임 시퀀스가 골든과 다름!\n"
            f"golden : {golden[name]}\n"
            f"current: {captured[name]}"
        )


def test_movement_templates_expected_bytes():
    """핵심 바이트 값 스모크 체크 — 템플릿 자체의 회귀 감지 (골든과 독립)."""
    mt = control.MOVEMENT_TEMPLATES
    assert mt["stop"] == [127] * 8
    assert mt["forward"][2] == 67       # AN_NEUTRAL - 60
    assert mt["backward"][2] == 187     # AN_NEUTRAL + 60
    assert mt["rotate_ccw"][4] == 118   # CCW 힘 고정값
    assert mt["rotate_cw"][5] == 118    # CW 힘 고정값


def test_can_ids():
    assert control.CAN_MOVEMENT_ID == 0x01E3
    assert control.CAN_CONTROL_ID == 0x02E3
    assert control.HEARTBEAT_ID == 0x764
