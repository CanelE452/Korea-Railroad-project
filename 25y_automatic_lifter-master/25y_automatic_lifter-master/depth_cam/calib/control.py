# calib/control.py
# =============================================================================
# (호환 shim) CAN 통신 제어부 — 실제 구현은 calib/can/ 패키지로 이동 (2026-07).
#
# 기존 import 경로를 그대로 유지한다:
#   from calib.control import can_init, issue_command_forward, ...
#
# 새 코드는 calib.can 을 직접 사용 권장:
#   from calib.can import issue_lift_up, bus_healthy, ...
#
# 이동 이유 (기존 4벌 중복 CAN 구현 통합):
#   - protocol.py : ID/템플릿 단일 진실 (lift 코드/ID family 현장 플래그 포함)
#   - bus.py      : movement 재송신 루프(fire-once 제거) + 워치독 + 재초기화
#                   + 전 프레임 JSONL 로그
#   - commands.py : 주행(기존 바이트 동일) + 리프트/리치/폴드(신규)
# =============================================================================
from __future__ import annotations

# ---- 고수준 API (기존 시그니처 유지) ----
# 상대 import — `calib.control` / `depth_cam.calib.control` 두 경로 모두 지원.
from .can.commands import (  # noqa: F401
    can_init, can_close,
    start_heartbeat, stop_heartbeat, send_heartbeat,
    issue_command_forward, issue_command_forward_slow, issue_command_backward,
    issue_command_forward_and_turn, issue_command_backward_and_turn,
    issue_command_rotate_in_place, issue_command_stop,
    issue_lift_up, issue_lift_down, issue_lift_stop,
    issue_reach_forward, issue_reach_backward, issue_reach_stop,
    issue_fold, issue_unfold, issue_fold_stop,
    is_mock, bus_healthy, keepalive, arm_watchdog,
)

# ---- 프로토콜 상수 재노출 (eval/eval_motion.py 등이 참조) ----
from .can.protocol import (  # noqa: F401
    MessageFlag,
    CAN_CHANNEL, CAN_BITRATE, USE_EXTENDED_IDS,
    CAN_ID_FAMILY, CAN_MOVEMENT_ID, CAN_CONTROL_ID,
    HEARTBEAT_ID, HEARTBEAT_PERIOD, HEARTBEAT_DATA,
    AN_NEUTRAL, AN_FORWARD, AN_BACKWARD, AN_LEFT, AN_RIGHT,
    AN_FORWARD_SLOW, AN_ROTATE_CCW, AN_ROTATE_CW, AN_N,
    JOYSTICK_FORWARD, JOYSTICK_BACKWARD, JOYSTICK_LEFT, JOYSTICK_RIGHT,
    JOYSTICK_ROTATE_CCW, JOYSTICK_ROTATE_CW, JOYSTICK_FORWARD_SLOW,
    MOVEMENT_TEMPLATES, CONTROL_TEMPLATES,
    LIFT_UP_CODE, LIFT_DOWN_CODE,
)

from .can.bus import BUS as _BUS

__all__ = [
    "can_init", "can_close", "start_heartbeat", "stop_heartbeat", "send_heartbeat",
    "issue_command_forward", "issue_command_forward_slow", "issue_command_backward",
    "issue_command_forward_and_turn", "issue_command_backward_and_turn",
    "issue_command_rotate_in_place", "issue_command_stop",
    "issue_lift_up", "issue_lift_down", "issue_lift_stop",
    "issue_reach_forward", "issue_reach_backward", "issue_reach_stop",
    "issue_fold", "issue_unfold", "issue_fold_stop",
    "is_mock", "bus_healthy", "keepalive", "arm_watchdog",
    "inject_test_channel", "reset_sync_for_test",
]


# ---- 테스트 훅 (tests_can/test_golden_driving.py 가 사용) ----
def inject_test_channel(ch) -> None:
    """TX 스레드 없이 채널만 교체 — 골든 캡처/단위 테스트 전용."""
    _BUS.inject_test_channel(ch)


def reset_sync_for_test(seed: int = 0x0A) -> None:
    _BUS.reset_sync_for_test(seed)
