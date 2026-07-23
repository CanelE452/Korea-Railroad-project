# calib/can/commands.py
# =============================================================================
# 고수준 명령 API.
#
# 주행 명령 (issue_command_*): 구 calib/control.py 와 **바이트/순서 동일**
#   (tests_can/golden 으로 검증). 차이는 bus 의 TX 스레드가 마지막 movement
#   프레임을 50ms 주기로 재송신한다는 것뿐 (fire-once 제거).
#
# 리프트/리치/폴드 명령 (신규): forklift_ctrl/controller.py 의 모드 시퀀스를
#   한 곳에 캡슐화 — 정지 → 모드 프레임 N회 → settle → 동작 프레임 유지
#   → 종료 시 모드 복귀 + settle. 모드 전환 레이스(파렛트 단계 오류 의심
#   원인)를 호출부가 아니라 여기서 차단한다.
# =============================================================================
from __future__ import annotations

import time

from .bus import BUS, is_mock

# settle 시간 — calib.config 의 MODE_SWITCH_SETTLE_S 우선, 없으면 0.2s
try:
    from .. import config as _cfg
    MODE_SWITCH_SETTLE_S = float(getattr(_cfg, "MODE_SWITCH_SETTLE_S", 0.2))
except Exception:
    MODE_SWITCH_SETTLE_S = 0.2

MODE_FRAME_REPEAT = 3   # 모드/동작 프레임 반복 송신 횟수 (수신 보장)

# 테스트에서 monkeypatch 가능한 sleep
_sleep = time.sleep


__all__ = [
    "can_init", "can_close",
    "start_heartbeat", "stop_heartbeat", "send_heartbeat",
    "issue_command_forward", "issue_command_forward_slow", "issue_command_backward",
    "issue_command_forward_and_turn", "issue_command_backward_and_turn",
    "issue_command_rotate_in_place", "issue_command_stop",
    "issue_lift_up", "issue_lift_down", "issue_lift_stop",
    "issue_reach_forward", "issue_reach_backward", "issue_reach_stop",
    "issue_fold", "issue_unfold", "issue_fold_stop",
    "is_mock", "bus_healthy", "keepalive", "arm_watchdog",
]


# =============================================================================
# 초기화/종료/하트비트 (구 control.py 시그니처 유지)
# =============================================================================
def can_init(channel: int = None, bitrate: int = None, is_extended_id: bool = None) -> bool:
    from . import protocol as P
    kwargs = {}
    if channel is not None:
        kwargs["channel"] = channel
    if bitrate is not None:
        kwargs["bitrate"] = bitrate
    if is_extended_id is not None:
        kwargs["is_extended_id"] = is_extended_id
    return BUS.init(**kwargs)


def can_close():
    BUS.close()


def send_heartbeat():
    BUS.send_heartbeat()


def start_heartbeat():
    """(하위호환) TX 스레드 기동 — 하트비트 포함 전체 재송신 루프."""
    BUS.start_tx_thread()


def stop_heartbeat():
    """(하위호환) TX 스레드 정지."""
    BUS.stop_tx_thread()


def bus_healthy() -> bool:
    return BUS.bus_healthy()


def keepalive():
    BUS.keepalive()


def arm_watchdog():
    BUS.arm_watchdog()


# =============================================================================
# 주행 명령 — 구 control.py 와 바이트/순서 동일 (골든 계약)
# =============================================================================
def _prime_driving_channel(burst_n: int = 1):
    for _ in range(burst_n):
        BUS.send_control("driving_mode", src="drv:prime")
        BUS.send_heartbeat(src="drv:prime")


def issue_command_forward() -> None:
    """전진 직진"""
    _prime_driving_channel(1)
    BUS.send_movement("forward", src="drv:fwd")


def issue_command_forward_slow() -> None:
    """저속 전진 (트럭 적재면 모서리 탐색용 — T16)"""
    _prime_driving_channel(1)
    BUS.send_movement("forward_slow", src="drv:fwd_slow")


def issue_command_backward() -> None:
    """후진 직진"""
    _prime_driving_channel(1)
    BUS.send_movement("backward", src="drv:back")


def issue_command_forward_and_turn(turn_dir: int) -> None:
    """전진 + 회전 (turn_dir: +1=좌, -1=우)"""
    _prime_driving_channel(1)
    name = "forward_left" if turn_dir > 0 else "forward_right"
    BUS.send_movement(name, src=f"drv:{name}")


def issue_command_backward_and_turn(turn_dir: int) -> None:
    """후진 + 회전 (turn_dir: +1=좌, -1=우)"""
    _prime_driving_channel(1)
    name = "backward_left" if turn_dir > 0 else "backward_right"
    BUS.send_movement(name, src=f"drv:{name}")


def issue_command_rotate_in_place(turn_dir: int) -> None:
    """제자리 회전 (turn_dir: +1=좌(CCW), -1=우(CW))"""
    ctrl = "rotate_ccw_ctrl" if turn_dir > 0 else "rotate_cw_ctrl"
    BUS.send_control(ctrl, src="drv:rot")
    BUS.send_heartbeat(src="drv:rot")
    name = "rotate_ccw" if turn_dir > 0 else "rotate_cw"
    BUS.send_movement(name, src=f"drv:{name}")


def issue_command_stop() -> None:
    """정지 (driving_mode + stop + heartbeat)"""
    BUS.send_control("driving_mode", src="drv:stop")
    BUS.send_movement("stop", src="drv:stop")
    BUS.send_heartbeat(src="drv:stop")
    BUS.set_active_control("driving_mode")


# =============================================================================
# 리프트/리치/폴드 — 모드 시퀀스 캡슐화 (신규)
# =============================================================================
def _control_action_start(mode_name: str, action_name: str):
    """모드 전환 시퀀스: 정지 → 모드 프레임 반복 → settle → 동작 프레임 유지.

    동작 프레임은 즉시 N회 송신 + TX 스레드가 CTRL_RESEND_PERIOD 주기로 유지.
    """
    # 1) 주행 중이면 안전 정지
    BUS.send_movement("stop", src=f"{mode_name}:pre_stop")
    # 2) 모드 프레임 반복 (수신 보장)
    for _ in range(MODE_FRAME_REPEAT):
        BUS.send_control(mode_name, src=f"{mode_name}:enter")
        BUS.send_heartbeat(src=f"{mode_name}:enter")
        _sleep(0.005)
    BUS.set_active_control(mode_name)
    # 3) settle — 차량 컨트롤러가 모드를 소화할 시간
    _sleep(MODE_SWITCH_SETTLE_S)
    # 4) 동작 프레임 — 즉시 반복 송신 + 재송신 루프에 등록 (해제 전까지 유지)
    for _ in range(MODE_FRAME_REPEAT):
        BUS.send_control(action_name, src=f"{mode_name}:{action_name}")
        _sleep(0.005)
    BUS.set_active_control(action_name)


def _control_action_stop(mode_name: str):
    """동작 해제 시퀀스: 모드 기본 프레임 복귀 → settle → 주행 모드 복귀."""
    # 1) 동작 해제 — 모드 기본 프레임으로 (controller.py 의 finally 와 동일 개념)
    for _ in range(MODE_FRAME_REPEAT):
        BUS.send_control(mode_name, src=f"{mode_name}:release")
        BUS.send_movement("stop", src=f"{mode_name}:release")
        BUS.send_heartbeat(src=f"{mode_name}:release")
        _sleep(0.005)
    BUS.set_active_control(mode_name)
    _sleep(MODE_SWITCH_SETTLE_S)
    # 2) 주행 모드 복귀
    for _ in range(MODE_FRAME_REPEAT):
        BUS.send_control("driving_mode", src="driving_mode:restore")
        _sleep(0.005)
    BUS.set_active_control("driving_mode")


def issue_lift_up() -> None:
    """포크 상승 시작 — issue_lift_stop() 전까지 유지."""
    _control_action_start("lift_mode", "lift_up")


def issue_lift_down() -> None:
    """포크 하강 시작 — issue_lift_stop() 전까지 유지."""
    _control_action_start("lift_mode", "lift_down")


def issue_lift_stop() -> None:
    """리프트 동작 해제 + 주행 모드 복귀."""
    _control_action_stop("lift_mode")


def issue_reach_forward() -> None:
    _control_action_start("reach_mode", "reach_forward")


def issue_reach_backward() -> None:
    _control_action_start("reach_mode", "reach_backward")


def issue_reach_stop() -> None:
    _control_action_stop("reach_mode")


def issue_fold() -> None:
    _control_action_start("folding_mode", "fold")


def issue_unfold() -> None:
    _control_action_start("folding_mode", "unfold")


def issue_fold_stop() -> None:
    _control_action_stop("folding_mode")
