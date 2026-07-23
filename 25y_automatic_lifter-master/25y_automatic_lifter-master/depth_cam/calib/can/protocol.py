# calib/can/protocol.py
# =============================================================================
# CAN 프로토콜 단일 진실 (Single Source of Truth).
#
# 기존 4개 구현 (CAN/control_forklift.py, CAN/control_forklift_v2.py,
# forklift_ctrl/controller.py, calib/control.py) 에 흩어져 있던 ID / 템플릿 /
# 하트비트 규격을 여기로 통합. 이 파일은 canlib 을 import 하지 않는 순수 데이터
# 모듈이라 어떤 환경에서도 import/테스트 가능.
#
# ⚠ 하드웨어 미확정 사항 2건 — 현장 벤치 테스트로 확정 후 이 파일만 수정:
#   1. CAN_ID_FAMILY: 0xE3 (기본) vs 0xE4.
#      v2/controller.py/control.py 와 실동작한 파렛트 주행이 모두 0xE3.
#      v1(control_forklift.py) 만 0xE4. 차량이 0x02E3 control 에 무반응이면
#      환경변수 CAN_ID_FAMILY=E4 로 전환해 재시험.
#   2. LIFT_UP_CODE / LIFT_DOWN_CODE: v1 + forklift_ctrl/controller.py 는
#      up=0x25/down=0x15, v2 는 반대. 다수결(기본값) up=0x25/down=0x15.
#      벤치 리프트 조그에서 반대로 움직이면 환경변수 LIFT_CODES_SWAPPED=1.
# =============================================================================
from __future__ import annotations

import os
from typing import Dict, List


# -----------------------------------------------------------------------------
# 메시지 플래그 (기존 4곳 중복 정의 → 여기 1곳)
# -----------------------------------------------------------------------------
class MessageFlag:
    STD = 0x0000  # 표준 11-bit ID
    EXT = 0x0004  # 확장 29-bit ID


# -----------------------------------------------------------------------------
# 채널/비트레이트 기본값
# -----------------------------------------------------------------------------
CAN_CHANNEL = 0
CAN_BITRATE = 500_000
USE_EXTENDED_IDS = False


# -----------------------------------------------------------------------------
# CAN ID — family 바이트로 파생 (벤치 1단계에서 확정)
# -----------------------------------------------------------------------------
def _resolve_id_family() -> int:
    raw = os.environ.get("CAN_ID_FAMILY", "").strip().upper().lstrip("0X")
    if raw in ("E3", ""):
        return 0xE3
    if raw == "E4":
        return 0xE4
    print(f"[protocol] CAN_ID_FAMILY={raw!r} 무시 (E3/E4 만 허용) — 0xE3 사용")
    return 0xE3


CAN_ID_FAMILY = _resolve_id_family()
CAN_MOVEMENT_ID = 0x0100 | CAN_ID_FAMILY   # 0x01E3 (기본)
CAN_CONTROL_ID = 0x0200 | CAN_ID_FAMILY    # 0x02E3 (기본)

# Heartbeat (v2/controller/control.py 합의값 — v1 의 300ms 는 폐기)
HEARTBEAT_ID = 0x764
HEARTBEAT_PERIOD = 0.200
HEARTBEAT_DATA = [0x00]


# -----------------------------------------------------------------------------
# 조이스틱 강도 / movement 템플릿 (calib/control.py 와 바이트 동일)
# -----------------------------------------------------------------------------
AN_NEUTRAL = 127
JOYSTICK_FORWARD = 60
JOYSTICK_BACKWARD = 60
JOYSTICK_LEFT = 60
JOYSTICK_RIGHT = 60
JOYSTICK_ROTATE_CCW = 30
JOYSTICK_ROTATE_CW = 30

# 저속 전진 (트럭 적재면 모서리 탐색 T16 용 — 신규)
JOYSTICK_FORWARD_SLOW = 25

AN_FORWARD = min(255, AN_NEUTRAL - JOYSTICK_FORWARD)        # 67
AN_BACKWARD = max(0, AN_NEUTRAL + JOYSTICK_BACKWARD)        # 187
AN_LEFT = min(255, AN_NEUTRAL + JOYSTICK_LEFT)              # 187
AN_RIGHT = max(0, AN_NEUTRAL - JOYSTICK_RIGHT)              # 67
AN_FORWARD_SLOW = min(255, AN_NEUTRAL - JOYSTICK_FORWARD_SLOW)  # 102

# 제자리 회전 힘 — v2 기준 118 고정 (CCW→data[4], CW→data[5])
AN_ROTATE_CCW = 118
AN_ROTATE_CW = 118

AN_N = AN_NEUTRAL

# Byte 의미(관측 기반):
# [0]?, [1]=steer(left/right), [2]=drive(fwd/back), [3]?, [4]=rot_CCW, [5]=rot_CW, [6]?, [7]?
MOVEMENT_TEMPLATES: Dict[str, List[int]] = {
    "stop":           [AN_N, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N],
    "forward":        [AN_N, AN_N, AN_FORWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "forward_slow":   [AN_N, AN_N, AN_FORWARD_SLOW, AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward":       [AN_N, AN_N, AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "turn_left":      [AN_N, AN_LEFT, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N],
    "turn_right":     [AN_N, AN_RIGHT, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N],
    "rotate_ccw":     [AN_N, AN_N, AN_N, AN_N, AN_ROTATE_CCW, AN_N, AN_N, AN_N],
    "rotate_cw":      [AN_N, AN_N, AN_N, AN_N, AN_N, AN_ROTATE_CW, AN_N, AN_N],
    "forward_left":   [AN_N, AN_LEFT, AN_FORWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "forward_right":  [AN_N, AN_RIGHT, AN_FORWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward_left":  [AN_N, AN_LEFT, AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward_right": [AN_N, AN_RIGHT, AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
}


# -----------------------------------------------------------------------------
# control 템플릿 — forklift_ctrl/controller.py 의 전체 세트 흡수
# (기존 calib/control.py 는 driving/rotate/emergency 만 갖고 있어 리프트 불가였음)
#
# 프레임 형식: [0x42, 0, 0, <code>, <sync>, 0x40, 0x69, 0x93]
#   - data[3] = 기능 코드, data[4] = sync 카운터 (emergency 제외, runtime 갱신)
# -----------------------------------------------------------------------------
_LIFT_CODES_SWAPPED = os.environ.get("LIFT_CODES_SWAPPED", "0") == "1"
# 기본: v1 + forklift_ctrl/controller.py 다수결 (up=0x25 / down=0x15).
# v2(control_forklift_v2.py) 는 반대 — 벤치 리프트 조그로 확정할 것.
LIFT_UP_CODE = 0x15 if _LIFT_CODES_SWAPPED else 0x25
LIFT_DOWN_CODE = 0x25 if _LIFT_CODES_SWAPPED else 0x15


def _ctrl(code: int) -> List[int]:
    return [0x42, 0x00, 0x00, code, 0x00, 0x40, 0x69, 0x93]


CONTROL_TEMPLATES: Dict[str, List[int]] = {
    # 모드 프레임
    "driving_mode":    _ctrl(0x0A),
    "lift_mode":       _ctrl(0x05),
    "folding_mode":    _ctrl(0x06),
    "reach_mode":      _ctrl(0x09),
    # 동작 프레임
    "lift_up":         _ctrl(LIFT_UP_CODE),
    "lift_down":       _ctrl(LIFT_DOWN_CODE),
    "fold":            _ctrl(0x26),
    "unfold":          _ctrl(0x16),
    "reach_forward":   _ctrl(0x19),
    "reach_backward":  _ctrl(0x29),
    # 회전 시 동반 control (driving_mode 와 동일 바이트 — 기존 유지)
    "rotate_ccw_ctrl": _ctrl(0x0A),
    "rotate_cw_ctrl":  _ctrl(0x0A),
    # 비상 (sync 미적용)
    "emergency":       [0x80, 0x00, 0x00, 0x00, 0x01, 0x40, 0x69, 0x93],
}

# sync 카운터 초기값 (기존 구현과 동일)
SYNC_COUNTER_SEED = 0x0A


def next_sync(counter: int) -> int:
    """sync 카운터 규칙: (n+1) & 0x0F."""
    return (counter + 1) & 0x0F
