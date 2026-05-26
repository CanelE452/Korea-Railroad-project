"""FSM 다이어그램 패널 stub.

원본 25y_automatic_lifter-master/depth_cam 에 ui/diagram.py 가 없어서
main_rec.py:23 의 `from ui.diagram import draw_fsm_diagram_panel` 이
ImportError를 일으킨다. 시연 시각화는 핵심이 아니므로 stub로 대체.

원하면 나중에 실제 그래프(예: graphviz, matplotlib)로 교체 가능.
지금은 현재 FSM 상태/서브상태를 텍스트로만 표시하는 검은 패널.
"""
from __future__ import annotations

import cv2
import numpy as np

_BG = (30, 30, 30)
_EDGE = (70, 70, 70)
_TEXT_WHITE = (240, 240, 240)
_TEXT_DIM = (160, 160, 160)
_TEXT_HI = (0, 220, 0)
_TEXT_WARN = (0, 165, 255)


def draw_fsm_diagram_panel(fsm, panel_size=(480, 720)):
    """현재 FSM 상태를 검은 패널에 텍스트로 표시.

    Args:
        fsm: CalibrationFSM 인스턴스. .state, .align.sub, .recover.sub 사용.
        panel_size: (height, width)

    Returns:
        (H, W, 3) BGR uint8 패널.
    """
    h, w = panel_size
    panel = np.full((h, w, 3), _BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), _EDGE, 1)

    def put(y, text, color=_TEXT_WHITE, scale=0.5, thick=1):
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    state = getattr(fsm, "state", "?")
    align_sub = getattr(getattr(fsm, "align", None), "sub", None) or "-"
    recover_sub = getattr(getattr(fsm, "recover", None), "sub", None) or "-"

    y = 28
    put(y, "FSM STATE", _TEXT_HI, 0.6, 2); y += 32
    put(y, f"  {state}", _TEXT_WHITE, 0.7, 2); y += 36

    put(y, "ALIGN sub", _TEXT_DIM, 0.5, 1); y += 22
    put(y, f"  {align_sub}", _TEXT_WHITE, 0.55, 1); y += 30

    put(y, "RECOVER sub", _TEXT_DIM, 0.5, 1); y += 22
    put(y, f"  {recover_sub}", _TEXT_WHITE, 0.55, 1); y += 30

    # cmd_status
    cmd_status = getattr(fsm, "cmd_status", None)
    if cmd_status is not None:
        code = getattr(cmd_status, "code", "-")
        label = getattr(cmd_status, "label", "-")
        y += 8
        put(y, "Current CMD", _TEXT_DIM, 0.5, 1); y += 22
        put(y, f"  {code} ({label})", _TEXT_WARN, 0.55, 1); y += 30

        try:
            progress = cmd_status.progress_text()
            if progress:
                put(y, f"  {progress}", _TEXT_DIM, 0.45, 1); y += 24
        except Exception:
            pass

    # 하단 가이드 텍스트
    foot_y = h - 60
    put(foot_y, "(stub) FSM diagram", _TEXT_DIM, 0.4, 1)
    put(foot_y + 18, "Replace ui/diagram.py for full viz", _TEXT_DIM, 0.4, 1)

    return panel
