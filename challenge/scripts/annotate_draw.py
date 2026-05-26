"""annotate.py — 시각화 모듈.

상수: KP_NAMES / KP_COLORS / CUBOID_EDGES / PANEL_W
함수:
  draw_overlay(img, kps_2d, active, pose)        : 이미지 위 cuboid wireframe + 점
  draw_line_input(img, line_pts, mouse, zoom, pan): TWO-LINE 입력 진행 표시
  build_panel(h, active, kps_2d, pose, ...)       : 우측 키 안내 + 상태 패널
  render(state, frame_idx, total, frame_name)     : 전체 화면 합성 (image + zoom + overlay + panel)
"""
from __future__ import annotations
import numpy as np
import cv2

from annotate_pnp import PALLET_DIMS


# Camera-facing convention (2026-05-22):
#   0~3 = 카메라에 가까운 near face (운용 시 = fork pocket 면)
#   4~7 = 반대편 far face
# 사용자가 "보이는 면" 에 0~3 클릭. 학습/추론 둘 다 동일 컨벤션.
KP_NAMES = [
    "NearTopLeft",     "NearTopRight",    "NearBottomRight",  "NearBottomLeft",
    "FarTopLeft",      "FarTopRight",     "FarBottomRight",   "FarBottomLeft",
    "Centroid",
]

# 색상 — 앞면(0~3) 따뜻한 색, 뒷면(4~7) 차가운 색, centroid 흰색
KP_COLORS = [
    (0,   0, 255),   # 0 red
    (0, 128, 255),   # 1 orange
    (0, 255, 255),   # 2 yellow
    (0, 255,   0),   # 3 green   (앞면 4개)
    (255, 255,   0), # 4 cyan
    (255,   0,   0), # 5 blue
    (255,   0, 128), # 6 magenta
    (128,  0, 255),  # 7 purple
    (255, 255, 255), # 8 white centroid
]

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # 앞면 — 두껍게
    (4, 5), (5, 6), (6, 7), (7, 4),   # 뒷면
    (0, 4), (1, 5), (2, 6), (3, 7),   # 수직
]

PANEL_W = 280  # 우측 키 안내 패널 폭


def draw_line_input(img, line_pts, mouse_xy, zoom, pan):
    """진행 중인 TWO-LINE input 을 image 위에 그린다."""
    if not line_pts:
        if mouse_xy is not None:
            mu = (mouse_xy[0] / zoom) + pan[0]
            mv = (mouse_xy[1] / zoom) + pan[1]
            cv2.drawMarker(img, (int(mu), int(mv)), (0, 255, 255),
                           cv2.MARKER_CROSS, 14, 1)
        return
    colors = [(0, 255, 255), (0, 255, 255), (0, 200, 255), (0, 200, 255)]
    for i, p in enumerate(line_pts):
        cv2.circle(img, (int(p[0]), int(p[1])), 4, colors[i], -1)
        cv2.putText(img, f"L{i//2+1}-{['A','B'][i%2]}",
                    (int(p[0]) + 6, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    if len(line_pts) >= 2:
        cv2.line(img, (int(line_pts[0][0]), int(line_pts[0][1])),
                       (int(line_pts[1][0]), int(line_pts[1][1])),
                       (0, 255, 255), 2, cv2.LINE_AA)
    if len(line_pts) >= 4:
        cv2.line(img, (int(line_pts[2][0]), int(line_pts[2][1])),
                       (int(line_pts[3][0]), int(line_pts[3][1])),
                       (0, 200, 255), 2, cv2.LINE_AA)
    if mouse_xy is not None and len(line_pts) in (1, 3):
        mu = (mouse_xy[0] / zoom) + pan[0]
        mv = (mouse_xy[1] / zoom) + pan[1]
        col = (0, 255, 255) if len(line_pts) == 1 else (0, 200, 255)
        last = line_pts[-1]
        cv2.line(img, (int(last[0]), int(last[1])), (int(mu), int(mv)),
                 col, 1, cv2.LINE_AA)


def draw_overlay(img, kps_2d, active_idx, pose=None, extrap_mask=None):
    """이미지에 cuboid wireframe + 사용자 클릭 점 그리기.

    v6 컨벤션 경고: pose.v4_warning=True 시 화면 상단에 빨간 경고 표시.
      - _v6_lr_viol / _v6_tb_viol / _v6_fr_viol: pose pair-wise invariant 위반 카운트
      - _v6_click_lr_viol / _v6_click_tb_viol: 사용자 클릭 LR/TB pair 부등호 위반

    v7: extrap_mask 가 주어지면 외삽 점은 outlined (속 빈 원) 으로 표시 →
    직접 click 과 시각 구분.
    """
    vis = img.copy()
    if pose is not None:
        proj = pose["projected_all"]
        # v7: project_3d sentinel = (-1, -1) — 그 외 음수 u/v 는 valid (image 밖).
        pts = [(int(p[0]), int(p[1])) if not (p[0] == -1.0 and p[1] == -1.0)
               else None for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            if pts[a] and pts[b]:
                col = (0, 220, 0) if k < 4 else (0, 160, 0)
                thick = 3 if k < 4 else 1
                cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
    for i, p in enumerate(kps_2d):
        if p is None:
            continue
        c = (int(p[0]), int(p[1]))
        r = 7 if i == active_idx else 5
        is_extrap = (extrap_mask is not None and i < len(extrap_mask)
                     and extrap_mask[i])
        if is_extrap:
            # 외삽 점: 속 빈 원 (두꺼운 외곽선 + 작은 중심 점)
            cv2.circle(vis, c, r, KP_COLORS[i], 2)
            cv2.circle(vis, c, 1, KP_COLORS[i], -1)
            cv2.circle(vis, c, r + 2, (0, 0, 0), 1)
            cv2.putText(vis, f"{i}*", (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)
        else:
            cv2.circle(vis, c, r, KP_COLORS[i], -1)
            cv2.circle(vis, c, r + 2, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)
    return vis


def _pose_dim_short(pose):
    if pose is None:
        return ""
    d = pose.get("dims", PALLET_DIMS)
    return f"front={d[0]*100:.0f}cm"


def build_panel(h, active_idx, kps_2d, pose, frame_idx, total, zoom, dirty,
                mode="click", trans_step=0.02, rot_step=5.0):
    """우측 키 안내 + 현재 상태 패널."""
    panel = np.full((h, PANEL_W, 3), 25, dtype=np.uint8)

    def put(y, text, color=(220, 220, 220), scale=0.42, thick=1):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    y = 18
    mode_color = (0, 255, 0) if mode == "click" else (0, 200, 255)
    put(y, f"MODE: {mode.upper()}  [m=toggle]", mode_color, 0.5, 2); y += 22

    if mode == "click":
        put(y, "KEYBOARD - CLICK", (255, 255, 0), 0.5, 1); y += 22
        put(y, "L click  set point", (200, 200, 200)); y += 16
        put(y, "R click  delete", (200, 200, 200));    y += 16
        put(y, "0-8      select idx", (200, 200, 200)); y += 16
        put(y, "d        delete kp[active]", (200, 200, 200)); y += 16
        put(y, "t        TWO-LINE input *", (0, 255, 255)); y += 16
        put(y, "x        parallelogram extrap *", (0, 255, 255)); y += 16
        put(y, "  (* = extrap, PnP weight 0.3)", (140, 200, 255), 0.36); y += 14
        put(y, "s        save+next", (0, 255, 0));      y += 16
        put(y, "f        near-only save+next", (0, 255, 0)); y += 16
        put(y, "g        auto-fill save (4+pts)", (0, 255, 0)); y += 16
        put(y, "n / p    next / prev", (200, 200, 200)); y += 16
        put(y, ", / .    -10 / +10",   (200, 200, 200)); y += 16
        put(y, "c        centroid auto", (200, 200, 200)); y += 16
        put(y, "z        undo last",   (200, 200, 200)); y += 16
        put(y, "r        reset all",   (200, 200, 200)); y += 16
        put(y, "+ / -    zoom in/out", (200, 200, 200)); y += 16
        put(y, "h j k l  pan (vim)",   (200, 200, 200)); y += 16
        put(y, "q        quit",        (180, 180, 180)); y += 22
    else:
        put(y, "KEYBOARD - MANIPULATE", (255, 255, 0), 0.5, 1); y += 22
        put(y, "translate (camera frame)", (160, 200, 255), 0.42); y += 16
        put(y, "  w/x   up/down  (Y)",   (200, 200, 200)); y += 16
        put(y, "  a/d   left/right (X)", (200, 200, 200)); y += 16
        put(y, "  q/e   near/far  (Z)",  (200, 200, 200)); y += 16
        put(y, "rotate (pallet local)",  (160, 200, 255), 0.42); y += 16
        put(y, "  j/l   yaw -/+",   (200, 200, 200)); y += 16
        put(y, "  i/k   pitch -/+", (200, 200, 200)); y += 16
        put(y, "  u/o   roll -/+",  (200, 200, 200)); y += 16
        put(y, "step", (160, 200, 255), 0.42); y += 16
        put(y, f"  1/2  trans x/2 x2 ({trans_step*100:.1f}cm)", (200, 200, 200)); y += 16
        put(y, f"  3/4  rot x/2 x2  ({rot_step:.1f}\xb0)", (200, 200, 200)); y += 16
        put(y, "save / quit", (160, 200, 255), 0.42); y += 16
        put(y, "  S    save+next", (0, 255, 0)); y += 16
        put(y, "  m    back to CLICK", (200, 200, 200)); y += 16
        put(y, "  Q    quit", (180, 180, 180)); y += 22

    put(y, "KEYPOINTS", (255, 255, 0), 0.5, 1); y += 22
    n_set = sum(1 for k in kps_2d if k is not None)
    put(y, f"set: {n_set}/9", (200, 200, 200)); y += 16
    for i in range(9):
        col = KP_COLORS[i]
        mark = ">" if i == active_idx else " "
        done = "[x]" if kps_2d[i] is not None else "[ ]"
        put(y, f"{mark} {i} {done} {KP_NAMES[i][:14]}", col, 0.4); y += 15
    y += 8

    put(y, "STATUS", (255, 255, 0), 0.5, 1); y += 22
    put(y, f"frame {frame_idx+1}/{total}", (200, 200, 200)); y += 16
    put(y, f"zoom x{zoom:.1f}", (200, 200, 200)); y += 16
    if dirty:
        put(y, "*UNSAVED*", (0, 0, 255), 0.5, 2); y += 18
    if pose is not None:
        err = pose["reproj_error_px"]
        col = (0, 255, 0) if err < 5 else (0, 200, 255) if err < 10 else (0, 0, 255)
        put(y, f"reproj {err:.2f}px", col, 0.5, 2); y += 18
        dim_text = _pose_dim_short(pose)
        if dim_text:
            put(y, dim_text, (180, 220, 255), 0.45); y += 16
    return panel


def render(state, frame_idx, total_frames, frame_name):
    """State → 화면 합성 (image + zoom + overlay + 우측 패널)."""
    vis = draw_overlay(state.img, state.kps_2d, state.active, state.pose,
                       extrap_mask=getattr(state, "extrap_mask", None))
    if state.mode == "click" and state.line_mode:
        draw_line_input(vis, state.line_pts or [], state.last_mouse, state.zoom, state.pan)
    h, w = vis.shape[:2]
    if state.zoom > 1.001:
        crop_w = int(w / state.zoom)
        crop_h = int(h / state.zoom)
        state.pan[0] = max(0, min(w - crop_w, state.pan[0]))
        state.pan[1] = max(0, min(h - crop_h, state.pan[1]))
        crop = vis[state.pan[1]:state.pan[1] + crop_h,
                   state.pan[0]:state.pan[0] + crop_w]
        vis = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    name = KP_NAMES[state.active]
    col = KP_COLORS[state.active]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.3, overlay, 0.7, 0)
    cv2.putText(vis, f"Click #{state.active}: {name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(vis, frame_name[:20], (w - 220, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # v6 컨벤션 critical 경고 (zoom 후에도 항상 보이도록 상단 bar 아래에 표시).
    # fix v6 strict invariants (LR/TB/FR pair) 위반 또는 사용자 click LR/TB 모순.
    if state.pose is not None and state.pose.get("v4_warning"):
        msgs = []
        lrv = state.pose.get("_v6_lr_viol", 0)
        tbv = state.pose.get("_v6_tb_viol", 0)
        frv = state.pose.get("_v6_fr_viol", 0)
        if lrv > 0:
            msgs.append(f"LR-viol {lrv}/4")
        if tbv > 0:
            msgs.append(f"TB-viol {tbv}/4")
        if frv > 0:
            msgs.append(f"FR-viol {frv}/4")
        clrv = state.pose.get("_v6_click_lr_viol", 0)
        ctbv = state.pose.get("_v6_click_tb_viol", 0)
        if clrv > 0:
            msgs.append(f"CLICK-LR {clrv}/4")
        if ctbv > 0:
            msgs.append(f"CLICK-TB {ctbv}/4")
        warn_msg = "[v6] " + " | ".join(msgs) if msgs else "[v6] convention violation"
        cv2.rectangle(vis, (0, 30), (w, 52), (0, 0, 0), -1)
        cv2.putText(vis, warn_msg, (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2, cv2.LINE_AA)
    if state.mode == "manip":
        cv2.rectangle(vis, (1, 1), (w - 2, h - 2), (255, 180, 0), 3)
    panel = build_panel(h, state.active, state.kps_2d, state.pose,
                        frame_idx, total_frames, state.zoom, state.dirty,
                        mode=state.mode, trans_step=state.trans_step,
                        rot_step=state.rot_step_deg)
    return np.hstack([vis, panel])
