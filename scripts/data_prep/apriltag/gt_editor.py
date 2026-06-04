"""반자동 GT Editor — 자동 GT를 초기값으로, 키보드/클릭으로 미세조정.

흐름:
  1. 자동 GT(`gt/{name}.json`) 로드 → init pose
  2. 수동 보정 결과(`gt_manual/{name}.json`)가 있으면 그걸 우선 로드
  3. 키보드로 6DoF 미세조정
  4. 클릭으로 2D 코너 → solvePnPRefineLM 정밀 정렬
  5. 저장 (gt_manual + overlay)
  6. 다음 프레임은 이전 보정 결과를 init으로 propagate

조작 (image 창 활성):
  navigation
    n / p     : 다음 / 이전 프레임
  translation (camera frame, meters)
    a / d     : -X / +X
    w / s     : -Y / +Y
    q / e     : -Z / +Z
  rotation (pallet local frame, degrees)
    j / l     : yaw -  / yaw +   (pallet Y)
    i / k     : pitch- / pitch+  (pallet X)
    u / o     : roll - / roll +  (pallet Z)
  step size
    1 / 2     : trans step / *2
    3 / 4     : rot step / *2
  click refine
    c         : 클릭 모드 토글
    [click]   : 가장 가까운 cuboid corner로 자동 매칭
    z         : 마지막 클릭 취소
    x         : 모든 클릭 제거
    r         : solvePnPRefineLM 적용 (>=4 점 필요)
  save / misc
    SPACE     : gt_manual/ 에 저장
    R (shift) : 자동 GT로 reset
    m         : propagate 토글 (다음 프레임 init으로 현재 결과 사용)
    ESC       : 종료

사용법:
    python scripts/data_prep/gt_editor.py \
        --capture data/pallet/raw_data/capture0403middle \
        [--start <timestamp>]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d_isaac


PALLET_DIMS = (1.10, 1.30, 0.11)
KP3D = make_pallet_keypoints_3d_isaac(width=PALLET_DIMS[0], depth=PALLET_DIMS[1], height=PALLET_DIMS[2])  # (9,3) Isaac canonical ordering
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
KP_COLORS = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,128),(128,0,255)]


def load_K(path):
    K = np.loadtxt(path)
    return K


def project(T_cp, K):
    """Returns (uv (N,2) float, valid (N,) bool). valid=False when behind camera."""
    R, t = T_cp[:3,:3], T_cp[:3,3]
    pts = (R @ KP3D.T).T + t
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if p[2] > 0:
            uv[i, 0] = fx * p[0] / p[2] + cx
            uv[i, 1] = fy * p[1] / p[2] + cy
            valid[i] = True
    return uv, valid


def draw_overlay(img, T_cp, K, hud_lines, clicks=None, click_mode=False):
    vis = img.copy()
    uv, valid = project(T_cp, K)
    pts = [(int(p[0]), int(p[1])) if v else None for p, v in zip(uv, valid)]

    # cuboid edges (thin lines so corners stay visible)
    # valid = both endpoints in front of camera; cv2.line clips to image automatically
    for i, j in EDGES:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 1, cv2.LINE_AA)

    # corner keypoints with index labels (drawn on top of lines)
    h, w = img.shape[:2]
    for idx in range(8):
        pt = pts[idx]
        if pt is None:
            continue
        # Only draw circle/label if within image bounds
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(vis, pt, 4, (0, 0, 0), -1)
            cv2.circle(vis, pt, 3, KP_COLORS[idx], -1)
            cv2.putText(vis, str(idx), (pt[0]+7, pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, str(idx), (pt[0]+7, pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[idx], 1)

    # centroid
    if pts[8] is not None and 0 <= pts[8][0] < w and 0 <= pts[8][1] < h:
        cv2.circle(vis, pts[8], 7, (255, 255, 255), -1)
        cv2.putText(vis, "C", (pts[8][0]+6, pts[8][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # axes (X red, Y green, Z blue) at centroid
    axis_len = 0.2
    R = T_cp[:3,:3]; t = T_cp[:3,3]
    centroid_3d = KP3D[8]
    origin_cam = R @ centroid_3d + t
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    def proj_cam(pc):
        if pc[2] <= 0: return None
        return (int(fx*pc[0]/pc[2]+cx), int(fy*pc[1]/pc[2]+cy))
    o2 = proj_cam(origin_cam)
    if o2:
        for axis_idx, color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
            ax_3d = np.zeros(3); ax_3d[axis_idx] = axis_len
            ax_cam = R @ (centroid_3d + ax_3d) + t
            p2 = proj_cam(ax_cam)
            if p2:
                cv2.arrowedLine(vis, o2, p2, color, 2, tipLength=0.2)

    # clicks
    if clicks is not None:
        for kp_idx, px, py in clicks:
            cv2.circle(vis, (int(px), int(py)), 8, (0, 255, 255), 2)
            cv2.putText(vis, f"k{kp_idx}", (int(px)+10, int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # HUD
    bar_h = 22 * len(hud_lines) + 10
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (0, 0, 0), -1)
    for i, line in enumerate(hud_lines):
        color = (0, 255, 255) if click_mode and i == 0 else (255, 255, 255)
        cv2.putText(vis, line, (8, 18 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def rodrigues(axis, angle_rad):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def apply_translation_cam(T, dx, dy, dz):
    T_new = T.copy()
    T_new[:3, 3] += np.array([dx, dy, dz])
    return T_new


def apply_rotation_local(T, axis_local, angle_deg):
    R_delta = rodrigues(axis_local, np.deg2rad(angle_deg))
    T_new = T.copy()
    T_new[:3, :3] = T[:3, :3] @ R_delta
    return T_new


def load_pose_from_json(path):
    with open(path) as f:
        d = json.load(f)
    arr = np.array(d["objects"][0]["pose_transform"], float)
    return arr


def save_pose_json(path, T_cp, image_shape, K):
    h, w = image_shape[:2]
    uv, valid = project(T_cp, K)
    # NaN → sentinel -1 for invalid corners (JSON compatible)
    uv_out = np.where(valid[:, None], uv, -1.0)
    cuboid = uv_out[:8].tolist()
    centroid = uv_out[8].tolist()
    ann = {
        "camera_data": {
            "width": w, "height": h,
            "intrinsics": {"fx": float(K[0,0]), "fy": float(K[1,1]),
                           "cx": float(K[0,2]), "cy": float(K[1,2])},
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "pose_transform": T_cp.tolist(),
            "projected_cuboid": cuboid,
            "projected_cuboid_centroid": centroid,
            "dimensions_m": {"width": PALLET_DIMS[0],
                             "depth": PALLET_DIMS[1],
                             "height": PALLET_DIMS[2]},
            "gt_source": "manual_refined",
        }],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ann, f, indent=2)


def pnp_refine(T_init, K, clicks):
    """clicks = [(kp_idx, px, py), ...]

    >=4 clicks: fresh solvePnP (EPNP/ITERATIVE) — 초기값 무시, 바닥부터 풀이
    >=6 clicks: +RefineLM으로 polishing
    """
    if len(clicks) < 4:
        return T_init, "need >=4 points"
    obj_pts = np.array([KP3D[k] for k, _, _ in clicks], dtype=np.float64)
    img_pts = np.array([[px, py] for _, px, py in clicks], dtype=np.float64)
    try:
        # Fresh solve (no init) using EPNP if >=4 points
        flag = cv2.SOLVEPNP_EPNP if len(clicks) >= 4 else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, np.zeros(5),
            flags=flag,
        )
        if not ok:
            return T_init, "solvePnP failed"
        # Polish with iterative refine if enough points
        if len(clicks) >= 4:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts, img_pts, K, np.zeros(5), rvec, tvec,
            )
        R_new, _ = cv2.Rodrigues(rvec)
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = tvec.flatten()
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, np.zeros(5))
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean()
        return T_new, f"solved ({len(clicks)} pts), reproj={err:.1f}px"
    except Exception as e:
        return T_init, f"error: {e}"


def nearest_kp_idx(uv, valid, click_xy):
    dists = np.full(8, np.inf)
    cp = np.array(click_xy, dtype=np.float64)
    for i in range(8):
        if valid[i]:
            dists[i] = np.linalg.norm(uv[i] - cp)
    return int(dists.argmin()), float(dists.min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True, help="capture root dir")
    p.add_argument("--start", default=None, help="start frame timestamp (without .png)")
    p.add_argument("--worklist", default=None,
                   help="worklist .txt (one frame name per line). 자동 탐색: capture/gt_editor_worklist.txt")
    args = p.parse_args()

    rgb_dir = os.path.join(args.capture, "rgb")
    auto_gt_dir = os.path.join(args.capture, "gt")
    manual_gt_dir = os.path.join(args.capture, "gt_manual")
    manual_overlay_dir = os.path.join(args.capture, "gt_manual_overlay")
    K = load_K(os.path.join(args.capture, "cam_K.txt"))

    all_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith((".png", ".jpg"))])

    # Worklist filtering
    wl_path = args.worklist or os.path.join(args.capture, "gt_editor_worklist.txt")
    if os.path.exists(wl_path):
        with open(wl_path) as wf:
            wanted = [line.strip() for line in wf if line.strip() and not line.startswith("#")]
        wanted_set = set(wanted)
        files = [f for f in all_files if os.path.splitext(f)[0] in wanted_set]
        # Preserve worklist order
        name_to_file = {os.path.splitext(f)[0]: f for f in files}
        files = [name_to_file[n] for n in wanted if n in name_to_file]
        print(f"[worklist] {wl_path} → {len(files)} frames (of {len(all_files)} total)")
    else:
        files = all_files
        print(f"[all] {len(files)} frames (no worklist at {wl_path})")
    names = [os.path.splitext(f)[0] for f in files]
    name2idx = {n: i for i, n in enumerate(names)}

    idx = 0
    if args.start and args.start in name2idx:
        idx = name2idx[args.start]

    cv2.namedWindow("gt_editor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("gt_editor", 1600, 1200)

    state = {
        "T": None,           # current pose
        "T_orig": None,      # auto GT pose for reset
        "trans_step": 0.02,  # meters
        "rot_step": 2.0,     # degrees
        "click_mode": False,
        "clicks": [],        # list of (kp_idx, px, py)
        "pending_kp": None,  # when click_mode: set by 0-7 keys
        "msg": "",
        "propagate": True,
        "last_T": None,      # for propagation
    }

    def load_frame(i):
        name = names[i]
        img_path = os.path.join(rgb_dir, files[i])
        img = cv2.imread(img_path)
        # Try manual first, then auto, then propagate
        manual_path = os.path.join(manual_gt_dir, f"{name}.json")
        auto_path = os.path.join(auto_gt_dir, f"{name}.json")
        T_orig = None
        if os.path.exists(auto_path):
            T_orig = load_pose_from_json(auto_path)
        if os.path.exists(manual_path):
            T = load_pose_from_json(manual_path)
            src = "manual"
        elif T_orig is not None:
            T = T_orig.copy()
            src = "auto"
        elif state["last_T"] is not None and state["propagate"]:
            T = state["last_T"].copy()
            src = "propagated"
        else:
            T = np.eye(4); T[2, 3] = 1.5
            src = "default"
        state["T"] = T
        state["T_orig"] = T_orig if T_orig is not None else T.copy()
        state["clicks"] = []
        state["msg"] = f"loaded ({src})"
        return img, name

    img, name = load_frame(idx)

    def on_mouse(event, x, y, flags, param):
        if not state["click_mode"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["pending_kp"] is not None:
                kp_idx = state["pending_kp"]
                state["pending_kp"] = None
                state["clicks"].append((kp_idx, float(x), float(y)))
                state["msg"] = f"manual kp{kp_idx} at ({x},{y})"
            else:
                uv, valid = project(state["T"], K)
                kp_idx, dist = nearest_kp_idx(uv, valid, (x, y))
                state["clicks"].append((kp_idx, float(x), float(y)))
                state["msg"] = f"nearest kp{kp_idx} at ({x},{y}) dist={dist:.1f}"
    cv2.setMouseCallback("gt_editor", on_mouse)

    while True:
        # Exit cleanly if window was closed with X button
        try:
            if cv2.getWindowProperty("gt_editor", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        click_hint = ""
        if state["click_mode"]:
            click_hint = "CLICK MODE"
            if state["pending_kp"] is not None:
                click_hint += f" [next=kp{state['pending_kp']}]"
            click_hint += f" clicks={len(state['clicks'])}"
        hud = [
            f"[{idx+1}/{len(names)}] {name} {click_hint}",
            f"step T={state['trans_step']:.3f}m R={state['rot_step']:.1f}deg | propagate={'ON' if state['propagate'] else 'off'}",
            state["msg"],
        ]
        vis = draw_overlay(img, state["T"], K, hud, state["clicks"], state["click_mode"])
        cv2.imshow("gt_editor", vis)
        k = cv2.waitKey(0) & 0xFF

        if k == 27:  # ESC
            break

        # navigation
        elif k == ord('n'):
            if state["T"] is not None:
                state["last_T"] = state["T"].copy()
            idx = min(idx + 1, len(names) - 1)
            img, name = load_frame(idx)
        elif k == ord('p'):
            if state["T"] is not None:
                state["last_T"] = state["T"].copy()
            idx = max(idx - 1, 0)
            img, name = load_frame(idx)

        # translation (camera frame)
        elif k == ord('a'):
            state["T"] = apply_translation_cam(state["T"], -state["trans_step"], 0, 0); state["msg"] = "-X"
        elif k == ord('d'):
            state["T"] = apply_translation_cam(state["T"], +state["trans_step"], 0, 0); state["msg"] = "+X"
        elif k == ord('w'):
            state["T"] = apply_translation_cam(state["T"], 0, -state["trans_step"], 0); state["msg"] = "-Y (up)"
        elif k == ord('s'):
            state["T"] = apply_translation_cam(state["T"], 0, +state["trans_step"], 0); state["msg"] = "+Y (down)"
        elif k == ord('q'):
            state["T"] = apply_translation_cam(state["T"], 0, 0, -state["trans_step"]); state["msg"] = "-Z (close)"
        elif k == ord('e'):
            state["T"] = apply_translation_cam(state["T"], 0, 0, +state["trans_step"]); state["msg"] = "+Z (far)"

        # rotation (pallet local axes)
        elif k == ord('j'):
            state["T"] = apply_rotation_local(state["T"], [0,1,0], -state["rot_step"]); state["msg"] = "yaw-"
        elif k == ord('l'):
            state["T"] = apply_rotation_local(state["T"], [0,1,0], +state["rot_step"]); state["msg"] = "yaw+"
        elif k == ord('i'):
            state["T"] = apply_rotation_local(state["T"], [1,0,0], -state["rot_step"]); state["msg"] = "pitch-"
        elif k == ord('k'):
            state["T"] = apply_rotation_local(state["T"], [1,0,0], +state["rot_step"]); state["msg"] = "pitch+"
        elif k == ord('u'):
            state["T"] = apply_rotation_local(state["T"], [0,0,1], -state["rot_step"]); state["msg"] = "roll-"
        elif k == ord('o'):
            state["T"] = apply_rotation_local(state["T"], [0,0,1], +state["rot_step"]); state["msg"] = "roll+"

        # digit keys: in click mode = kp select; else = step size
        elif k == ord('0') and state["click_mode"]:
            state["pending_kp"] = 0; state["msg"] = "next click = kp0"
        elif k == ord('5') and state["click_mode"]:
            state["pending_kp"] = 4; state["msg"] = "next click = kp4"
        elif k == ord('6') and state["click_mode"]:
            state["pending_kp"] = 5; state["msg"] = "next click = kp5"
        elif k == ord('7') and state["click_mode"]:
            state["pending_kp"] = 6; state["msg"] = "next click = kp6"
        elif k == ord('8') and state["click_mode"]:
            state["pending_kp"] = 7; state["msg"] = "next click = kp7"
        elif k == ord('1'):
            if state["click_mode"]:
                state["pending_kp"] = 0; state["msg"] = "next click = kp0"
            else:
                state["trans_step"] = max(0.001, state["trans_step"] * 0.5); state["msg"] = f"Tstep={state['trans_step']:.4f}"
        elif k == ord('2'):
            if state["click_mode"]:
                state["pending_kp"] = 1; state["msg"] = "next click = kp1"
            else:
                state["trans_step"] = min(0.5, state["trans_step"] * 2); state["msg"] = f"Tstep={state['trans_step']:.4f}"
        elif k == ord('3'):
            if state["click_mode"]:
                state["pending_kp"] = 2; state["msg"] = "next click = kp2"
            else:
                state["rot_step"] = max(0.1, state["rot_step"] * 0.5); state["msg"] = f"Rstep={state['rot_step']:.2f}"
        elif k == ord('4'):
            if state["click_mode"]:
                state["pending_kp"] = 3; state["msg"] = "next click = kp3"
            else:
                state["rot_step"] = min(45, state["rot_step"] * 2); state["msg"] = f"Rstep={state['rot_step']:.2f}"

        # click / refine
        elif k == ord('c'):
            state["click_mode"] = not state["click_mode"]
            state["msg"] = "click mode " + ("ON" if state["click_mode"] else "off")
        elif k == ord('z'):
            if state["clicks"]:
                state["clicks"].pop()
                state["msg"] = f"undo, {len(state['clicks'])} left"
        elif k == ord('x'):
            state["clicks"] = []
            state["msg"] = "cleared clicks"
        elif k == ord('r'):
            T_new, msg = pnp_refine(state["T"], K, state["clicks"])
            state["T"] = T_new
            state["msg"] = msg

        # save / reset / misc
        elif k == 32:  # SPACE
            save_path = os.path.join(manual_gt_dir, f"{name}.json")
            save_pose_json(save_path, state["T"], img.shape, K)
            ov_path = os.path.join(manual_overlay_dir, f"{name}.jpg")
            os.makedirs(os.path.dirname(ov_path), exist_ok=True)
            cv2.imwrite(ov_path, vis)
            state["msg"] = f"SAVED {name}"
        elif k == ord('R'):
            state["T"] = state["T_orig"].copy()
            state["msg"] = "reset to auto"
        elif k == ord('m'):
            state["propagate"] = not state["propagate"]
            state["msg"] = "propagate " + ("ON" if state["propagate"] else "off")
        elif k == ord('f'):
            # Toggle fullscreen
            prop = cv2.getWindowProperty("gt_editor", cv2.WND_PROP_FULLSCREEN)
            new_prop = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty("gt_editor", cv2.WND_PROP_FULLSCREEN, new_prop)
            state["msg"] = "fullscreen " + ("on" if new_prop == cv2.WINDOW_FULLSCREEN else "off")
        else:
            if k != 255:  # ignore no-key
                state["msg"] = f"key={k} (no action)"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
