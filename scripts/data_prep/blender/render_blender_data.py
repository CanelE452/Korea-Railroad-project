"""Blender 합성 데이터 렌더링 메인 스크립트.

사용법:
    # Blender GUI에서 실행 (MCP execute_blender_code)
    # 또는 커맨드라인:
    blender synth_data_scene.blend --background --python render_blender_data.py
"""

import json
import os
import sys

import bpy
import numpy as np

# 이 파일의 디렉토리를 sys.path에 추가 (모듈 import 용)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from blender_config import (
    NUM_FRAMES, IMAGE_WIDTH, IMAGE_HEIGHT, K,
    OUTPUT_DIR, OVERLAY_DIR,
    ORIENTATION_OVERRIDES, CANONICAL_BBOX_MIN, CANONICAL_BBOX_MAX,
    PALLET_SURFACE_Z, CUBOID_EDGES, CORNER_COLORS_RGB, PALLET_NAMES,
)
from blender_math import (
    euler_to_rotation_matrix, rotation_matrix_to_quat_xyzw,
    rotation_matrix_to_euler_deg, build_view_matrix,
    canonical_corners_yup, yup_to_zup,
)
from randomizers import (
    get_obj, setup_render,
    randomize_pallet, randomize_camera, randomize_distractors,
    randomize_boxes, randomize_hdri,
)


def _get_pallet_world_bbox(pallet_obj):
    """Get pallet's world AABB from all child meshes."""
    from mathutils import Vector as V
    all_corners = []
    for child in pallet_obj.children_recursive:
        if child.type == 'MESH' and child.bound_box:
            for c in child.bound_box:
                all_corners.append(child.matrix_world @ V(c))
    if not all_corners:
        return None, None
    xs = [c.x for c in all_corners]
    ys = [c.y for c in all_corners]
    zs = [c.z for c in all_corners]
    return np.array([min(xs), min(ys), min(zs)]), np.array([max(xs), max(ys), max(zs)])


def compute_annotation(pallet_name, pallet_obj, cam_pos, look_at):
    """Compute NDDS annotation directly from Blender mesh bounding box."""
    import bpy
    bpy.context.view_layer.update()

    bbox_min, bbox_max = _get_pallet_world_bbox(pallet_obj)
    if bbox_min is None:
        return {}, np.zeros((9, 2)), 0.0, (0, 0, 0)

    # 8 DOPE corners from world AABB (Z=UP in Blender)
    # DOPE order: Front=max_Y side, Rear=min_Y side, Top=max_Z, Bottom=min_Z
    mn, mx = bbox_min, bbox_max
    corners_world = np.array([
        [mn[0], mx[1], mx[2]],  # 0 FrontTopRight
        [mx[0], mx[1], mx[2]],  # 1 FrontTopLeft
        [mx[0], mn[1], mx[2]],  # 2 RearTopLeft  (swapped for Blender Y axis)
        [mn[0], mn[1], mx[2]],  # 3 RearTopRight
        [mn[0], mx[1], mn[2]],  # 4 FrontBottomRight
        [mx[0], mx[1], mn[2]],  # 5 FrontBottomLeft
        [mx[0], mn[1], mn[2]],  # 6 RearBottomLeft
        [mn[0], mn[1], mn[2]],  # 7 RearBottomRight
    ])
    centroid_world = (bbox_min + bbox_max) / 2.0
    points_3d = np.vstack([corners_world, centroid_world[np.newaxis, :]])

    # Project to camera
    R_w2c, t_w2c = build_view_matrix(cam_pos, look_at, up=(0, 0, 1))
    pts_cam = (R_w2c @ points_3d.T).T + t_w2c
    projected = (K @ pts_cam.T).T
    uv = projected[:, :2] / projected[:, 2:3]

    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_WIDTH) &
                (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_HEIGHT))
    visibility = float(in_frame.sum()) / 9.0

    # Pose: object frame = pallet world frame relative to camera
    t_obj_cam = R_w2c @ centroid_world + t_w2c
    # Rotation: pallet is axis-aligned box, rotation = just the Z rotation
    pallet_z_rot = pallet_obj.rotation_euler[2]
    cz, sz = np.cos(pallet_z_rot), np.sin(pallet_z_rot)
    R_obj_world = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_obj_cam = R_w2c @ R_obj_world

    quat_xyzw = rotation_matrix_to_quat_xyzw(R_obj_cam)
    pitch, yaw, roll = rotation_matrix_to_euler_deg(R_obj_cam)

    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = R_obj_cam
    pose_4x4[:3, 3] = t_obj_cam

    data = {
        "camera_data": {
            "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
            "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                           "cx": float(K[0, 2]), "cy": float(K[1, 2])},
            "location_worldframe": [float(v) for v in cam_pos],
        },
        "objects": [{
            "class": "pallet", "name": pallet_name,
            "visibility": visibility,
            "location": [float(v) for v in t_obj_cam],
            "quaternion_xyzw": quat_xyzw,
            "euler_angles": {"pitch": pitch, "yaw": yaw, "roll": roll},
            "pose_transform": pose_4x4.tolist(),
            "projected_cuboid_centroid": [float(uv[8, 0]), float(uv[8, 1])],
            "projected_cuboid": [[float(uv[k, 0]), float(uv[k, 1])] for k in range(8)],
            "cuboid": [[float(points_3d[k, j]) for j in range(3)] for k in range(8)],
        }],
    }
    return data, uv, visibility, (pitch, yaw, roll)


def draw_overlay(render_path, overlay_path, uv, visibility, rot_info):
    """Draw keypoint overlay using PIL."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] PIL not available, skipping overlay")
        return

    img = Image.open(render_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, j in CUBOID_EDGES:
        x0, y0 = int(uv[i, 0]), int(uv[i, 1])
        x1, y1 = int(uv[j, 0]), int(uv[j, 1])
        if (0 <= x0 < IMAGE_WIDTH and 0 <= y0 < IMAGE_HEIGHT) or \
           (0 <= x1 < IMAGE_WIDTH and 0 <= y1 < IMAGE_HEIGHT):
            draw.line([(x0, y0), (x1, y1)], fill=(0, 255, 0), width=2)

    for i in range(8):
        cx, cy = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= cx < IMAGE_WIDTH and 0 <= cy < IMAGE_HEIGHT:
            r = 5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=CORNER_COLORS_RGB[i], outline=(0, 0, 0))
            draw.text((cx + 8, cy - 8), str(i), fill=(255, 255, 255))

    cent_x, cent_y = int(uv[8, 0]), int(uv[8, 1])
    if 0 <= cent_x < IMAGE_WIDTH and 0 <= cent_y < IMAGE_HEIGHT:
        r = 7
        draw.ellipse([cent_x - r, cent_y - r, cent_x + r, cent_y + r],
                     fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((cent_x + 10, cent_y - 8), "C", fill=(255, 255, 255))

    pitch, yaw, roll = rot_info
    y_pos = 10
    for line in [f"vis: {visibility:.2f}", f"pitch: {pitch:.1f}",
                 f"yaw: {yaw:.1f}", f"roll: {roll:.1f}"]:
        draw.text((10, y_pos), line, fill=(0, 255, 255))
        y_pos += 18

    img.save(overlay_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    setup_render()
    scene = bpy.context.scene

    print(f"[render] Rendering {NUM_FRAMES} frames to {OUTPUT_DIR}")

    for frame_idx in range(NUM_FRAMES):
        print(f"\n--- Frame {frame_idx:06d} ---")

        MAX_RETRIES = 5
        vis_pre = 0.0
        pallet_name = pallet_pos = cam_pos = look_at = None

        for retry in range(MAX_RETRIES):
            pallet_name, _ = randomize_pallet()
            pallet_obj = get_obj(pallet_name)
            pallet_pos = tuple(pallet_obj.location)

            cam_pos, look_at = randomize_camera(pallet_pos)
            if cam_pos is None:
                break

            randomize_distractors(pallet_pos, cam_pos)
            randomize_boxes(pallet_obj, pallet_name)
            randomize_hdri()
            bpy.context.view_layer.update()

            _, _, vis_pre, _ = compute_annotation(pallet_name, pallet_obj, cam_pos, look_at)
            if vis_pre >= 0.60:
                break
            print(f"  [RETRY {retry + 1}] vis={vis_pre:.2f} < 0.60")

        if cam_pos is None:
            print("  [SKIP] No camera")
            continue

        print(f"  {pallet_name} at ({pallet_pos[0]:.1f}, {pallet_pos[1]:.1f}), vis={vis_pre:.2f}")

        render_path = os.path.join(OUTPUT_DIR, f"{frame_idx:06d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        annotation, uv, visibility, rot_info = compute_annotation(
            pallet_name, pallet_obj, cam_pos, look_at)

        json_path = os.path.join(OUTPUT_DIR, f"{frame_idx:06d}.json")
        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=2)

        overlay_path = os.path.join(OVERLAY_DIR, f"overlay_{frame_idx:06d}.png")
        draw_overlay(render_path, overlay_path, uv, visibility, rot_info)

        print(f"  Rendered + JSON (vis={visibility:.2f})")

    print(f"\n[render] Done! {NUM_FRAMES} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
