"""씬 랜덤화 함수: 팔레트, 디스트랙터, 박스, 카메라, HDRI."""

import math
import random

import bpy
from mathutils import Vector as mathutils_Vector

from blender_config import (
    PALLET_NAMES, DISTRACTOR_NAMES, BOX_NAMES,
    PALLET_SURFACE_Z, PALLET_TOP_Z, HDRI_BASE_STRENGTH, FX, IMAGE_WIDTH,
)


def get_obj(name):
    return bpy.data.objects.get(name)


def get_obj_aabb_world(obj):
    corners = [obj.matrix_world @ mathutils_Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def aabb_overlap_xy(a_min, a_max, b_min, b_max):
    return not (a_max[0] < b_min[0] or b_max[0] < a_min[0] or
                a_max[1] < b_min[1] or b_max[1] < a_min[1])


def obj_z_min_local(obj):
    bb = obj.bound_box
    return min(v[2] for v in bb)


def obj_height(obj):
    bb = obj.bound_box
    zs = [v[2] for v in bb]
    return max(zs) - min(zs)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium Contrast'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    if hasattr(scene.eevee, 'taa_render_samples'):
        scene.eevee.taa_render_samples = 64

    cam = scene.camera
    if cam and cam.data:
        cam.data.sensor_fit = 'HORIZONTAL'
        cam.data.sensor_width = 36.0
        cam.data.lens = FX * 36.0 / IMAGE_WIDTH
        cam.data.shift_x = 0
        cam.data.shift_y = 0


def randomize_pallet():
    """Select one pallet, hide others, place in open asphalt area."""
    chosen_idx = random.randint(0, len(PALLET_NAMES) - 1)
    chosen_name = PALLET_NAMES[chosen_idx]

    px = random.uniform(-12, -6)
    py = random.uniform(-12, -2)

    for name in PALLET_NAMES:
        obj = get_obj(name)
        if obj is None:
            continue
        if name == chosen_name:
            obj.hide_render = False
            obj.hide_viewport = False
            obj.location = (px, py, 0)
            yaw = random.uniform(0, 2 * math.pi)
            obj.rotation_euler = (0, 0, yaw)
        else:
            obj.hide_render = True
            obj.hide_viewport = True

    return chosen_name, chosen_idx


def randomize_camera(pallet_pos):
    """Lifter view: height 0.3~0.6m, distance 1.5~3.5m, always looks at pallet."""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return None, None

    px, py, pz = pallet_pos

    distance = random.uniform(1.5, 3.0)
    height = random.uniform(0.5, 1.2)  # higher to see pallet top, not just horizon
    h_angle = random.uniform(0, 2 * math.pi)

    cam_x = px + distance * math.cos(h_angle)
    cam_y = py + distance * math.sin(h_angle)
    cam_z = height

    cam.location = (cam_x, cam_y, cam_z)

    look_x, look_y, look_z = px, py, PALLET_SURFACE_Z / 2.0
    look_at = (look_x, look_y, look_z)
    cam_pos = (cam_x, cam_y, cam_z)

    direction = mathutils_Vector(look_at) - mathutils_Vector(cam_pos)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    return cam_pos, look_at


def randomize_distractors(pallet_pos, cam_pos):
    """Place distractors to LEFT/RIGHT of camera-pallet axis."""
    placed_aabbs = []
    for name in PALLET_NAMES + BOX_NAMES:
        obj = get_obj(name)
        if obj and not obj.hide_render and obj.type == 'MESH':
            try:
                placed_aabbs.append(get_obj_aabb_world(obj))
            except Exception:
                pass

    px, py = pallet_pos[0], pallet_pos[1]
    cx, cy = cam_pos[0], cam_pos[1]
    cam_to_pallet_angle = math.atan2(py - cy, px - cx)

    shuffled = list(DISTRACTOR_NAMES)
    random.shuffle(shuffled)
    n_use = random.randint(2, min(5, len(shuffled)))
    used = shuffled[:n_use]
    hidden = shuffled[n_use:]

    for name in used:
        obj = get_obj(name)
        if obj is None:
            continue
        obj.hide_render = False
        obj.hide_viewport = False

        placed = False
        for _ in range(10):
            side = random.choice([-1, 1])
            # ±70~110 degrees from camera direction (strict sides, never front)
            offset_angle = cam_to_pallet_angle + side * random.uniform(math.pi * 0.4, math.pi * 0.6)
            dist = random.uniform(1.5, 3.0)
            x = px + dist * math.cos(offset_angle)
            y = py + dist * math.sin(offset_angle)

            z_offset = -obj_z_min_local(obj) * obj.scale[2]
            obj.location = (x, y, max(z_offset, 0))
            obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))

            bpy.context.view_layer.update()
            a_min, a_max = get_obj_aabb_world(obj)

            overlap = False
            for b_min, b_max in placed_aabbs:
                if aabb_overlap_xy(a_min, a_max, b_min, b_max):
                    overlap = True
                    break
            if not overlap:
                placed_aabbs.append((a_min, a_max))
                placed = True
                break

        if not placed:
            obj.hide_render = True
            obj.hide_viewport = True

    for name in hidden:
        obj = get_obj(name)
        if obj:
            obj.hide_render = True
            obj.hide_viewport = True


def randomize_boxes(pallet_obj, pallet_name=None):
    """Place 3-7 boxes on the pallet in 2x2 grid layers."""
    num_boxes = random.randint(3, 7)
    selected = sorted(random.sample(range(len(BOX_NAMES)), min(num_boxes, len(BOX_NAMES))))

    px, py, pz = pallet_obj.location
    # Use actual measured pallet height, not fixed value
    top_z = PALLET_TOP_Z.get(pallet_name, PALLET_SURFACE_Z) if pallet_name else PALLET_SURFACE_Z
    base_z = pz + top_z

    grid_offsets = [(-0.2, -0.15), (0.2, -0.15), (-0.2, 0.15), (0.2, 0.15)]
    layer = 0
    col = 0
    box_height_approx = 0.34

    for i, name in enumerate(BOX_NAMES):
        obj = get_obj(name)
        if obj is None:
            continue

        if i in selected:
            obj.hide_render = False
            obj.hide_viewport = False

            grid_idx = col % len(grid_offsets)
            dx, dy = grid_offsets[grid_idx]
            z_min_local = obj_z_min_local(obj)
            bz = base_z + layer * box_height_approx - z_min_local * obj.scale[2]

            obj.location = (px + dx, py + dy, bz)
            obj.rotation_euler = (0, 0, random.uniform(-0.05, 0.05))

            col += 1
            if col % len(grid_offsets) == 0:
                layer += 1
        else:
            obj.hide_render = True
            obj.hide_viewport = True


def _collect_hdri_images():
    hdris = []
    for img in bpy.data.images:
        fp = img.filepath.lower()
        if '.hdr' in fp or '.exr' in fp:
            hdris.append(img)
    return hdris


def randomize_hdri():
    """Randomize HDRI image, rotation, and exposure."""
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return

    hdris = _collect_hdri_images()

    for node in world.node_tree.nodes:
        if node.type == 'TEX_ENVIRONMENT' and hdris:
            node.image = random.choice(hdris)
        if node.type == 'MAPPING':
            node.inputs['Rotation'].default_value[2] = random.uniform(0, 2 * math.pi)
        if node.type == 'BACKGROUND':
            exposure = random.uniform(-0.2, 0.2)
            node.inputs['Strength'].default_value = HDRI_BASE_STRENGTH * (2.0 ** exposure)
