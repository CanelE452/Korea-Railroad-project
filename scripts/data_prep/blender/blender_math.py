"""좌표 변환, 투영, 쿼터니언 등 수학 헬퍼."""

import numpy as np


def euler_to_rotation_matrix(euler_deg):
    """XYZ intrinsic Euler angles (degrees) -> 3x3 rotation matrix."""
    rx, ry, rz = np.radians(euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_quat_xyzw(R):
    """3x3 rotation matrix -> [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def rotation_matrix_to_euler_deg(R):
    """3x3 rotation matrix -> (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))


def build_view_matrix(cam_pos, look_at_target, up=(0, 0, 1)):
    """cam_pos + look_at -> world-to-camera (R, t). OpenCV convention."""
    cam_pos = np.array(cam_pos, dtype=np.float64)
    target = np.array(look_at_target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(forward, up)
        norm_r = np.linalg.norm(right)
    right = right / norm_r
    cam_up = np.cross(right, forward)

    R_w2c = np.array([right, -cam_up, forward], dtype=np.float64)
    t_w2c = -R_w2c @ cam_pos
    return R_w2c, t_w2c


def canonical_corners_yup(bbox_min, bbox_max):
    """Canonical bbox (Y=UP) -> 8 DOPE corners.
    Order: 0=FTR, 1=FTL, 2=FBL, 3=FBR, 4=RTR, 5=RTL, 6=RBL, 7=RBR."""
    mn, mx = np.array(bbox_min), np.array(bbox_max)
    return np.array([
        [mn[0], mx[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mn[0], mn[1], mn[2]],
    ])


def yup_to_zup(pts):
    """Y=UP -> Blender Z=UP coordinate conversion."""
    out = np.zeros_like(pts)
    out[..., 0] = pts[..., 0]
    out[..., 1] = -pts[..., 2]
    out[..., 2] = pts[..., 1]
    return out
