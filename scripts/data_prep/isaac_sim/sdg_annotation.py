import json
import numpy as np
from sdg_math import (euler_to_rotation_matrix, rotation_matrix_to_quat_xyzw,
                       rotation_matrix_to_euler_deg, build_view_matrix,
                       _canonical_corners)


def _compute_visibility(cam_pos, look_at_target, K, image_width, image_height,
                        bbox_min, bbox_max, pallet_pos, pallet_rot_deg, scale,
                        R_canonical=None):
    if R_canonical is not None:
        cbmin, cbmax = np.array(bbox_min), np.array(bbox_max)
        corners_c = _canonical_corners(cbmin, cbmax)
        centroid_c = (cbmin + cbmax) / 2.0
        R_pallet = euler_to_rotation_matrix(pallet_rot_deg)
        R_random = R_pallet @ R_canonical.T
        corners_world = (scale * (R_random @ corners_c.T)).T + np.array(pallet_pos)
        centroid_world = scale * (R_random @ centroid_c) + np.array(pallet_pos)
    else:
        bmn, bmx = np.array(bbox_min), np.array(bbox_max)
        corners_raw = np.array([
            [bmx[0], bmn[1], bmx[2]], [bmn[0], bmn[1], bmx[2]],
            [bmn[0], bmx[1], bmx[2]], [bmx[0], bmx[1], bmx[2]],
            [bmx[0], bmn[1], bmn[2]], [bmn[0], bmn[1], bmn[2]],
            [bmn[0], bmx[1], bmn[2]], [bmx[0], bmx[1], bmn[2]],
        ])
        centroid_c = (bmn + bmx) / 2.0
        R_pallet = euler_to_rotation_matrix(pallet_rot_deg)
        corners_world = (scale * (R_pallet @ corners_raw.T)).T + np.array(pallet_pos)
        centroid_world = scale * (R_pallet @ centroid_c) + np.array(pallet_pos)
    points_3d = np.vstack([corners_world, centroid_world[np.newaxis, :]])
    R_w2c, t_w2c = build_view_matrix(cam_pos, look_at_target)
    pts_cam = (R_w2c @ points_3d.T).T + t_w2c
    projected = (K @ pts_cam.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < image_width) &
                (uv[:, 1] >= 0) & (uv[:, 1] < image_height))
    return float(in_frame.sum()) / 9.0


def write_ndds_json(filepath, cam_pos, look_at_target,
                    K, image_width, image_height,
                    bbox_min, bbox_max, pallet_pos, pallet_rot_deg, scale,
                    R_canonical=None):
    R_pallet = euler_to_rotation_matrix(pallet_rot_deg)

    if R_canonical is not None:
        # Canonical bbox 방식: corners는 canonical 공간, pose는 R_random만
        cbmin, cbmax = np.array(bbox_min), np.array(bbox_max)
        corners_c = _canonical_corners(cbmin, cbmax)
        centroid_c = (cbmin + cbmax) / 2.0
        R_random = R_pallet @ R_canonical.T
        corners_world = (scale * (R_random @ corners_c.T)).T + np.array(pallet_pos)
        centroid_world = scale * (R_random @ centroid_c) + np.array(pallet_pos)
        R_for_pose = R_random
    else:
        bmn, bmx = np.array(bbox_min), np.array(bbox_max)
        corners_c = _canonical_corners(bmn, bmx)
        centroid_c = (bmn + bmx) / 2.0
        corners_world = (scale * (R_pallet @ corners_c.T)).T + np.array(pallet_pos)
        centroid_world = scale * (R_pallet @ centroid_c) + np.array(pallet_pos)
        R_for_pose = R_pallet

    points_3d = np.vstack([corners_world, centroid_world[np.newaxis, :]])

    R_w2c, t_w2c = build_view_matrix(cam_pos, look_at_target)

    pts_cam = (R_w2c @ points_3d.T).T + t_w2c
    projected = (K @ pts_cam.T).T
    uv = projected[:, :2] / projected[:, 2:3]

    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < image_width) &
                (uv[:, 1] >= 0) & (uv[:, 1] < image_height))
    visibility = float(in_frame.sum()) / 9.0

    R_obj_cam = R_w2c @ R_for_pose
    t_obj_cam = R_w2c @ centroid_world + t_w2c

    quat_xyzw = rotation_matrix_to_quat_xyzw(R_obj_cam)
    pitch, yaw, roll = rotation_matrix_to_euler_deg(R_obj_cam)

    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = R_obj_cam
    pose_4x4[:3, 3] = t_obj_cam

    data = {
        "camera_data": {
            "width": image_width,
            "height": image_height,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            },
            "location_worldframe": [float(v) for v in cam_pos],
        },
        "objects": [{
            "class": "pallet",
            "visibility": visibility,
            "location": [float(v) for v in t_obj_cam],
            "quaternion_xyzw": quat_xyzw,
            "euler_angles": {"pitch": pitch, "yaw": yaw, "roll": roll},
            "pose_transform": pose_4x4.tolist(),
            "projected_cuboid_centroid": [float(uv[8, 0]), float(uv[8, 1])],
            "projected_cuboid": [[float(uv[k, 0]), float(uv[k, 1])] for k in range(8)],
            "cuboid": [[float(points_3d[k, 0]), float(points_3d[k, 1]), float(points_3d[k, 2])] for k in range(8)],
        }]
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
