"""AprilTag 기반 Real GT 생성 파이프라인.

tag detection → tag pose → pallet GT pose 변환 → NDDS 포맷 저장 → overlay 검증

사용법:
    # 단일 이미지 파일럿 (overlay 확인)
    python scripts/data_prep/apriltag/apriltag_gt.py \
        --image data/pallet/real_data/real_dev/dev_001_tag.jpg \
        --output data/pallet/real_data/real_dev/dev_001.json \
        --visualize

    # 배치 처리
    python scripts/data_prep/apriltag/apriltag_gt.py \
        --input_dir data/pallet/real_data/real_test_seen \
        --pattern "*_tag.jpg" \
        --visualize

환경: conda activate pallet-pose
의존성: pip install pupil-apriltags (또는 dt-apriltags)
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# AprilTag detector — pupil-apriltags 또는 dt-apriltags
try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_LIB = "pupil"
except ImportError:
    try:
        from dt_apriltags import Detector as AprilTagDetector
        APRILTAG_LIB = "dt"
    except ImportError:
        print("[ERROR] AprilTag library not found.")
        print("  Install: pip install pupil-apriltags")
        print("  Or:      pip install dt-apriltags")
        sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))
from pnp_solver import make_pallet_keypoints_3d, make_camera_matrix


# ============================================================
# Configuration — 촬영 전 확정 필요
# ============================================================

DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.150       # 150mm, 실측 후 수정

# Tag → Pallet rigid transform (4x4)
# AprilTag가 팔레트 상면 중앙에 부착된 경우의 기본값.
# 실측 후 정확한 값으로 교체할 것.
# T_pallet_from_tag: tag 좌표계 → pallet 좌표계
# pallet 좌표계: DOPE convention (X=right, Y=down, Z=forward)
DEFAULT_T_PALLET_FROM_TAG = np.eye(4, dtype=np.float64)
# TODO: 실측 후 이 값을 config에서 로드하도록 변경


def create_detector(tag_family=DEFAULT_TAG_FAMILY):
    """AprilTag detector 생성."""
    if APRILTAG_LIB == "pupil":
        return AprilTagDetector(
            families=tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            decode_sharpening=0.25,
        )
    else:
        return AprilTagDetector(
            families=tag_family,
            nthreads=4,
        )


def detect_tag_pose(image_gray, detector, camera_params, tag_size_m):
    """이미지에서 AprilTag를 감지하고 pose를 추정.

    Args:
        image_gray: grayscale 이미지
        detector: AprilTag detector
        camera_params: (fx, fy, cx, cy)
        tag_size_m: 태그 크기 (m)

    Returns:
        list of dict: [{tag_id, R, t, corners, center, reproj_error}, ...]
    """
    results = detector.detect(
        image_gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_m,
    )

    detections = []
    for r in results:
        # pose_R: (3,3), pose_t: (3,1)
        R = r.pose_R
        t = r.pose_t.flatten()

        # 4x4 transform (tag → camera)
        T_cam_from_tag = np.eye(4)
        T_cam_from_tag[:3, :3] = R
        T_cam_from_tag[:3, 3] = t

        detections.append({
            "tag_id": r.tag_id,
            "tag_family": r.tag_family.decode() if isinstance(r.tag_family, bytes) else str(r.tag_family),
            "R": R,
            "t": t,
            "T_cam_from_tag": T_cam_from_tag,
            "corners": r.corners,       # (4, 2) pixel coords
            "center": r.center,          # (2,) pixel coord
            "decision_margin": r.decision_margin,
            "hamming": r.hamming,
        })

    return detections


def tag_pose_to_pallet_pose(T_cam_from_tag, T_pallet_from_tag):
    """Tag pose → Pallet pose 변환.

    T_cam_from_pallet = T_cam_from_tag @ inv(T_pallet_from_tag)
    """
    T_tag_from_pallet = np.linalg.inv(T_pallet_from_tag)
    T_cam_from_pallet = T_cam_from_tag @ T_tag_from_pallet
    return T_cam_from_pallet


def project_cuboid(T_cam_from_pallet, camera_matrix, pallet_dims=(1.1, 1.1, 0.15)):
    """Pallet pose로 3D cuboid를 2D에 투영.

    Returns:
        projected_cuboid: list of [x, y] (8 corners)
        projected_centroid: [x, y]
    """
    kp3d = make_pallet_keypoints_3d(*pallet_dims)  # (9, 3)
    R = T_cam_from_pallet[:3, :3]
    t = T_cam_from_pallet[:3, 3]

    # 3D → camera frame → 2D
    pts_cam = (R @ kp3d.T).T + t  # (9, 3)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    projected = []
    for p in pts_cam:
        if p[2] <= 0:
            projected.append([-1, -1])
        else:
            u = fx * p[0] / p[2] + cx
            v = fy * p[1] / p[2] + cy
            projected.append([float(u), float(v)])

    return projected[:8], projected[8]


def save_gt_annotation(output_path, image_shape, camera_intrinsics,
                       T_cam_from_pallet, projected_cuboid, projected_centroid,
                       tag_detection, pallet_dims=(1.1, 1.1, 0.15)):
    """NDDS 호환 GT annotation JSON 저장."""
    h, w = image_shape[:2]
    pose_list = T_cam_from_pallet.tolist()

    annotation = {
        "camera_data": {
            "width": w,
            "height": h,
            "intrinsics": {
                "fx": camera_intrinsics[0],
                "fy": camera_intrinsics[1],
                "cx": camera_intrinsics[2],
                "cy": camera_intrinsics[3],
            },
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "pose_transform": pose_list,
            "projected_cuboid": projected_cuboid,
            "projected_cuboid_centroid": projected_centroid,
            "dimensions_m": {
                "width": pallet_dims[0],
                "height": pallet_dims[2],
                "depth": pallet_dims[1],
            },
            "gt_source": "apriltag",
            "tag_id": f"{tag_detection['tag_family']}_{tag_detection['tag_id']}",
            "tag_decision_margin": float(tag_detection["decision_margin"]),
            "tag_hamming": int(tag_detection["hamming"]),
        }],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotation, f, indent=2)
    return annotation


def draw_overlay(image, projected_cuboid, projected_centroid, tag_corners=None):
    """GT cuboid + tag corners overlay."""
    vis = image.copy()

    # Cuboid edges
    EDGES = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    pts = [tuple(int(c) for c in p) for p in projected_cuboid]
    for i, j in EDGES:
        if pts[i][0] >= 0 and pts[j][0] >= 0:
            cv2.line(vis, pts[i], pts[j], (0, 255, 0), 2)

    # Keypoints
    COLORS = [
        (0,0,255),(0,128,255),(0,255,255),(0,255,0),
        (255,255,0),(255,0,0),(255,0,128),(128,0,255),
    ]
    for idx, pt in enumerate(pts):
        if pt[0] >= 0:
            cv2.circle(vis, pt, 5, COLORS[idx % len(COLORS)], -1)
            cv2.putText(vis, str(idx), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[idx % len(COLORS)], 1)

    # Centroid
    cx, cy = int(projected_centroid[0]), int(projected_centroid[1])
    if cx >= 0:
        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
        cv2.putText(vis, "C", (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # AprilTag corners
    if tag_corners is not None:
        for i, corner in enumerate(tag_corners):
            pt = (int(corner[0]), int(corner[1]))
            cv2.circle(vis, pt, 4, (0, 0, 255), -1)
        # Tag outline
        for i in range(4):
            p1 = tuple(int(c) for c in tag_corners[i])
            p2 = tuple(int(c) for c in tag_corners[(i+1) % 4])
            cv2.line(vis, p1, p2, (0, 0, 255), 1)

    return vis


def process_single(image_path, output_json, detector, camera_params,
                   camera_matrix, tag_size_m, T_pallet_from_tag,
                   visualize=False, vis_output=None):
    """단일 이미지 처리."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [SKIP] Cannot read: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detect_tag_pose(gray, detector, camera_params, tag_size_m)

    if not detections:
        print(f"  [SKIP] No tag detected: {image_path}")
        return False

    # 가장 confidence 높은 detection 사용
    det = max(detections, key=lambda d: d["decision_margin"])
    print(f"  Tag {det['tag_id']} detected (margin={det['decision_margin']:.1f})")

    # Tag → Pallet pose
    T_cam_from_pallet = tag_pose_to_pallet_pose(det["T_cam_from_tag"], T_pallet_from_tag)

    # Project cuboid
    cuboid, centroid = project_cuboid(T_cam_from_pallet, camera_matrix)

    # Save annotation
    save_gt_annotation(
        output_json, img.shape, camera_params,
        T_cam_from_pallet, cuboid, centroid, det,
    )
    print(f"  GT saved: {output_json}")

    # Visualization
    if visualize:
        vis = draw_overlay(img, cuboid, centroid, det["corners"])
        vis_path = vis_output or output_json.replace(".json", "_gt_overlay.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  Overlay: {vis_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="AprilTag → Pallet GT Pose")
    parser.add_argument("--image", help="단일 이미지 경로")
    parser.add_argument("--output", help="출력 JSON 경로 (단일 모드)")
    parser.add_argument("--input_dir", help="배치 입력 디렉토리")
    parser.add_argument("--pattern", default="*_tag.jpg", help="배치 파일 패턴")
    parser.add_argument("--tag_family", default=DEFAULT_TAG_FAMILY)
    parser.add_argument("--tag_size", type=float, default=DEFAULT_TAG_SIZE_M,
                        help="Tag 크기 (m)")
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    camera_params = (args.fx, args.fy, args.cx, args.cy)
    camera_matrix = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)

    detector = create_detector(args.tag_family)
    T_pallet_from_tag = DEFAULT_T_PALLET_FROM_TAG

    print(f"AprilTag GT Pipeline")
    print(f"  Tag: {args.tag_family}, size={args.tag_size}m")
    print(f"  Camera: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"  Library: {APRILTAG_LIB}")
    print()

    if args.image:
        # 단일 모드
        output = args.output or args.image.replace("_tag.jpg", ".json")
        process_single(args.image, output, detector, camera_params,
                       camera_matrix, args.tag_size, T_pallet_from_tag,
                       args.visualize)
    elif args.input_dir:
        # 배치 모드
        pattern = os.path.join(args.input_dir, args.pattern)
        files = sorted(glob.glob(pattern))
        print(f"Found {len(files)} files matching {pattern}")

        success = 0
        for f in files:
            basename = os.path.basename(f).replace("_tag.jpg", "")
            output_json = os.path.join(args.input_dir, basename + ".json")
            ok = process_single(f, output_json, detector, camera_params,
                                camera_matrix, args.tag_size, T_pallet_from_tag,
                                args.visualize)
            if ok:
                success += 1

        print(f"\nDone: {success}/{len(files)} processed successfully")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
