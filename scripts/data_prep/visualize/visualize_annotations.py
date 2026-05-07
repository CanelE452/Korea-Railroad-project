"""9-point cuboid keypoint + pose axis 오버레이 시각화.

사용법:
    python scripts/data_prep/visualize/visualize_annotations.py \
        --data_dir data/pallet/test_render_v2 \
        --output_dir data/pallet/test_render_v2/overlay
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


# cuboid 엣지 정의 (NDDS 순서: Front face + Rear face + 연결)
# 0:FrontTopRight 1:FrontTopLeft 2:FrontBottomLeft 3:FrontBottomRight
# 4:RearTopRight  5:RearTopLeft  6:RearBottomLeft  7:RearBottomRight
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # front face (top: Z=max)
    (4, 5), (5, 6), (6, 7), (7, 4),  # rear face (bottom: Z=min)
    (0, 4), (1, 5), (2, 6), (3, 7),  # front-rear 연결
]

CORNER_COLORS = [
    (0, 0, 255),    # 0: FrontTopRight - red
    (0, 128, 255),  # 1: FrontTopLeft - orange
    (0, 255, 255),  # 2: FrontBottomLeft - yellow
    (0, 255, 0),    # 3: FrontBottomRight - green
    (255, 0, 0),    # 4: RearTopRight - blue
    (255, 128, 0),  # 5: RearTopLeft - teal
    (255, 0, 128),  # 6: RearBottomLeft - purple
    (255, 0, 255),  # 7: RearBottomRight - magenta
]
CENTROID_COLOR = (255, 255, 255)  # white


def draw_cuboid(img, pts_2d, centroid_2d):
    """8 cuboid corners + edges + centroid 그리기."""
    h, w = img.shape[:2]

    # edges
    for i, j in CUBOID_EDGES:
        p1 = pts_2d[i]
        p2 = pts_2d[j]
        if all(0 <= p1[k] < [w, h][k] for k in range(2)) or \
           all(0 <= p2[k] < [w, h][k] for k in range(2)):
            cv2.line(img, tuple(p1), tuple(p2), (0, 255, 0), 2)

    # corner points
    for i, pt in enumerate(pts_2d):
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, tuple(pt), 6, CORNER_COLORS[i], -1)
            cv2.circle(img, tuple(pt), 6, (0, 0, 0), 1)
            cv2.putText(img, str(i), (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # centroid
    cx, cy = centroid_2d
    if 0 <= cx < w and 0 <= cy < h:
        cv2.circle(img, (cx, cy), 8, CENTROID_COLOR, -1)
        cv2.circle(img, (cx, cy), 8, (0, 0, 0), 2)
        cv2.putText(img, "C", (cx + 10, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_pose_axes(img, obj, K):
    """pose_transform에서 XYZ 축을 이미지에 그리기."""
    h, w = img.shape[:2]
    pose = np.array(obj["pose_transform"])
    R = pose[:3, :3]
    t = pose[:3, 3]

    K_mat = np.array([
        [K["fx"], 0, K["cx"]],
        [0, K["fy"], K["cy"]],
        [0, 0, 1],
    ])

    # 원점 (centroid) 투영
    origin_px = K_mat @ t
    origin_px = (origin_px[:2] / origin_px[2]).astype(int)

    # 축 길이 (물체 크기에 비례)
    axis_len = 0.3

    axis_colors = [
        (0, 0, 255),   # X: red
        (0, 255, 0),   # Y: green
        (255, 0, 0),   # Z: blue
    ]
    axis_labels = ["X", "Y", "Z"]

    for i in range(3):
        direction = R[:, i] * axis_len
        end_3d = t + direction
        end_px = K_mat @ end_3d
        end_px = (end_px[:2] / end_px[2]).astype(int)

        if 0 <= origin_px[0] < w and 0 <= origin_px[1] < h:
            cv2.arrowedLine(img, tuple(origin_px), tuple(end_px),
                            axis_colors[i], 3, tipLength=0.15)
            cv2.putText(img, axis_labels[i], (end_px[0] + 5, end_px[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_colors[i], 2)


def draw_info_text(img, obj):
    """euler angles + visibility 정보 표시."""
    euler = obj["euler_angles"]
    vis = obj["visibility"]
    lines = [
        f"pitch: {euler['pitch']:.1f}",
        f"yaw:   {euler['yaw']:.1f}",
        f"roll:  {euler['roll']:.1f}",
        f"vis:   {vis:.2f}",
    ]
    y0 = 25
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)


def process_frame(rgb_path, json_path, output_path):
    """한 프레임 처리."""
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"  [SKIP] cannot read {rgb_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    K = data["camera_data"]["intrinsics"]

    for obj in data["objects"]:
        cuboid = obj["projected_cuboid"]
        centroid = obj["projected_cuboid_centroid"]

        pts_2d = np.array(cuboid, dtype=int)
        centroid_2d = (int(centroid[0]), int(centroid[1]))

        draw_cuboid(img, pts_2d, centroid_2d)
        draw_pose_axes(img, obj, K)
        draw_info_text(img, obj)

    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "overlay")
    os.makedirs(args.output_dir, exist_ok=True)

    # PNG + 동일 basename JSON 매칭 (rgb_XXXX.png 또는 XXXXXX.png 모두 지원)
    rgb_files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
    # overlay 디렉토리의 파일 제외
    rgb_files = [f for f in rgb_files if "overlay" not in os.path.basename(f)]
    print(f"Found {len(rgb_files)} PNG images")

    for rgb_path in rgb_files:
        basename = os.path.splitext(os.path.basename(rgb_path))[0]
        # rgb_0000 -> 000000, 또는 000000 -> 000000
        if basename.startswith("rgb_"):
            json_name = f"{int(basename.replace('rgb_', '')):06d}.json"
        else:
            json_name = basename + ".json"
        json_path = os.path.join(args.data_dir, json_name)

        if not os.path.exists(json_path):
            print(f"  [SKIP] no JSON for {basename}")
            continue

        output_path = os.path.join(args.output_dir, f"overlay_{basename}.png")
        process_frame(rgb_path, json_path, output_path)
        print(f"  {basename} -> {os.path.basename(output_path)}")

    print(f"\nDone! {len(rgb_files)} overlays -> {args.output_dir}")


if __name__ == "__main__":
    main()
