"""키포인트 규칙 자동 검증 스크립트.

Guide convention (pallet lying flat, Y=UP):
  0→1 = +X (medium, ~1.0m)
  0→3 = -Y (height, ~0.15m)
  0→4 = -Z (long,   ~1.2m)

검증 항목:
1. Edge lengths: 0→1(medium), 0→3(height), 0→4(long)
2. Orthogonality: 3개 주축 직교성
3. Parallel edges: 대변 길이 일치
4. Right-hand rule: cross(0→1, 0→3) parallel to 0→4
5. Rotation matrix: det=1, orthogonality
6. Cross-model consistency
"""
import json
import numpy as np
import sys
import os
import glob


def build_view_matrix(cam_pos, look_at_target, up=(0, 0, 1)):
    cam_pos = np.array(cam_pos, dtype=np.float64)
    target = np.array(look_at_target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R_w2c = np.array([right, -cam_up, forward], dtype=np.float64)
    t_w2c = -R_w2c @ cam_pos
    return R_w2c, t_w2c


def load_frame(json_path):
    with open(json_path) as f:
        data = json.load(f)
    obj = data["objects"][0]
    cuboid_world = np.array(obj["cuboid"])
    projected = np.array(obj["projected_cuboid"])
    centroid_2d = np.array(obj["projected_cuboid_centroid"])
    R_obj_cam = np.array(obj["pose_transform"])[:3, :3]
    t_obj_cam = np.array(obj["pose_transform"])[:3, 3]
    cam_pos = np.array(data["camera_data"]["location_worldframe"])
    return cuboid_world, projected, centroid_2d, R_obj_cam, t_obj_cam, cam_pos


def verify_single(json_path, model_name=""):
    cuboid, projected, centroid_2d, R_obj_cam, t_obj_cam, cam_pos = load_frame(json_path)
    errors = []
    warnings = []

    # === 1. Edge lengths ===
    e01 = np.linalg.norm(cuboid[1] - cuboid[0])  # medium (~1.0m)
    e03 = np.linalg.norm(cuboid[3] - cuboid[0])  # height (~0.15m)
    e04 = np.linalg.norm(cuboid[4] - cuboid[0])  # long   (~1.2m)

    # height(0→3) should be smallest
    if e03 > e01:
        errors.append(f"0->3({e03:.3f}) > 0->1({e01:.3f}): height > medium")
    if e03 > e04:
        errors.append(f"0->3({e03:.3f}) > 0->4({e04:.3f}): height > long")
    # long(0→4) should be >= medium(0→1)
    if e01 > e04 * 1.15:
        warnings.append(f"0->1({e01:.3f}) > 0->4({e04:.3f})*1.15: medium > long")

    # === 2. Orthogonality ===
    v01 = cuboid[1] - cuboid[0]
    v03 = cuboid[3] - cuboid[0]
    v04 = cuboid[4] - cuboid[0]
    for name, va, vb in [("01-03", v01, v03), ("01-04", v01, v04), ("03-04", v03, v04)]:
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if abs(angle - 90) > 3:
            errors.append(f"edges {name}: angle={angle:.1f} (need ~90)")

    # === 3. Parallel edges (opposite edges same length) ===
    pairs = [
        ((0, 1), (3, 2)), ((0, 1), (4, 5)), ((0, 1), (7, 6)),
        ((0, 3), (1, 2)), ((0, 3), (4, 7)),
        ((0, 4), (1, 5)), ((0, 4), (3, 7)),
    ]
    for (a, b), (c, d) in pairs:
        la = np.linalg.norm(cuboid[b] - cuboid[a])
        lb = np.linalg.norm(cuboid[d] - cuboid[c])
        if la > 0 and abs(la - lb) / la > 0.03:
            errors.append(f"edge {a}->{b}({la:.3f}) != {c}->{d}({lb:.3f})")

    # === 4. Right-hand rule ===
    x_world = v01 / np.linalg.norm(v01)  # 0→1 = +X (medium)
    y_world = v03 / np.linalg.norm(v03)  # 0→3 = -Y (height)
    z_world = v04 / np.linalg.norm(v04)  # 0→4 = -Z (long)

    cross_01_03 = np.cross(v01, v03)
    cross_dir = cross_01_03 / np.linalg.norm(cross_01_03)
    rhr_dot = np.dot(cross_dir, z_world)
    if abs(rhr_dot) < 0.95:
        errors.append(f"Right-hand rule: cross(01,03).z_world={rhr_dot:.3f}")

    # === 5. Rotation matrix validity ===
    det_R = np.linalg.det(R_obj_cam)
    if abs(det_R - 1.0) > 0.01:
        errors.append(f"det(R)={det_R:.4f} (should be 1.0)")
    orth_err = np.max(np.abs(R_obj_cam @ R_obj_cam.T - np.eye(3)))
    if orth_err > 0.01:
        errors.append(f"R orthogonality error={orth_err:.4f}")

    return {
        "model": model_name,
        "file": os.path.basename(json_path),
        "edges": {"01_med": e01, "03_height": e03, "04_long": e04},
        "rhr_dot": rhr_dot,
        "errors": errors,
        "warnings": warnings,
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pallet/test_canonical"
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        print(f"[ERROR] No JSON files in {data_dir}")
        sys.exit(1)

    models = ["scene.usd", "scene_1.usd", "scene_2.usd", "scene_3.usd"]
    all_pass = True

    print(f"=== Keypoint Verification (Y=UP convention) ===")
    print(f"Data: {data_dir} ({len(json_files)} files)\n")

    results = []
    for i, jf in enumerate(json_files):
        model = models[i % len(models)] if i < len(models) else f"model_{i}"
        result = verify_single(jf, model)
        results.append(result)

        e = result["edges"]
        status = "PASS" if not result["errors"] else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"[{status}] {result['model']} ({result['file']})")
        print(f"  0->1={e['01_med']:.3f}m(med) 0->3={e['03_height']:.3f}m(h) 0->4={e['04_long']:.3f}m(long) RHR={result['rhr_dot']:.3f}")
        for err in result["errors"]:
            print(f"  [ERROR] {err}")
        for warn in result["warnings"]:
            print(f"  [WARN]  {warn}")
        print()

    # Cross-model consistency
    print("=== Cross-Model Consistency ===")
    long_edges = [r["edges"]["04_long"] for r in results]
    med_edges = [r["edges"]["01_med"] for r in results]
    h_edges = [r["edges"]["03_height"] for r in results]

    print(f"  med    edges (0->1): {[f'{e:.3f}' for e in med_edges]}  std={np.std(med_edges):.4f}")
    print(f"  height edges (0->3): {[f'{e:.3f}' for e in h_edges]}  std={np.std(h_edges):.4f}")
    print(f"  long   edges (0->4): {[f'{e:.3f}' for e in long_edges]}  std={np.std(long_edges):.4f}")

    if np.std(long_edges) > 0.01:
        print(f"  [WARN] Long edge variance too high (models not scaled consistently)")

    print(f"\n{'='*50}")
    if all_pass:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: SOME FAILURES - see errors above")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
