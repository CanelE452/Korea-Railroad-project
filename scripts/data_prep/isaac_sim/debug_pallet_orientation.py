"""각 팔레트 모델을 개별 렌더링하여 방향/축 확인용 진단 스크립트."""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

USD_DIR = os.path.join(PROJECT_ROOT, "data", "pallet", "models_usd")
usd_files = sorted([f for f in os.listdir(USD_DIR) if f.endswith(".usd")])

print("=== Pallet Model Orientation Diagnosis ===\n")

from pxr import Gf, Usd, UsdGeom

for usd_file in usd_files:
    usd_path = os.path.join(USD_DIR, usd_file)
    stage = Usd.Stage.Open(usd_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    rng = bbox.ComputeAlignedRange()
    mn = rng.GetMin()
    mx = rng.GetMax()
    dims = [mx[i] - mn[i] for i in range(3)]
    min_idx = dims.index(min(dims))

    print(f"--- {usd_file} ---")
    print(f"  bbox min: ({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})")
    print(f"  bbox max: ({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})")
    print(f"  dims: ({dims[0]:.4f}, {dims[1]:.4f}, {dims[2]:.4f})")
    print(f"  thin axis: {'XYZ'[min_idx]} (idx={min_idx})")

    # 메시 노멀 분석
    normal_counts = {'+X': 0, '-X': 0, '+Y': 0, '-Y': 0, '+Z': 0, '-Z': 0}
    total_normals = 0
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        normals = mesh.GetNormalsAttr().Get()
        if normals is None:
            continue
        mesh_count += 1
        for n in normals:
            total_normals += 1
            abs_vals = [abs(n[0]), abs(n[1]), abs(n[2])]
            dominant = abs_vals.index(max(abs_vals))
            if abs_vals[dominant] > 0.5:
                sign = '+' if n[dominant] > 0 else '-'
                axis = 'XYZ'[dominant]
                normal_counts[f'{sign}{axis}'] += 1

    print(f"  meshes: {mesh_count}, total normals: {total_normals}")
    print(f"  normal counts: {normal_counts}")

    thin_axis = 'XYZ'[min_idx]
    plus = normal_counts[f'+{thin_axis}']
    minus = normal_counts[f'-{thin_axis}']
    print(f"  thin axis normals: +{thin_axis}={plus}, -{thin_axis}={minus}")
    top_side = f"+{thin_axis}" if plus >= minus else f"-{thin_axis}"
    print(f"  => top side (more normals): {top_side}")

    # 회전 후보
    if min_idx == 2:
        print(f"  => Z-thin: no rotation needed, base_rot=(0,0,0)")
    elif min_idx == 1:
        if plus >= minus:
            print(f"  => Y-thin, top=+Y: base_rot=(90,0,0) [+Y → +Z]")
        else:
            print(f"  => Y-thin, top=-Y: base_rot=(-90,0,0) [-Y → +Z]")
    else:
        if plus >= minus:
            print(f"  => X-thin, top=+X: base_rot=(0,-90,0) [+X → +Z]")
        else:
            print(f"  => X-thin, top=-X: base_rot=(0,90,0) [-X → +Z]")
    print()
